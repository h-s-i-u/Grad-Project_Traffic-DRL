"""Routing policies, framed as point-to-point navigation for many vehicles.

This is the heart of the strategy difference. The proposal's problem is:
many vehicles route from origin to destination, and naive "everyone takes the
predicted-fastest road" behaviour creates the herding effect (羊群效應). We model
exactly that, as an ablation:

  1. static            - free-flow shortest path, prediction ignored        (baseline 1)
  2. prediction_greedy - predicted-fastest path, per vehicle, NO coordination (baseline 2/3, HERDING)
  3. load_aware        - incremental load-aware assignment, no global penalty (ablation: coordination only)
  4. global_penalty    - load-aware assignment WITH proposal eq. (4)         (our method)

A policy returns one node-path per vehicle (or None if unreachable).
"""
import abc
from collections import namedtuple

import networkx as nx
import numpy as np

import config as C


def _closed_weight(closure, attr):
    """Dijkstra weight that refuses to price a closed edge.

    Returning None (not inf) is load-bearing. networkx treats None as "this edge
    cannot be used" and raises NetworkXNoPath when no alternative exists, which is
    the honest outcome. With inf it would still return a path THROUGH the closed
    road -- infinite cost, but downstream nothing checks the cost, so the trip would
    be scored as delivered along a road that is shut.
    """
    def w(u, v, d):
        return None if closure.blocked(u, v) else d[attr]
    return w


def _route_fixed(g, demand, weight, closure=None):
    """Shortest path for every (o, d) under a FIXED edge weight.

    No load feedback: every vehicle sees the same costs, so they pile onto the
    same links. This is precisely the mechanism behind the herding effect.

    With a `closure`, vehicles dispatched before its cutoff keep routing on the open
    network -- they were already on their way -- and only later ones are re-routed.
    That split is the entire point of S3: everyone downstream of the incident gets
    handed the same new fastest path at the same moment.
    """
    paths = []
    cut = closure.cutoff(len(demand)) if closure is not None else len(demand)
    w_closed = _closed_weight(closure, weight) if closure is not None else None
    for k, (o, d) in enumerate(demand):
        try:
            paths.append(nx.shortest_path(g, o, d,
                                          weight=w_closed if k >= cut else weight))
        except nx.NetworkXNoPath:
            paths.append(None)
    return paths


def policy_static(g, demand, closure=None):
    """Baseline 1: free-flow shortest path (Dijkstra), prediction ignored."""
    return _route_fixed(g, demand, "t0", closure)


def policy_prediction_greedy(g, demand, weight="tpred", closure=None):
    """Predicted-fastest path, per vehicle, uncoordinated — the herding mechanism.

    `weight` picks WHOSE prediction, giving the proposal's three separate baselines:
        "tpred"        -> (4) STGCN+STGAT ensemble  = the herding baseline
        "tpred_stgcn"  -> (2) pure STGCN + Dijkstra
        "tpred_stgat"  -> (3) pure STGAT + Dijkstra
    The variant attributes only exist if the graph was built with `speed_variants`
    (see network.build_graph); routing on a missing attribute would silently fall
    back to weight=1, so callers must check first — run_compare does.
    """
    return _route_fixed(g, demand, weight, closure)


def _bpr(t0, load, cap):
    """Bureau-of-Public-Roads volume-delay function."""
    return t0 * (1.0 + C.BPR_A * (load / cap) ** C.BPR_B)


def policy_incremental(g, demand, use_penalty, n_batches=None, closure=None):
    """Incremental load-aware assignment.

    Vehicles are assigned in batches; before each batch the edge cost is
    recomputed from the load left by earlier batches, so later vehicles are
    steered onto alternatives. This is the transparent stand-in for a PPO agent
    trained with the global-penalty reward (see README) and the evaluation
    harness that agent will later be scored against.

    Generalized edge cost realizes proposal eq. (4):
        cost_e = alpha * t_e(load)
               + lambda1 * t_ref * scale * max(0, rho_e - rho_th)^2   # saturation overflow
               + lambda2 * t_ref * scale * max(0, rho_e - mean_rho)   # per-edge surrogate of Var(rho)

    With use_penalty=False the lambda terms drop out, leaving plain congestion-aware
    assignment (the ablation that isolates the global penalty's marginal effect).
    """
    n_batches = n_batches or C.N_BATCHES
    edges = list(g.edges())
    t0 = {e: g.edges[e]["t0"] for e in edges}
    cap = {e: g.edges[e]["cap"] for e in edges}
    load = {e: 0.0 for e in edges}
    t_ref = float(np.mean([t0[e] for e in edges])) if edges else 1.0
    # Tested per VEHICLE, not per batch: a batch can straddle the closure, and these
    # policies are the oracle -- rounding the incident to the nearest batch boundary
    # would give (5)(6) a few vehicles' worth of foresight that (7) does not get.
    cut = closure.cutoff(len(demand)) if closure is not None else len(demand)
    w_closed = _closed_weight(closure, "cost") if closure is not None else None

    paths = [None] * len(demand)
    for batch in np.array_split(np.arange(len(demand)), n_batches):
        rho = {e: load[e] / cap[e] for e in edges}
        mean_rho = float(np.mean([rho[e] for e in edges])) if edges else 0.0
        for e in edges:
            cost = C.ALPHA * _bpr(t0[e], load[e], cap[e])
            if use_penalty:
                overflow = max(0.0, rho[e] - C.RHO_THRESHOLD) ** 2
                spread = max(0.0, rho[e] - mean_rho)
                cost += t_ref * C.PENALTY_SCALE * (
                    C.LAMBDA_SAT * overflow + C.LAMBDA_VAR * spread
                )
            g.edges[e]["cost"] = cost

        for k in batch:
            o, d = demand[k]
            try:
                p = nx.shortest_path(g, o, d,
                                     weight=w_closed if k >= cut else "cost")
            except nx.NetworkXNoPath:
                p = None
            paths[k] = p
            if p:
                for e in zip(p[:-1], p[1:]):
                    load[e] += 1.0
    return paths


def policy_load_aware(g, demand, closure=None):
    """Ablation: coordination via congestion feedback only (no global penalty)."""
    return policy_incremental(g, demand, use_penalty=False, closure=closure)


def policy_global_penalty(g, demand, closure=None):
    """Our method: congestion feedback + proposal eq. (4) global penalty."""
    return policy_incremental(g, demand, use_penalty=True, closure=closure)


# ===================================================================
# DRL / PPO scaffold   (proposal §4.4: POMDP + PPO + global-penalty reward)
# -------------------------------------------------------------------
# The policies above are analytic. The proposal's decision module is a *learned*
# PPO agent (Residual E-GAT actor + pointer decoder) whose action is "pick the next
# neighbour node" and whose reward is eq. (4). This section scaffolds that interface
# so a trained agent slots straight into run_compare with no other changes:
#
#   RoutingEnv        - the POMDP the agent acts in / trains against (reward = eq.4)
#   DRLRoutingAgent   - the agent contract the rollout depends on (.act)
#   GreedyOracleAgent - analytic placeholder so the whole slot runs *today*
#   EGATActorCritic   - trainable Residual E-GAT actor-critic (PyTorch; PPO-ready)
#   PPOTrainer        - PPO loop skeleton (clipped surrogate, eq.5)
#   policy_drl        - rolls an agent out to produce paths, like the other policies
#
# `policy_global_penalty` is the analytic ORACLE the agent should learn to match or
# beat; this env's per-step reward is exactly that eq.(4) generalized cost.
# ===================================================================

# Per-candidate edge features for the learned policy:
#   [t0, tpred, rho, is_dest, dist_to_dest, mean_rho, rho-mean_rho, overflow]
# The last three expose eq. (4)'s load-spread / saturation signal, so the agent can
# learn to even out load (suppress herding / lower Gini), not just shorten trips.
EDGE_FEATURE_DIM = 8
_BIG = 1.0e3  # finite stand-in for "unreachable" distance in network features

# One decision point: at `node`, heading to `dest`, choose one of `neighbors`.
#   feats : [k, EDGE_FEATURE_DIM] per-candidate features for a learned policy
#   gcost : [k] eq.(4) generalized cost of each candidate edge (for the oracle)
#   to_go : [k] shortest free-flow time from each candidate to dest (A*-style hint)
Observation = namedtuple(
    "Observation",
    ["node", "dest", "neighbors", "feats", "gcost", "to_go",
     "node_dyn", "edge_rho", "vehicle"])   # last 3: per-vehicle E-GAT encoder inputs


class RoutingEnv:
    """Sequential multi-vehicle routing POMDP (proposal §4.4).

    State  s_t = (G, X_pred, rho_t, o_t): graph + predicted edge times + current
                 saturation + (current node, destination).
    Action a_t in N(v_t): pick a neighbour of the current node.
    Reward r_t = -(eq.4 generalized cost of the chosen edge):
                 -(alpha*t_e(load) + lambda1*t_ref*scale*max(0,rho-rho_th)^2
                                   + lambda2*t_ref*scale*max(0,rho-mean_rho)).

    Vehicles are routed one after another and the load PERSISTS across them, so the
    penalty couples vehicles — that coupling is what suppresses the herding effect.
    One episode = routing every vehicle in `demand`.

    Optional training-time shaping (default off -> reward stays faithful to eq.4):
    `arrival_bonus` is added when a vehicle reaches its destination and
    `fail_penalty` subtracted if it gets stuck. Pass `demand_fn` (a zero-arg callable
    returning a demand list) to resample demand on every reset().

    ⚠ Both must be scaled to a whole TRIP, not to one edge. Every step costs about
    one edge time, so a trip of H hops accumulates roughly -H*t_ref; if
    arrival_bonus + fail_penalty < that, the highest-return behaviour is to end the
    episode early by walking into a corner (`_valid_neighbors` excludes visited
    nodes, so this is easy). At METR-LA's ~1.5 hops the old edge-scaled defaults
    (2*t_ref, 5*t_ref) were comfortably above the trip cost; on the 47-hop Taichung
    arena they were ~7x too small, and a 20-iteration agent duly learned to abandon
    289 of 300 vehicles while posting ATT -74%. train_drl.py now derives the scale
    from sampled shortest-path times.

    `shaping_gamma` enables potential-based reward shaping (Ng, Harada & Russell
    1999) with Phi(s) = -free-flow time from s to the destination:
        F(s, s') = gamma * Phi(s') - Phi(s)
    This is dense per-step feedback on whether the vehicle is getting closer, which
    is what makes credit assignment over ~47 steps tractable. It does NOT replace the
    terminal scaling above: potential-based shaping is policy-invariant by
    construction, so it cannot fix an objective whose optimum is already "give up".
    """

    def __init__(self, g, demand=None, use_penalty=True, max_hops=60,
                 arrival_bonus=0.0, fail_penalty=0.0, demand_fn=None,
                 shaping_gamma=None, reward_scale=1.0, togo_refresh=0, closure=None):
        self.g = g
        self.demand_fn = demand_fn
        self.demand = list(demand) if demand is not None else None
        self.use_penalty = use_penalty
        self.max_hops = max_hops
        self.arrival_bonus = arrival_bonus
        self.fail_penalty = fail_penalty
        self.shaping_gamma = shaping_gamma
        # Divides every reward. Scaling by a positive constant cannot change the
        # optimal policy, but it decides whether PPO can be optimised at all: on the
        # Taichung arena a trip costs ~414 s and a failed one another ~2,071, so raw
        # seconds gave episode returns near -640,000 and a value loss of 4.3e6 against
        # a policy loss of 0.24. `clip_grad_norm_` rescales by the TOTAL norm, so the
        # actor's gradient was being crushed by the critic's. Pass the mean trip time
        # here to put rewards in units of "one trip" and keep both terms comparable.
        self.reward_scale = float(reward_scale) if reward_scale else 1.0
        self.succ = {v: list(g.successors(v)) for v in g.nodes()}
        self._t0 = {e: g.edges[e]["t0"] for e in g.edges()}
        self._tpred = {e: g.edges[e]["tpred"] for e in g.edges()}
        self._cap = {e: float(g.edges[e].get("cap", C.EDGE_CAPACITY)) for e in g.edges()}  # per-edge capacity
        self.n_edges = max(1, g.number_of_edges())
        self.N = g.number_of_nodes()
        self.edge_list = list(g.edges())          # canonical edge order (matches the agent's edge_index)
        self.t_ref = float(np.mean(list(self._t0.values()))) if self._t0 else 1.0
        self._rg = g.reverse(copy=False)          # for shortest free-flow time to dest
        # 0 = free-flow to-go (original behaviour); K > 0 = eq.4-cost to-go
        # recomputed every K vehicles. See _dist_to_dest for why this matters.
        self.togo_refresh = int(togo_refresh or 0)
        self._dist_cache = {}
        # S3 arterial closure (closure.py). MASKED, never removed from g: the actor's
        # edge_index/edge_static are state-dict buffers sized from this graph, so a
        # graph with fewer edges cannot load a checkpoint at all. Masking also keeps
        # the load wave-1 vehicles left on the closed road inside edge_rho, which is
        # what physically happened.
        self.closure = closure
        self._closed_now = False
        self._enc = None                          # per-vehicle (node_dyn, edge_rho) for the E-GAT encoder
        self.reset()

    # --- load / saturation bookkeeping ---
    def _rho(self, e):
        return self.load.get(e, 0.0) / self._cap[e]

    @property
    def _mean_rho(self):
        # mean of per-edge saturation; kept as a running sum so this stays O(1)
        return self._rho_sum / self.n_edges

    def _gcost(self, u, v):
        """eq.(4) marginal cost of traversing (u, v) at the current load."""
        e = (u, v)
        cost = C.ALPHA * _bpr(self._t0[e], self.load.get(e, 0.0), self._cap[e])
        if self.use_penalty:
            rho = self._rho(e)
            overflow = max(0.0, rho - C.RHO_THRESHOLD) ** 2
            spread = max(0.0, rho - self._mean_rho)
            cost += self.t_ref * C.PENALTY_SCALE * (
                C.LAMBDA_SAT * overflow + C.LAMBDA_VAR * spread)
        return cost

    def _dist_to_dest(self, dest):
        """Estimated cost from every node to `dest`, cached per destination.

        With `togo_refresh == 0` this is the FREE-FLOW time, i.e. it ignores the load
        that is already on the network. That makes the agent's only sense of "how far
        is left" congestion-blind: it can read rho on the candidate edge in front of
        it, but not on anything beyond. Measured consequence -- the agent's routes are
        SHORTER than the oracle's (detour 1.096x vs 1.131x) and yet 13.6% slower on
        the same trips, which is what picking into congestion looks like.

        With `togo_refresh = K` the estimate uses _gcost -- the same eq.4 cost the
        reward is built from -- and is recomputed every K vehicles. That gives the
        agent a cheap approximation of the lookahead the oracle gets from running a
        full Dijkstra per vehicle.

        Cost: one Dijkstra per DESTINATION per refresh. Under the hotspot scenario
        there are only N_HOTSPOTS (4) destinations, so 800 vehicles at K=25 is 128
        Dijkstras on 1,690 edges -- a fraction of a second per episode.
        """
        cached = self._dist_cache.get(dest)
        if cached is None:
            # A closed road must vanish from the to-go estimate REGARDLESS of
            # togo_refresh: the closure is a topology fact, not a congestion one. Skip
            # this and the agent's "how far is left" -- which feeds feats[:,4], to_go,
            # node_dyn[:,0] AND the potential shaping -- points down a road that no
            # longer exists. That is worse than the congestion-blindness of §13.16,
            # and nothing raises.
            blocked = self.closure.blocked if self._closed_now else None
            if self.togo_refresh:
                # The reversed graph carries edge (v, u) for original (u, v), so the
                # weight callback has to flip its arguments before pricing the edge.
                def w(a, b, _d):
                    return None if (blocked and blocked(b, a)) else self._gcost(b, a)
                d = nx.single_source_dijkstra_path_length(self._rg, dest, weight=w)
            elif blocked:
                def w(a, b, _d):
                    return None if blocked(b, a) else _d["t0"]
                d = nx.single_source_dijkstra_path_length(self._rg, dest, weight=w)
            else:
                d = nx.single_source_dijkstra_path_length(self._rg, dest, weight="t0")
            finite = [x for x in d.values() if np.isfinite(x)]
            cached = (d, max(finite) if finite else 1.0)
            self._dist_cache[dest] = cached
        return cached[0]

    def _dist_max(self, dest):
        """Largest finite to-go value, computed once alongside the distances.

        Used to normalise and to clamp unreachable nodes. It has to travel with the
        distances rather than be a constant: once the estimate is congestion-aware its
        scale grows with the load, so a fixed clamp would start truncating real
        values as the network fills up.
        """
        self._dist_to_dest(dest)
        return self._dist_cache[dest][1]

    def _compute_enc_ctx(self):
        """Per-vehicle E-GAT encoder inputs: node features [N,3] and edge rho [E]
        (snapshotted at the start of each vehicle's trip)."""
        dist = self._dist_to_dest(self._dest)
        dmax = self._dist_max(self._dest)
        node_dyn = np.zeros((self.N, 3), dtype=np.float32)
        for v in range(self.N):
            d = dist.get(v, dmax * 2.0)
            node_dyn[v, 0] = min(d, dmax * 2.0) / (dmax + 1e-9)    # dist-to-dest (normalized)
            node_dyn[v, 1] = 1.0 if v == self._dest else 0.0        # is-dest
            outs = self.succ[v]
            if outs:
                s = sum(self.load.get((v, w), 0.0) / self._cap[(v, w)] for w in outs)
                node_dyn[v, 2] = s / len(outs)                        # mean out-saturation
        edge_rho = np.empty(len(self.edge_list), dtype=np.float32)
        for i, e in enumerate(self.edge_list):
            edge_rho[i] = self.load.get(e, 0.0) / self._cap[e]
        return node_dyn, edge_rho

    # --- episode control ---
    def reset(self):
        if self.demand_fn is not None:
            self.demand = list(self.demand_fn())
        if self.demand is None:
            raise ValueError("RoutingEnv requires `demand` or `demand_fn`.")
        self.load = {}
        self.total_load = 0.0
        self._closed_now = False       # the closure has not happened yet at vehicle 0
        self._dist_cache = {}          # ...so any to-go cached under it must go too
        self._rho_sum = 0.0            # running sum of per-edge saturation (for _mean_rho)
        self.paths = [None] * len(self.demand)
        self._vi = -1                 # current vehicle index
        self._cur = self._dest = None
        self._visited = None
        self._hops = 0
        self.done = False
        # Failure-mode counters. "Not served" has three quite different causes and
        # they need different fixes: a dead end means the policy trapped itself (all
        # neighbours already visited), hitting max_hops means it wandered, and a
        # trivial trip was never routable. Reporting one served% hides which.
        self.n_deadend = self.n_maxhops = self.n_trivial = 0
        self._start_next_vehicle()
        return self._observe()

    def _valid_neighbors(self):
        ws = [w for w in self.succ[self._cur] if w not in self._visited]
        if self._closed_now:
            ws = [w for w in ws if not self.closure.blocked(self._cur, w)]
        return ws

    def _start_next_vehicle(self):
        """Advance to the next vehicle that actually has a choice to make."""
        while True:
            self._vi += 1
            # The instant the closure lands, every cached to-go is measured on a
            # network that no longer exists. Checked on the transition rather than
            # every vehicle so the (much larger) post-closure stretch still gets the
            # normal caching.
            if self.closure is not None:
                now = self.closure.active(self._vi, len(self.demand))
                if now != self._closed_now:
                    self._closed_now = now
                    self._dist_cache.clear()
            # Congestion-aware to-go goes stale as load builds, so drop the cache
            # every K vehicles and let the next lookup re-run Dijkstra. With the
            # free-flow estimate (togo_refresh == 0) nothing ever changes, so the
            # cache is kept for the whole episode as before.
            if self.togo_refresh and self._vi % self.togo_refresh == 0:
                self._dist_cache.clear()
            if self._vi >= len(self.demand):
                self.done = True
                self._cur = None
                return
            o, d = self.demand[self._vi]
            self._cur, self._dest = o, d
            self._visited = {o}
            self._hops = 0
            if o == d or not self._valid_neighbors():
                self.paths[self._vi] = None       # trivial or dead-end -> failed trip
                self.n_trivial += 1
                continue
            self.paths[self._vi] = [o]
            self._enc = self._compute_enc_ctx()   # snapshot encoder inputs for this vehicle
            return

    def _observe(self):
        if self.done:
            return None
        nbrs = self._valid_neighbors()
        dist = self._dist_to_dest(self._dest)
        mean_rho = self._mean_rho
        feats = np.zeros((len(nbrs), EDGE_FEATURE_DIM), dtype=np.float32)
        gcost = np.zeros(len(nbrs), dtype=np.float32)
        to_go = np.full(len(nbrs), np.inf, dtype=np.float32)
        for i, w in enumerate(nbrs):
            e = (self._cur, w)
            tg = dist.get(w, np.inf)
            to_go[i] = tg
            gcost[i] = self._gcost(self._cur, w)
            rho = self._rho(e)
            feats[i] = (self._t0[e], self._tpred[e], rho,
                        1.0 if w == self._dest else 0.0, min(tg, _BIG),
                        mean_rho, rho - mean_rho, max(0.0, rho - C.RHO_THRESHOLD))
        node_dyn, edge_rho = self._enc
        return Observation(self._cur, self._dest, nbrs, feats, gcost, to_go,
                           node_dyn, edge_rho, self._vi)

    # ---- beam-search decoding (inference only; training always uses step()) ----
    def _beam_feats(self, cur, nbrs, delta, dist):
        """The same candidate features _observe builds, for an arbitrary beam state.

        `delta` is the beam's OWN load contribution so far. Carrying it is what makes
        width=1 reproduce the greedy rollout exactly: step() adds 1.0 to each edge as
        the vehicle traverses it, so by hop 3 the greedy agent is already looking at
        saturations it raised itself. Scoring every beam against a frozen snapshot
        instead would be cheaper and subtly different, and the equivalence test would
        fail for a reason that has nothing to do with the search.
        """
        rho_sum = self._rho_sum + sum(v / self._cap[e] for e, v in delta.items())
        mean_rho = rho_sum / self.n_edges
        feats = np.zeros((len(nbrs), EDGE_FEATURE_DIM), dtype=np.float32)
        gcost = np.zeros(len(nbrs), dtype=np.float32)
        to_go = np.empty(len(nbrs), dtype=np.float32)
        for i, w in enumerate(nbrs):
            e = (cur, w)
            load = self.load.get(e, 0.0) + delta.get(e, 0.0)
            rho = load / self._cap[e]
            cost = C.ALPHA * _bpr(self._t0[e], load, self._cap[e])
            if self.use_penalty:
                cost += self.t_ref * C.PENALTY_SCALE * (
                    C.LAMBDA_SAT * max(0.0, rho - C.RHO_THRESHOLD) ** 2
                    + C.LAMBDA_VAR * max(0.0, rho - mean_rho))
            gcost[i] = cost
            tg = dist.get(w, np.inf)
            to_go[i] = tg
            feats[i] = (self._t0[e], self._tpred[e], rho,
                        1.0 if w == self._dest else 0.0, min(tg, _BIG),
                        mean_rho, rho - mean_rho, max(0.0, rho - C.RHO_THRESHOLD))
        return feats, gcost, to_go

    def beam_route(self, agent, width, max_hops):
        """Decode the current vehicle with beam search.

        Returns (route, reason, traversed): `route` is the completed path or None,
        `traversed` is what the vehicle actually drove and must be charged for.

        🔴 THOSE TWO DIFFER ON FAILURE, and matching that is not optional. step()
        charges load hop by hop, so a vehicle that drives 40 edges and then traps
        itself leaves all 40 on the network even though its route is recorded as None
        -- the roads were occupied. (metrics.edge_loads works from the returned paths
        and therefore does NOT count them, so the load a failed trip leaves is visible
        to later vehicles' decisions but not to the reported metrics. That asymmetry is
        pre-existing; beam search has to reproduce it or the two decoders diverge for a
        reason unrelated to the search. Measured: discarding it left the environment 81
        edges lighter by vehicle 13, which flipped a candidate whose two options were
        within 0.52 of each other.)

        Greedy decoding commits to the argmax at every hop, so one bad step is
        unrecoverable -- and in this environment "unrecoverable" is literal, because
        _valid_neighbors excludes visited nodes and the vehicle can walk itself into a
        corner. Measured under the S3 closure: 143 of 766 trips dead-end, against 29 on
        the undisturbed network.

        Beam search keeps `width` partial routes and expands all of them, so a branch
        that traps itself simply drops out while the others continue. Every edge is
        still proposed and scored by the policy -- this is a DECODING change, not a
        different decision maker, which is why the base paper (Lei et al. 2022) reports
        greedy and beam side by side rather than treating beam as a separate method.

        Beams are EXPANDED by cumulative log-probability (the policy's own preference)
        and the winner is CHOSEN among completed routes by realised eq.4 cost -- the
        same quantity the reward is built from. Ranking the search itself by cost would
        quietly turn this into a cost-guided search rather than a wider reading of the
        policy.
        """
        dist = self._dist_to_dest(self._dest)
        score = _beam_scorer(agent, self._enc, self._dest)
        live = [{"cur": self._cur, "seen": {self._cur}, "delta": {},
                 "logp": 0.0, "cost": 0.0, "path": [self._cur]}]
        done, hit_cap = [], False
        for _ in range(max_hops):
            pool = []
            for b in live:
                nbrs = [w for w in self.succ[b["cur"]] if w not in b["seen"]]
                if self._closed_now:
                    nbrs = [w for w in nbrs if not self.closure.blocked(b["cur"], w)]
                if not nbrs:
                    continue                       # this branch trapped itself
                feats, gcost, to_go = self._beam_feats(b["cur"], nbrs, b["delta"], dist)
                lp = score(b["cur"], nbrs, feats, gcost, to_go)
                for i, w in enumerate(nbrs):
                    pool.append((b["logp"] + float(lp[i]), b, w, float(gcost[i])))
            if not pool:
                break                              # every branch is stuck
            pool.sort(key=lambda r: -r[0])
            nxt = []
            for logp, b, w, c in pool[:width]:
                e = (b["cur"], w)
                nb = {"cur": w, "seen": b["seen"] | {w},
                      "delta": {**b["delta"], e: b["delta"].get(e, 0.0) + 1.0},
                      "logp": logp, "cost": b["cost"] + c, "path": b["path"] + [w]}
                (done if w == self._dest else nxt).append(nb)
            if done:
                break                              # first completion wins the race
            live = nxt
        else:
            hit_cap = True
        if done:
            best = min(done, key=lambda b: b["cost"])
            return best["path"], "reached", best["path"]
        # Failed. The vehicle still drove somewhere, and step() would have charged for
        # it, so hand back the branch the policy was actually following -- the highest
        # cumulative log-probability among the survivors. At width 1 that is the single
        # trajectory, which is what makes this identical to greedy.
        partial = max(live, key=lambda b: b["logp"])["path"] if live else [self._cur]
        # A dead end wants a wider beam or a stronger fail penalty; wandering wants more
        # hops. One "not served" number cannot say which.
        return None, ("max_hops" if hit_cap else "dead_end"), partial

    def commit(self, route, reason="reached", traversed=None):
        """Record a decoded route and advance. Beam counterpart of step().

        `traversed` is charged to the network; `route` is what gets reported. They are
        the same on success and differ on failure -- see beam_route.
        """
        for e in zip((traversed or [])[:-1], (traversed or [])[1:]):
            self.load[e] = self.load.get(e, 0.0) + 1.0
            self.total_load += 1.0
            self._rho_sum += 1.0 / self._cap[e]
        if route:
            self.paths[self._vi] = route
        else:
            self.paths[self._vi] = None
            if reason == "max_hops":
                self.n_maxhops += 1
            else:
                self.n_deadend += 1
        self._start_next_vehicle()
        return self._observe()

    def step(self, action_index):
        """Apply chosen neighbour (index into obs.neighbors).
        Returns (next_obs, reward, done, info)."""
        nbrs = self._valid_neighbors()
        u, w = self._cur, nbrs[action_index]
        reward = -self._gcost(u, w)
        if self.shaping_gamma is not None:
            # F = gamma*Phi(w) - Phi(u) with Phi = -dist_to_dest; equivalently
            # d(u) - gamma*d(w), i.e. positive for progress toward the destination.
            # Distances come from the same Dijkstra the observation already caches.
            # No special terminal handling: substituting Phi(terminal)=0 would reward
            # getting stuck FAR from the destination by exactly its remaining distance.
            dist = self._dist_to_dest(self._dest)
            # Clamp from the distances themselves, not from a constant: once the
            # estimate is congestion-aware its scale grows with the load, and a fixed
            # max_hops*t_ref ceiling would quietly start truncating real values.
            far = 2.0 * self._dist_max(self._dest)
            du = min(dist.get(u, far), far)
            dw = min(dist.get(w, far), far)
            reward += du - self.shaping_gamma * dw
        e = (u, w)
        self.load[e] = self.load.get(e, 0.0) + 1.0
        self.total_load += 1.0
        self._rho_sum += 1.0 / self._cap[e]
        self._visited.add(w)
        self._cur = w
        self._hops += 1
        self.paths[self._vi].append(w)
        reached = (w == self._dest)
        stuck = (self._hops >= self.max_hops) or (not self._valid_neighbors())
        if reached:
            reward += self.arrival_bonus
        elif stuck:
            reward -= self.fail_penalty
        reward /= self.reward_scale
        info = {"vehicle": self._vi, "reached_dest": reached}
        if reached or stuck:
            if not reached:
                self.paths[self._vi] = None       # failed before reaching dest
                if self._hops >= self.max_hops:
                    self.n_maxhops += 1
                else:
                    self.n_deadend += 1
            self._start_next_vehicle()
        return self._observe(), reward, self.done, info


def _beam_scorer(agent, enc, dest):
    """Per-candidate log-probabilities for beam expansion.

    The encoder output H is computed ONCE per vehicle and shared by every beam, which
    is why widening the beam multiplies only the decoder cost. RoutingEnv snapshots the
    encoder inputs when the vehicle starts (_start_next_vehicle), so H is fixed for the
    whole trip and this is not an approximation.

    Analytic agents have no decoder; they are scored by -(gcost + to_go), the exact
    quantity GreedyOracleAgent takes the argmin of. That keeps beam search testable
    without torch and makes the width=1 equivalence check meaningful for both.
    """
    node_dyn, edge_rho = enc
    if _TORCH_OK and hasattr(agent, "decode"):
        with torch.no_grad():
            H = agent.encode(node_dyn, edge_rho)

        def score(cur, cands, feats, gcost, to_go):
            with torch.no_grad():
                logits, _ = agent.decode(H, cur, dest, cands, feats)
                return torch.log_softmax(logits, dim=-1).cpu().numpy()
        return score

    def score(cur, cands, feats, gcost, to_go):
        # to_go is used RAW, infinities and all. GreedyOracleAgent takes
        # argmin(gcost + to_go) on the unclamped value, so clamping here to _BIG (as
        # the candidate FEATURES do) makes the two disagree wherever a neighbour cannot
        # reach the destination: argmin over [inf, inf] yields index 0, while argmax
        # over -(gcost + _BIG) yields whichever neighbour is momentarily cheapest.
        # Measured: 16 of 150 routes differed at width 1 before this was matched.
        v = -(gcost + to_go)
        if not np.isfinite(v).any():
            v = np.zeros_like(v)          # all hopeless -> tie, and ties take index 0
        m = v.max()
        return v - m - np.log(np.exp(v - m).sum())
    return score


class DRLRoutingAgent(abc.ABC):
    """Contract that the rollout (policy_drl) needs from any routing agent."""

    @abc.abstractmethod
    def act(self, obs, greedy=False):
        """Return an action index into obs.neighbors."""

    def load(self, path):     # analytic agents need no checkpoint
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


class GreedyOracleAgent(DRLRoutingAgent):
    """Analytic placeholder: A*-greedy on eq.(4) cost + free-flow time-to-go.

    A myopic (1-step) stand-in for the trained policy so policy_drl and the whole
    comparison run *today*. It is NOT learned; replace it with a trained
    EGATActorCritic via run_compare's --drl flag. Being myopic, it should
    under-perform the batched `policy_global_penalty` oracle — which is the point:
    it proves the slot works and gives the PPO agent a concrete bar to clear.
    """

    def act(self, obs, greedy=True):
        return int(np.argmin(obs.gcost + obs.to_go))


try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    from torch_geometric.nn import GATv2Conv
    _TORCH_OK = True
except Exception:                     # torch / torch_geometric optional for analytic policies
    _TORCH_OK = False


if _TORCH_OK:

    class _EGATEncoder(nn.Module):
        """Residual edge-aware GAT encoder (Lei et al. 2022 style): message passing
        over the road graph with edge features, residual connections + LayerNorm."""

        def __init__(self, node_dim, edge_dim, hidden, layers=3, heads=4):
            super().__init__()
            self.in_proj = nn.Linear(node_dim, hidden)
            self.in_norm = nn.LayerNorm(hidden)
            self.convs = nn.ModuleList([
                GATv2Conv(hidden, hidden, heads=heads, concat=False,
                          edge_dim=edge_dim, add_self_loops=False)
                for _ in range(layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        def forward(self, x, edge_index, edge_attr):
            h = self.in_norm(self.in_proj(x))
            for conv, norm in zip(self.convs, self.norms):
                h = norm(h + torch.relu(conv(h, edge_index, edge_attr)))   # residual + norm
            return h

    class EGATActorCritic(nn.Module):
        """Residual E-GAT actor-critic (proposal §4.4 decision module).

        Encoder: message passing over the road graph with edge features
        [t0, tpred, length, current rho] and node features [dist-to-dest, is-dest,
        out-rho] -> node embeddings that carry NETWORK-WIDE congestion context. It is
        re-encoded per vehicle as load builds, so congestion propagates across the
        graph (the global context the per-node MLP lacked). Decoder: scores each
        candidate neighbour from [h_cur, h_w, h_dest, local edge features]; the critic
        reads a graph-pooled state value.

        Built with the graph `g` (fixed topology -> edge_index/edge_static buffers).
        Duck-typed to DRLRoutingAgent (.act); the encode is cached per vehicle in
        rollout and recomputed (with grad) per vehicle in the PPO update.
        """

        NODE_DIM = 3
        EDGE_DIM = 4    # [t0, tpred, length, rho]

        def __init__(self, g, hidden=128, layers=3, heads=4, cand_dim=EDGE_FEATURE_DIM):
            super().__init__()
            edge_list = list(g.edges())
            ei = torch.tensor([[u for u, _ in edge_list],
                               [v for _, v in edge_list]], dtype=torch.long)
            es = torch.tensor([[g.edges[e]["t0"], g.edges[e]["tpred"], g.edges[e]["length"]]
                               for e in edge_list], dtype=torch.float)
            self.register_buffer("edge_index", ei)
            self.register_buffer("edge_static", es)
            self.encoder = _EGATEncoder(self.NODE_DIM, self.EDGE_DIM, hidden, layers, heads)
            self.cand_norm = nn.LayerNorm(cand_dim)
            self.actor = nn.Sequential(nn.Linear(3 * hidden + cand_dim, hidden),
                                       nn.ReLU(), nn.Linear(hidden, 1))
            self.critic = nn.Sequential(nn.Linear(3 * hidden, hidden),
                                        nn.ReLU(), nn.Linear(hidden, 1))
            self._cv, self._H = None, None       # rollout cache: (vehicle id, node embeddings)

        @property
        def device(self):
            return self.edge_static.device

        # ---- graph encode / candidate decode (inputs moved to the model's device) ----
        def encode(self, node_dyn, edge_rho):
            dev = self.device
            node_dyn = torch.as_tensor(node_dyn, dtype=torch.float32).to(dev)
            edge_rho = torch.as_tensor(edge_rho, dtype=torch.float32).to(dev)
            edge_attr = torch.cat([self.edge_static, edge_rho.unsqueeze(-1)], dim=-1)
            return self.encoder(node_dyn, self.edge_index, edge_attr)

        def decode(self, H, cur, dest, cand_ids, cand_feats):
            cand_ids = torch.as_tensor(cand_ids, dtype=torch.long).to(H.device)
            cand_feats = torch.as_tensor(cand_feats, dtype=torch.float32).to(H.device)
            hc, hd, hpool = H[cur], H[dest], H.mean(0)
            k = cand_feats.shape[0]
            ctx = torch.cat([hc, hd]).unsqueeze(0).expand(k, -1)          # [k, 2H]
            logits = self.actor(torch.cat([ctx, H[cand_ids],
                                           self.cand_norm(cand_feats)], dim=-1)).squeeze(-1)
            value = self.critic(torch.cat([hc, hd, hpool])).squeeze(-1)
            return logits, value

        # ---- rollout (encode cached per vehicle) ----
        def _rollout_H(self, obs):
            if obs.vehicle != self._cv:
                self._cv = obs.vehicle
                self._H = self.encode(torch.as_tensor(obs.node_dyn),
                                      torch.as_tensor(obs.edge_rho))
            return self._H

        @torch.no_grad()
        def act(self, obs, greedy=False):
            if len(obs.neighbors) == 1:
                return 0
            logits, _ = self.decode(self._rollout_H(obs), obs.node, obs.dest,
                                    torch.as_tensor(obs.neighbors),
                                    torch.as_tensor(obs.feats))
            if greedy:
                return int(torch.argmax(logits))
            return int(Categorical(logits=logits).sample())

        @torch.no_grad()
        def act_with_value(self, obs):
            """Rollout step for PPO: returns (action_idx, log_prob, value) as floats."""
            logits, value = self.decode(self._rollout_H(obs), obs.node, obs.dest,
                                        torch.as_tensor(obs.neighbors),
                                        torch.as_tensor(obs.feats))
            dist = Categorical(logits=logits)
            a = dist.sample()
            return int(a), float(dist.log_prob(a)), float(value)

        def evaluate(self, H, tr, action):
            """Re-evaluate a stored transition under a grad-enabled H -> logp, value, entropy."""
            logits, value = self.decode(H, tr["cur"], tr["dest"], tr["cands"], tr["cand_feats"])
            dist = Categorical(logits=logits)
            a = torch.as_tensor(action, device=logits.device)
            return dist.log_prob(a), value, dist.entropy()

        def reset_cache(self):
            self._cv, self._H = None, None

        def load(self, path):
            self.load_state_dict(torch.load(path, map_location="cpu"))
            self.eval()
            return self

        def save(self, path):
            torch.save(self.state_dict(), path)

    class PPOTrainer:
        """PPO (clipped surrogate, eq.5) for the E-GAT actor-critic on RoutingEnv.

        Transitions are grouped by vehicle: the graph is re-encoded once per vehicle
        (load is ~constant within a trip) in both rollout and update, so the encoder
        gets gradients without re-encoding per step. The update is mini-batched over
        vehicles (`mb_vehicles`) to bound memory. The per-vehicle encode is the heavy
        op; on the dense graph it is slow on CPU — this is where k-NN sparsification
        (config.KNN) or a GPU pays off.
        """

        def __init__(self, env, agent, lr=3e-4, clip_eps=0.2, gamma=0.99,
                     gae_lambda=0.95, value_coef=0.5, entropy_coef=0.01,
                     epochs=4, max_grad_norm=0.5, mb_vehicles=16):
            self.env, self.agent = env, agent
            self.clip_eps, self.gamma, self.lam = clip_eps, gamma, gae_lambda
            self.value_coef, self.entropy_coef = value_coef, entropy_coef
            self.epochs, self.max_grad_norm, self.mb_vehicles = epochs, max_grad_norm, mb_vehicles
            self.opt = torch.optim.Adam(agent.parameters(), lr=lr)

        def collect_episode(self):
            """One full pass over all vehicles -> list of transitions."""
            self.agent.reset_cache()
            traj, obs = [], self.env.reset()
            while not self.env.done and obs is not None:
                a, logp, value = self.agent.act_with_value(obs)
                tr = {"vehicle": obs.vehicle, "node_dyn": obs.node_dyn, "edge_rho": obs.edge_rho,
                      "cur": obs.node, "dest": obs.dest,
                      "cands": torch.as_tensor(obs.neighbors),
                      "cand_feats": torch.as_tensor(obs.feats),
                      "action": a, "logp": logp, "value": value}
                obs, reward, _, _ = self.env.step(a)
                tr["reward"] = reward
                traj.append(tr)
            return traj

        def _gae(self, traj):
            adv, gae, next_value = [0.0] * len(traj), 0.0, 0.0
            for t in reversed(range(len(traj))):
                # Each VEHICLE gets its own return. One episode routes hundreds of
                # vehicles back to back, so carrying the return across them would ask
                # the critic to predict the total cost of every remaining trip -- a
                # target whose scale and variance grow with the fleet size and which
                # no amount of training can fit. Vehicles still couple through the
                # load they leave behind, and the critic sees that coupling in the
                # state (per-edge rho), so cutting here discards nothing it could
                # actually have used.
                if t == len(traj) - 1 or traj[t + 1]["vehicle"] != traj[t]["vehicle"]:
                    next_value, gae = 0.0, 0.0
                delta = traj[t]["reward"] + self.gamma * next_value - traj[t]["value"]
                gae = delta + self.gamma * self.lam * gae
                adv[t] = gae
                next_value = traj[t]["value"]
            returns = [a + traj[t]["value"] for t, a in enumerate(adv)]
            return adv, returns

        def update(self, traj):
            if not traj:
                return {}
            adv, returns = self._gae(traj)
            adv_t = torch.tensor(adv, dtype=torch.float32)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            ret_t = torch.tensor(returns, dtype=torch.float32)
            old_logp = torch.tensor([tr["logp"] for tr in traj], dtype=torch.float32)
            # group transition indices by vehicle (encode once per vehicle, decode per step)
            veh_in, veh_idx = {}, {}
            for i, tr in enumerate(traj):
                vid = tr["vehicle"]
                if vid not in veh_in:
                    veh_in[vid] = (torch.as_tensor(tr["node_dyn"]), torch.as_tensor(tr["edge_rho"]))
                veh_idx.setdefault(vid, []).append(i)
            vids = list(veh_idx)
            stats = {}
            for _ in range(self.epochs):
                for s in torch.randperm(len(vids)).split(self.mb_vehicles):   # vehicle minibatches
                    idxs, logp, values, ent = [], [], [], []
                    for vid in (vids[j] for j in s.tolist()):
                        H = self.agent.encode(*veh_in[vid])
                        for i in veh_idx[vid]:
                            lp, v, e = self.agent.evaluate(H, traj[i], traj[i]["action"])
                            logp.append(lp); values.append(v); ent.append(e); idxs.append(i)
                    idx_t = torch.tensor(idxs)
                    logp = torch.stack(logp); values = torch.stack(values)
                    entropy = torch.stack(ent).mean()
                    dev = self.agent.device
                    a, r, ol = adv_t[idx_t].to(dev), ret_t[idx_t].to(dev), old_logp[idx_t].to(dev)
                    ratio = torch.exp(logp - ol)
                    clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    policy_loss = -torch.min(ratio * a, clipped * a).mean()
                    value_loss = ((values - r) ** 2).mean()
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.opt.step()
                    stats = {"policy_loss": float(policy_loss),
                             "value_loss": float(value_loss), "entropy": float(entropy)}
            return stats

        def train(self, iterations=100, log_every=10):
            self.agent.train()
            for it in range(1, iterations + 1):
                traj = self.collect_episode()
                stats = self.update(traj)
                if it % log_every == 0:
                    ret = sum(tr["reward"] for tr in traj)
                    print(f"[PPO] iter {it:4d}  return {ret:10.3f}  "
                          f"pi_loss {stats.get('policy_loss', 0):.4f}  "
                          f"v_loss {stats.get('value_loss', 0):.4f}")
            self.agent.eval()
            return self.agent


def policy_drl(g, demand, agent, max_hops=60, stats=None, togo_refresh=0,
               closure=None, beam=0):
    """Route every vehicle by rolling `agent` out in RoutingEnv (greedy).

    Same (g, demand) -> paths contract as the analytic policies, so it drops straight
    into run_compare. `agent` is anything implementing DRLRoutingAgent.act.

    `max_hops` must match the graph: city routes (Taichung) run 30-70 hops, so the
    60-hop default would abandon most trips mid-way and understate the agent. Use
    network.default_max_hops(dataset).
    """
    # togo_refresh MUST match what the agent was trained with: it changes the
    # observation the policy reads, so a mismatch silently degrades the agent rather
    # than raising. train_drl.py records it in <checkpoint>.meta.json for this reason.
    env = RoutingEnv(g, demand, use_penalty=True, max_hops=max_hops,
                     togo_refresh=togo_refresh, closure=closure)
    obs = env.reset()
    if beam and beam > 1:
        # Same policy, wider decoding. beam=1 goes through the greedy path below on
        # purpose: the two must agree, and routing it here would hide a divergence.
        while not env.done:
            env.commit(*env.beam_route(agent, beam, max_hops))
    else:
        while not env.done and obs is not None:
            obs, _, _, _ = env.step(agent.act(obs, greedy=True))
    if stats is not None:                 # optional failure-mode breakdown
        stats.update(dead_end=env.n_deadend, max_hops=env.n_maxhops,
                     trivial=env.n_trivial)
    return env.paths


def make_drl_agent(spec, g):
    """Build an agent from a run_compare --drl spec:
        'placeholder'/'oracle' -> GreedyOracleAgent (analytic, no torch/training)
        <path.pt>              -> trained EGATActorCritic (built on `g`) from checkpoint
    """
    if spec in (None, "", "placeholder", "oracle"):
        return GreedyOracleAgent()
    if not _TORCH_OK:
        raise ImportError("PyTorch + torch_geometric are required for the DRL agent.")
    return EGATActorCritic(g).load(spec)
