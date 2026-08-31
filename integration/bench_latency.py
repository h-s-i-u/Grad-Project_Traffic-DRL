#!/usr/bin/env python3
"""Inference latency of the decision layer (proposal section 5: "<50 ms, single decision").

The proposal claims the DRL agent answers a routing decision in under 50 ms, and the
experiment log already leans on that claim (13.11.4 defends the Gini gap as "the price
of <50 ms responsiveness") without ever having measured it. This script measures it.

Timing a run_compare rollout and dividing by the vehicle count does NOT measure it: it
mixes a per-vehicle graph encode with a per-hop decode, and the hop count is itself a
variable (mean 28.7, max 79 on the arena). The cost decomposes instead as

    one vehicle's route
      = [to-go Dijkstra]      only on a cache miss -- 1 vehicle in `togo_refresh`
      + [_compute_enc_ctx]    numpy: N-node loop + E-edge loop
      + [encoder forward]     Residual E-GAT over the whole graph, once per vehicle
      + n_hops x ( [_observe] + [decoder forward] + [env.step] )

Only the two forwards are model inference; the rest is state preparation that a
deployment would get from its traffic-state service. Both are reported, separately.

"Single decision" has three defensible readings and reporting only the cheapest one
would be gaming the claim, so all three come out:

    (a) one hop              the literal reading
    (b) one vehicle's route  what a driver actually waits for      <- the headline
    (c) the whole fleet      the "incident -> everyone reroutes" case of proposal 2.1

Percentiles, not means: `togo_refresh` makes the per-vehicle cost bimodal (one vehicle
in K pays a full Dijkstra) and a real-time claim lives or dies on its tail.

The Dijkstra baselines are in the same table on purpose. "<50 ms" alone says nothing;
what proposal 2.1 asserts is that recomputing a global optimum is too expensive to be
real-time, and that assertion has never been tested either.

NOTHING IN policies.py IS MODIFIED. The timers are installed on the RoutingEnv and
agent INSTANCES -- an instance attribute shadows the class method -- so no number that
has already been reported can move. `--verify` re-runs the same demand through the
untouched policies.policy_drl and compares every path, which is the same guard
test_beam.py uses for its width-1 equivalence check.

    cd integration
    python bench_latency.py --graph taichung --vehicles 800 \
           --drl checkpoints/taichung/drl_fusion_togo25.pt --beam 8 --scale --verify
    python bench_latency.py --graph taichung --vehicles 800 \
           --drl checkpoints/taichung/drl_fusion_togo25.pt --beam 8 --device cpu
    python bench_latency.py --graph taichung --vehicles 800 \
           --drl checkpoints/taichung/drl_fusion_togo25.pt --beam 8 \
           --close-road 臺灣大道 --close-at 0.10
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import networkx as nx
import numpy as np

import closure as clo
import config as C
import network as net
import policies as pol
from run_compare import make_demand

BUDGET_MS = 50.0        # proposal section 5: inference latency below 50 ms per
                        # single decision, "meeting real-time navigation needs"
PCTS = (50, 90, 99)


# ---------------------------------------------------------------- timing core
class Stat:
    """Samples of one component's EXCLUSIVE (self) time, tagged by vehicle."""

    __slots__ = ("name", "veh", "dt")

    def __init__(self, name):
        self.name, self.veh, self.dt = name, [], []

    def add(self, veh, dt):
        self.veh.append(veh)
        self.dt.append(dt)

    def samples(self, since=0):
        """Seconds, dropping vehicles below `since` (warm-up) but keeping veh < 0,
        which marks calls made outside a rollout (the standalone baselines)."""
        return [d for v, d in zip(self.veh, self.dt) if v < 0 or v >= since]


class Prof:
    """Nesting-aware profiler.

    A parent's time is reported EXCLUSIVE of its children, so the component table
    sums to the total instead of double counting `_compute_enc_ctx` inside the
    Dijkstra it triggers. Each sample is tagged with `env._vi` read at call entry,
    which is what makes per-vehicle attribution correct across the boundary where
    step() sets up the NEXT vehicle while still inside the current one's last hop.
    """

    def __init__(self, sync=None):
        self.sync = sync                 # torch.cuda.synchronize, or None on CPU
        self.stats = {}
        self._stack = []

    def stat(self, name):
        st = self.stats.get(name)
        if st is None:
            st = self.stats[name] = Stat(name)
        return st

    def run(self, name, veh, sync, fn, *a, **kw):
        self._stack.append(0.0)
        if sync and self.sync:
            self.sync()
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            if sync and self.sync:
                self.sync()
            dt = time.perf_counter() - t0
            child = self._stack.pop()
            self.stat(name).add(veh, dt - child)
            if self._stack:
                self._stack[-1] += dt         # report INCLUSIVE time to the parent

    def wrap(self, obj, attr, name, vfn, sync=False):
        fn = getattr(obj, attr)

        def wrapped(*a, **kw):
            return self.run(name, vfn(), sync, fn, *a, **kw)

        setattr(obj, attr, wrapped)


def unwrap(obj, attrs):
    """Drop instance attributes so the class methods show through again.

    Required between runs: wrapping an already-wrapped instance would nest the timers
    and every component would be counted twice.
    """
    for a in attrs:
        obj.__dict__.pop(a, None)


ENV_ATTRS = ("_dist_to_dest", "_compute_enc_ctx", "_observe", "step",
             "_beam_feats", "beam_route", "commit")
AGENT_ATTRS = ("encode", "decode", "act")


def instrument(env, agent, prof):
    vi = lambda: env._vi                                  # noqa: E731  (read at entry)

    # Cache hit and miss are two different operations sharing one name: the miss runs
    # a full Dijkstra over the reversed graph, the hit is a dict lookup. Averaging
    # them hides exactly the tail the real-time claim is about.
    inner = env._dist_to_dest

    def dist_to_dest(dest):
        name = "togo dijkstra (miss)" if dest not in env._dist_cache else "togo (cache hit)"
        return prof.run(name, env._vi, False, inner, dest)

    env._dist_to_dest = dist_to_dest

    prof.wrap(env, "_compute_enc_ctx", "enc ctx (numpy)", vi)
    prof.wrap(env, "_observe", "observe (numpy)", vi)
    prof.wrap(env, "step", "env step", vi)
    prof.wrap(env, "_beam_feats", "beam feats (numpy)", vi)
    prof.wrap(env, "beam_route", "beam route", vi)
    prof.wrap(env, "commit", "beam commit", vi)
    # The analytic agents (placeholder / oracle) have no encoder or decoder, so the
    # model rows are simply absent for them rather than zero -- which is the honest
    # rendering: they are not doing inference at all.
    for attr, name, sy in (("encode", "ENCODE (E-GAT)", True),
                           ("decode", "DECODE (actor)", True),
                           ("act", "act (dispatch)", False)):
        if hasattr(agent, attr):
            prof.wrap(agent, attr, name, vi, sync=sy)


def reset_cache(agent):
    if hasattr(agent, "reset_cache"):
        agent.reset_cache()


MODEL_COMPONENTS = ("ENCODE (E-GAT)", "DECODE (actor)")


def per_vehicle(prof, since=0, only=None):
    """{vehicle: seconds} summed over exclusive component times."""
    acc = defaultdict(float)
    for name, st in prof.stats.items():
        if only is not None and name not in only:
            continue
        for v, dt in zip(st.veh, st.dt):
            if v >= since:
                acc[v] += dt
    return acc


# ------------------------------------------------------------------ reporting
def pcts(xs):
    """(n, mean, p50, p90, p99, max) in ms."""
    if not xs:
        return 0, *(float("nan"),) * 5
    a = np.asarray(xs, dtype=float) * 1e3
    return (len(a), float(a.mean()), *(float(np.percentile(a, q)) for q in PCTS),
            float(a.max()))


def row(label, xs, width=34, total=False):
    n, mean, p50, p90, p99, mx = pcts(xs)
    tail = f"{sum(xs):9.2f}" if total else ""
    return (f"  {label:<{width}}{n:>8,}{mean:>10.3f}{p50:>9.3f}"
            f"{p90:>9.3f}{p99:>9.3f}{mx:>10.3f}{tail}")


def header(width=34, total=False):
    tail = f"{'total s':>9}" if total else ""
    return (f"  {'':<{width}}{'calls':>8}{'mean':>10}{'p50':>9}{'p90':>9}"
            f"{'p99':>9}{'max':>10}{tail}\n"
            f"  {'':<{width}}{'':>8}{'ms':>10}{'ms':>9}{'ms':>9}{'ms':>9}{'ms':>10}"
            + (f"{'':>9}" if total else ""))


def verdict(xs):
    """How the sample sits against the 50 ms budget."""
    if not xs:
        return "n/a"
    a = np.asarray(xs) * 1e3
    frac = float((a < BUDGET_MS).mean())
    mark = "PASS" if frac >= 0.99 else ("MARGINAL" if frac >= 0.90 else "FAIL")
    return f"{mark:<8} {frac:6.1%} of calls < {BUDGET_MS:.0f} ms"


# -------------------------------------------------------------------- rollouts
def rollout_greedy(g, demand, agent, max_hops, togo_refresh, closure, prof):
    """Greedy decode, timing every hop from the outside as well as by component.

    Two hops are marked and excluded from the per-hop figures because they carry work
    that is not theirs:
      `first`    -- the first hop of a vehicle pays that vehicle's encoder forward
      `boundary` -- the last hop's step() sets up the NEXT vehicle (Dijkstra + enc ctx)
    Both still count in full toward the per-vehicle number, which is reconstructed from
    the component tags and is therefore boundary-correct.
    """
    env = pol.RoutingEnv(g, demand, use_penalty=True, max_hops=max_hops,
                         togo_refresh=togo_refresh, closure=closure)
    instrument(env, agent, prof)
    reset_cache(agent)
    hops = []                       # (vehicle, model_s, env_s, first, boundary)
    obs = env.reset()
    # None, not env._vi: seeding this with the current vehicle makes the very first hop
    # of the rollout compare equal to itself and escape the `first` flag -- and that is
    # the one hop carrying the process's first encoder forward, so it lands in the
    # per-hop `max` column as a ~20 ms outlier that is really a per-vehicle cost.
    prev_v = None
    while not env.done and obs is not None:
        v = env._vi
        t0 = time.perf_counter()
        a = agent.act(obs, greedy=True)
        t1 = time.perf_counter()
        obs, _, _, _ = env.step(a)
        t2 = time.perf_counter()
        hops.append((v, t1 - t0, t2 - t1, v != prev_v, env._vi != v))
        prev_v = v
    unwrap(env, ENV_ATTRS)
    unwrap(agent, AGENT_ATTRS)
    return env.paths, hops


def rollout_beam(g, demand, agent, width, max_hops, togo_refresh, closure, prof):
    """Beam decode. The decision unit here is the whole vehicle: beam_route returns a
    finished path, so there is no per-hop request to time."""
    env = pol.RoutingEnv(g, demand, use_penalty=True, max_hops=max_hops,
                         togo_refresh=togo_refresh, closure=closure)
    instrument(env, agent, prof)
    reset_cache(agent)
    decode_s = []                   # (vehicle, seconds inside beam_route)
    env.reset()
    while not env.done:
        v = env._vi
        t0 = time.perf_counter()
        route, reason, traversed = env.beam_route(agent, width, max_hops)
        decode_s.append((v, time.perf_counter() - t0))
        env.commit(route, reason, traversed)
    unwrap(env, ENV_ATTRS)
    unwrap(agent, AGENT_ATTRS)
    return env.paths, decode_s


# ------------------------------------------------------------------- baselines
def bench_dijkstra(g, demand, weight, limit):
    """One routing request under a fixed edge weight -- policies 1 and 4."""
    out = []
    for o, d in demand[:limit]:
        t0 = time.perf_counter()
        try:
            nx.shortest_path(g, o, d, weight=weight)
        except nx.NetworkXNoPath:
            pass
        out.append(time.perf_counter() - t0)
    return out


def bench_oracle(g, demand, limit):
    """Policy 6 costs two things, and only reporting one of them would be unfair
    either way: re-pricing all E edges with eq.4 (once per batch, amortised over the
    batch) and one Dijkstra on that price (once per vehicle).

    Returns (reprice seconds, dijkstra seconds, vehicles per batch).
    """
    edges = list(g.edges())
    t0a = {e: g.edges[e]["t0"] for e in edges}
    cap = {e: g.edges[e]["cap"] for e in edges}
    load = {e: 0.0 for e in edges}
    t_ref = float(np.mean([t0a[e] for e in edges])) if edges else 1.0

    def reprice():
        rho = {e: load[e] / cap[e] for e in edges}
        mean_rho = float(np.mean([rho[e] for e in edges])) if edges else 0.0
        for e in edges:
            cost = C.ALPHA * pol._bpr(t0a[e], load[e], cap[e])
            cost += t_ref * C.PENALTY_SCALE * (
                C.LAMBDA_SAT * max(0.0, rho[e] - C.RHO_THRESHOLD) ** 2
                + C.LAMBDA_VAR * max(0.0, rho[e] - mean_rho))
            g.edges[e]["cost"] = cost

    rp = []
    for _ in range(5):
        t0 = time.perf_counter()
        reprice()
        rp.append(time.perf_counter() - t0)
    dj = bench_dijkstra(g, demand, "cost", limit)
    for e in edges:                                  # leave the graph as we found it
        g.edges[e].pop("cost", None)
    return rp, dj, max(1, len(demand) // C.N_BATCHES)


# ----------------------------------------------------------------- scale study
def bench_scale(ckpt, device, sync, reps, warmup, capacity_scale):
    """Same weights, 840-node arena vs the full 9,904-node simplified network.

    The E-GAT weights carry no node or edge count (that is the inductive property the
    proposal claims for STGAT in 3.3 and Lei et al. for the E-GAT), so the SAME
    checkpoint loads onto a different graph -- edge_index / edge_static are shape-bound
    buffers and are dropped. Routing QUALITY at this scale is untested and is not
    claimed here; only the forward's cost is being measured.
    """
    import torch
    from taichung_loader import load_taichung_graph

    root = C.ROOT / "Map"
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k: v for k, v in sd.items() if k not in ("edge_index", "edge_static")}
    out = []
    for label, ne, nn_ in (("arena", "arena_edges_taichung.csv", "arena_nodes_taichung.csv"),
                           ("full network", "simplified_edges_taichung.csv",
                            "simplified_nodes_taichung.csv")):
        if not (root / ne).is_file():
            print(f"  ! {root / ne} missing -- skipping '{label}'")
            continue
        g = load_taichung_graph(edges_csv=root / ne, nodes_csv=root / nn_,
                                default_speed_kmh=C.TAICHUNG_DEFAULT_SPEED_KMH,
                                capacity_scale=capacity_scale, largest_scc_only=True,
                                tpred_fallback=C.TAICHUNG_TPRED_FALLBACK, verbose=False)
        agent = pol.EGATActorCritic(g)
        missing, unexpected = agent.load_state_dict(sd, strict=False)
        # edge_index / edge_static are shape-bound buffers and were dropped on purpose.
        # Anything ELSE missing means the weights did not actually transfer, and a
        # randomly-initialised encoder would time just fine while measuring nothing.
        stray = [k for k in missing if k not in ("edge_index", "edge_static")]
        if stray or unexpected:
            raise SystemExit(f"error: checkpoint did not transfer to '{label}' "
                             f"(missing {stray}, unexpected {list(unexpected)})")
        agent.to(device).eval()
        scc = sorted(net.largest_scc(g))
        rng = np.random.default_rng(C.SEED)
        n_pairs = reps + warmup + 5
        pairs = [(int(a), int(b)) for a, b in
                 zip(rng.choice(scc, n_pairs), rng.choice(scc, n_pairs)) if a != b]
        env = pol.RoutingEnv(g, pairs, use_penalty=True,
                             max_hops=C.TAICHUNG_MAX_HOPS, togo_refresh=25)
        env.reset()

        def one_rep(i, sink):
            """All four ops on pair i. `sink` is None for an untimed warm-up rep."""
            o, dest = pairs[i % len(pairs)]
            env._dest, env._dist_cache = dest, {}
            t = time.perf_counter()
            env._dist_to_dest(dest)
            d_dj = time.perf_counter() - t
            t = time.perf_counter()
            enc = env._compute_enc_ctx()
            d_ec = time.perf_counter() - t
            if sync:
                sync()
            t = time.perf_counter()
            agent.encode(*enc)
            if sync:
                sync()
            d_en = time.perf_counter() - t
            t = time.perf_counter()
            try:
                nx.shortest_path(g, o, dest, weight="t0")
            except nx.NetworkXNoPath:
                pass
            d_sp = time.perf_counter() - t
            if sink is not None:
                for lst, v in zip(sink, (d_dj, d_ec, d_en, d_sp)):
                    lst.append(v)

        # Untimed reps first. A freshly built model's first forward carries lazy module
        # init, kernel autotuning and the CUDA caching allocator's first grow. Without
        # this the arena encode measured 3.13 ms mean against a 2.10 ms median, which
        # inflated the SMALL graph and therefore understated the 840 -> 9,904 ratio --
        # the one number this whole block exists to produce (13.25).
        for i in range(warmup):
            one_rep(i, None)
        dj, ec, en, sp = [], [], [], []
        for i in range(reps):
            one_rep(warmup + i, (dj, ec, en, sp))
        out.append({"label": label, "nodes": g.number_of_nodes(),
                    "edges": g.number_of_edges(), "togo dijkstra (miss)": dj,
                    "enc ctx (numpy)": ec, "ENCODE (E-GAT)": en,
                    "1 static Dijkstra": sp})
        del agent
    return out


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", choices=net.GRAPHS, default="taichung")
    ap.add_argument("--vehicles", type=int, default=800)
    ap.add_argument("--scenario", choices=["random", "hotspot"], default=C.SCENARIO)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--capacity-scale", type=float, default=None)
    ap.add_argument("--max-hops", type=int, default=None)
    ap.add_argument("--drl", required=True, metavar="placeholder|oracle|CKPT.pt")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="default: cuda when available. Run BOTH -- on a 1,690-edge "
                         "graph the forward is kernel-launch bound (13.13 measured "
                         "50-100 us per launch), so the GPU can lose")
    ap.add_argument("--beam", type=int, default=0, metavar="W",
                    help="also time beam-W decoding, which calls the decoder once per "
                         "live beam per hop. 18.3 H makes beam-8 the reported row, so "
                         "its latency is the one the report has to defend")
    ap.add_argument("--warmup", type=int, default=20, metavar="N",
                    help="discard the first N vehicles (CUDA context, lazy init, "
                         "cuDNN autotune all land on the first forward)")
    ap.add_argument("--baseline-limit", type=int, default=400, metavar="N",
                    help="how many single Dijkstra requests to time per baseline")
    ap.add_argument("--scale", action="store_true",
                    help="also time the encoder on the full 9,904-node network")
    ap.add_argument("--scale-reps", type=int, default=20)
    ap.add_argument("--scale-warmup", type=int, default=5, metavar="N",
                    help="untimed reps per graph before the timed ones. The ratio this "
                         "block reports is only as good as its SMALL-graph number, and "
                         "that is the one a cold start inflates")
    ap.add_argument("--verify", action="store_true",
                    help="re-run the same demand through the untouched "
                         "policies.policy_drl and require every path to match")
    ap.add_argument("--out", default=None)
    ap.add_argument("--close-road", metavar="PREFIX", default=None)
    ap.add_argument("--close-edge", metavar="FROM,TO", default=None)
    ap.add_argument("--close-at", type=float, default=0.5, metavar="FRAC")
    ap.add_argument("--close-demand", choices=["filter", "wave", "resample", "none"],
                    default="filter")
    args = ap.parse_args()

    # Road names are Chinese and Windows hands a redirected stdout cp1252, so piping
    # this to a file or a pager would die on --close-road rather than on anything real.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    C.SCENARIO, C.N_VEHICLES = args.scenario, args.vehicles
    max_hops = args.max_hops or net.default_max_hops(args.graph)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: --device cuda but torch.cuda.is_available() is False")
    sync = torch.cuda.synchronize if device == "cuda" else None

    g, info = net.build_graph_for(args.graph, capacity_scale=args.capacity_scale,
                                  verbose=False)
    scc = net.largest_scc(g)
    rng = np.random.default_rng(args.seed)
    demand, hubs = make_demand(g, scc, rng)

    closure = None
    if args.close_road or args.close_edge:
        probe, hubs0 = make_demand(g, scc, np.random.default_rng(args.seed))
        load, total = clo.baseline_load(g, probe, pol.policy_prediction_greedy)
        if args.close_road:
            label, edges = args.close_road, clo.edges_by_road(g, args.close_road)
        else:
            label, edges = args.close_edge, clo.edges_by_endpoints(g, args.close_edge)
        closure = clo.Closure(edges, at=args.close_at, label=label)
    demand, dem_info = clo.select_demand(g, closure, demand, args.close_demand)

    agent = pol.make_drl_agent(args.drl, g)
    togo_refresh = 0
    meta_path = os.path.splitext(args.drl)[0] + ".meta.json"
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            togo_refresh = int(json.load(f).get("togo_refresh", 0) or 0)
    if hasattr(agent, "to"):
        agent.to(device).eval()

    print(f"\n{'=' * 100}\nInference latency -- decision layer (proposal section 5: "
          f"< {BUDGET_MS:.0f} ms per single decision)\n{'=' * 100}")
    print(f"graph        : {args.graph}  {g.number_of_nodes():,} nodes / "
          f"{g.number_of_edges():,} directed edges")
    print(f"demand       : {len(demand):,} vehicles, scenario={C.SCENARIO}, "
          f"seed={args.seed}, max_hops={max_hops}")
    print(f"agent        : {args.drl}  (togo_refresh={togo_refresh})")
    print(f"device       : {device}"
          + (f"  [{torch.cuda.get_device_name(0)}]" if device == "cuda" else "")
          + f"   torch {torch.__version__}")
    print(f"warm-up      : first {args.warmup} vehicles discarded")
    if closure is not None:
        print(f"closure      : {closure.label} at {args.close_at:.0%} of the demand "
              f"({dem_info})")
    print("timers       : installed on the env/agent INSTANCES; policies.py untouched")

    # ---- greedy ----------------------------------------------------------
    prof_g = Prof(sync)
    paths_g, hops = rollout_greedy(g, demand, agent, max_hops, togo_refresh,
                                   closure, prof_g)

    if args.verify:
        reset_cache(agent)
        ref = pol.policy_drl(g, demand, agent, max_hops=max_hops,
                             togo_refresh=togo_refresh, closure=closure)
        bad = [i for i, (a, b) in enumerate(zip(paths_g, ref)) if a != b]
        if bad or len(ref) != len(paths_g):
            raise SystemExit(
                f"error: instrumented rollout diverged from policies.policy_drl on "
                f"{len(bad)} of {len(paths_g)} trips (first: vehicle {bad[0]}). The "
                f"timing loop is not reproducing the real one -- do not report these "
                f"numbers.")
        print(f"verify       : {len(paths_g):,}/{len(paths_g):,} paths identical to "
              f"policies.policy_drl  OK")

    print(f"\n--- component breakdown, greedy decoding "
          f"(exclusive time; vehicles >= {args.warmup}) ---")
    print(header(total=True))
    order = ["togo dijkstra (miss)", "togo (cache hit)", "enc ctx (numpy)",
             "ENCODE (E-GAT)", "observe (numpy)", "DECODE (actor)",
             "act (dispatch)", "env step"]
    for name in order:
        if name in prof_g.stats:
            print(row(name, prof_g.stats[name].samples(args.warmup), total=True))
    print(f"\n  UPPERCASE rows are model inference; the rest is state preparation a "
          f"deployment\n  would get from its traffic-state service, not from the agent.")

    served = sum(1 for p in paths_g if p)
    hop_n = len(hops)
    n_dec = len(prof_g.stat("DECODE (actor)").dt)
    print(f"\n  {hop_n:,} hops over {len(demand):,} vehicles "
          f"({served:,} served, {hop_n / max(1, served):.1f} hops per served trip); "
          f"the decoder ran\n  {n_dec:,} times ({n_dec / max(1, hop_n):.0%} of hops) -- "
          f"act() short-circuits where a node offers only one exit.")

    # ---- the three readings of "one decision" ----------------------------
    warm = [h for h in hops if h[0] >= args.warmup]
    plain = [h for h in warm if not h[3] and not h[4]]     # neither first nor boundary
    hop_model = [h[1] for h in plain]
    hop_e2e = [h[1] + h[2] for h in plain]
    veh_all = per_vehicle(prof_g, args.warmup)
    veh_model = per_vehicle(prof_g, args.warmup, only=MODEL_COMPONENTS)
    # _start_next_vehicle runs one index PAST the last vehicle to discover it is done,
    # so `veh_all` carries a trailing pseudo-vehicle that never routed anything.
    routed = [v for v in sorted(veh_all)
              if v < len(paths_g) and paths_g[v] is not None]
    veh_e2e_s = [veh_all[v] for v in routed]
    veh_model_s = [veh_model.get(v, 0.0) for v in routed]

    print(f"\n--- (a) one hop  [{len(plain):,} hops; the first hop of a vehicle and the "
          f"one that sets up\n        the next are excluded -- they carry work that is "
          f"not theirs] ---")
    print(header(38))
    print(row("model inference (decoder)", hop_model, 38))
    print(row("end-to-end (observe+decode+step)", hop_e2e, 38))

    print(f"\n--- (b) ONE VEHICLE'S ROUTE  [{len(routed):,} served vehicles; "
          f"reconstructed from the\n        component tags, so the vehicle-setup "
          f"boundary is attributed correctly] ---")
    print(header(38))
    print(row("model inference (encode+decode)", veh_model_s, 38))
    print(row("end-to-end (incl. state prep)", veh_e2e_s, 38))
    print(f"\n  vs the {BUDGET_MS:.0f} ms budget:  model {verdict(veh_model_s)}")
    print(f"                       end-to-end {verdict(veh_e2e_s)}")

    # Warm-up excluded and the rest scaled back up: the first forward carries CUDA
    # context creation and lazy init, which is a one-off of the process, not a cost
    # of re-planning a fleet.
    n_warm = max(1, len(demand) - args.warmup)
    fleet = sum(sum(st.samples(args.warmup)) for st in prof_g.stats.values())
    fleet_full = fleet * len(demand) / n_warm
    print(f"\n--- (c) the whole fleet re-planned  [proposal 2.1: the incident case] ---")
    print(f"  {'measured (' + format(n_warm, ',') + ' vehicles)':<32}"
          f"{fleet * 1e3:10.1f} ms   ({fleet:.2f} s)")
    print(f"  {'extrapolated to ' + format(len(demand), ',') + ' vehicles':<32}"
          f"{fleet_full * 1e3:10.1f} ms   ({fleet_full:.2f} s)")

    # ---- beam ------------------------------------------------------------
    beam_out = None
    if args.beam and args.beam > 1:
        prof_b = Prof(sync)
        paths_b, dec = rollout_beam(g, demand, agent, args.beam, max_hops,
                                    togo_refresh, closure, prof_b)
        vb_all = per_vehicle(prof_b, args.warmup)
        vb_model = per_vehicle(prof_b, args.warmup, only=MODEL_COMPONENTS)
        routed_b = [v for v in sorted(vb_all)
                    if v < len(paths_b) and paths_b[v] is not None]
        b_e2e = [vb_all[v] for v in routed_b]
        b_model = [vb_model.get(v, 0.0) for v in routed_b]
        b_search = [d for v, d in dec if v >= args.warmup]
        fleet_b = sum(sum(st.samples(args.warmup)) for st in prof_b.stats.values())
        print(f"\n--- beam-{args.beam} decoding  [same weights, wider decode: the decoder "
              f"runs once per\n    live beam per hop, so this is the row 18.3 H makes "
              f"the reported one] ---")
        print(header(38))
        for name in order + ["beam feats (numpy)", "beam route", "beam commit"]:
            if name in prof_b.stats:
                print(row(name, prof_b.stats[name].samples(args.warmup), 38))
        print()
        print(row("PER VEHICLE, model inference", b_model, 38))
        print(row("PER VEHICLE, end-to-end", b_e2e, 38))
        print(row("  of which inside beam_route", b_search, 38))
        print(f"\n  vs the {BUDGET_MS:.0f} ms budget:  model {verdict(b_model)}")
        print(f"                       end-to-end {verdict(b_e2e)}")
        print(f"  whole fleet: {fleet_b * 1e3:,.1f} ms "
              f"({fleet_b / max(fleet, 1e-9):.1f}x greedy)")
        beam_out = {"width": args.beam, "per_vehicle_model": pcts(b_model),
                    "per_vehicle_e2e": pcts(b_e2e), "inside_beam_route": pcts(b_search),
                    "fleet_s": fleet_b,
                    "components": {k: pcts(v.samples(args.warmup))
                                   for k, v in prof_b.stats.items()}}

    # ---- baselines -------------------------------------------------------
    print(f"\n--- one routing request, the Dijkstra baselines  [proposal 2.1 asserts "
          f"recomputing\n    a global optimum is too expensive for real time; this is "
          f"that assertion, measured] ---")
    print(header(38))
    lim = min(args.baseline_limit, len(demand))
    d_static = bench_dijkstra(g, demand, "t0", lim)
    d_pred = bench_dijkstra(g, demand, "tpred", lim)
    rp, d_orc, per_batch = bench_oracle(g, demand, lim)
    amort = [d + float(np.mean(rp)) / per_batch for d in d_orc]
    print(row("1 static (free-flow) Dijkstra", d_static, 38))
    print(row("4 hybrid (herding baseline)", d_pred, 38))
    print(row("6 oracle: Dijkstra on eq.4 cost", d_orc, 38))
    print(row(f"6 oracle: re-price {g.number_of_edges():,} edges", rp, 38))
    print(row(f"6 oracle: request + reprice/{per_batch}", amort, 38))
    print(f"\n  6 re-prices every edge once per batch ({C.N_BATCHES} batches, "
          f"~{per_batch} vehicles each),\n  so a request pays one Dijkstra plus its "
          f"share of a re-price. A single out-of-band\n  re-route pays the whole "
          f"re-price: {(float(np.mean(rp)) + float(np.mean(d_orc))) * 1e3:.1f} ms.")

    # ---- scale -----------------------------------------------------------
    scale_out = None
    if args.scale:
        if args.graph != "taichung":
            print("\n  ! --scale is taichung-only; skipped")
        elif args.drl in ("placeholder", "oracle"):
            print("\n  ! --scale needs a real checkpoint; skipped")
        else:
            print(f"\n--- scale: the same weights on the full network  [E-GAT is "
                  f"inductive, so the\n    checkpoint loads onto a different graph; "
                  f"only the forward's COST is measured here,\n    routing quality at "
                  f"this scale is untested and is not claimed.\n    "
                  f"{args.scale_warmup} untimed reps per graph, then "
                  f"{args.scale_reps} timed] ---")
            scale_out = bench_scale(args.drl, device, sync, args.scale_reps,
                                    args.scale_warmup,
                                    args.capacity_scale or C.TAICHUNG_CAPACITY_SCALE)
            for blk in scale_out:
                print(f"\n  {blk['label']}: {blk['nodes']:,} nodes / "
                      f"{blk['edges']:,} edges")
                print(header(38))
                for k in ("togo dijkstra (miss)", "enc ctx (numpy)",
                          "ENCODE (E-GAT)", "1 static Dijkstra"):
                    print(row(k, blk[k], 38))
            if len(scale_out) == 2:
                a, b = scale_out
                print(f"\n  scaling {a['nodes']:,} -> {b['nodes']:,} nodes "
                      f"({b['nodes'] / a['nodes']:.1f}x)   [MEDIANS: one slow rep on "
                      f"the small\n  graph shrinks the ratio, so the mean flatters "
                      f"whichever side is noisier]")
                for k in ("enc ctx (numpy)", "ENCODE (E-GAT)", "togo dijkstra (miss)",
                          "1 static Dijkstra"):
                    fa, fb = float(np.median(a[k])), float(np.median(b[k]))
                    print(f"    {k:<24} {fa * 1e3:8.3f} -> {fb * 1e3:9.3f} ms   "
                          f"x{fb / max(fa, 1e-12):6.1f}")

    # ---- json ------------------------------------------------------------
    out = {
        "graph": args.graph, "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
        "device": device, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "agent": args.drl, "togo_refresh": togo_refresh, "warmup": args.warmup,
        "vehicles": len(demand), "served": served, "hops": hop_n,
        "scenario": C.SCENARIO, "seed": args.seed, "budget_ms": BUDGET_MS,
        "closure": (None if closure is None
                    else {"label": closure.label, "at": args.close_at}),
        "pcts": ["n", "mean", *[f"p{q}" for q in PCTS], "max"],
        "greedy": {
            "components": {k: pcts(v.samples(args.warmup))
                           for k, v in prof_g.stats.items()},
            "hop_model": pcts(hop_model), "hop_e2e": pcts(hop_e2e),
            "per_vehicle_model": pcts(veh_model_s), "per_vehicle_e2e": pcts(veh_e2e_s),
            "fleet_s": fleet,
        },
        "beam": beam_out,
        "baselines": {"1_static": pcts(d_static), "4_hybrid": pcts(d_pred),
                      "6_oracle_dijkstra": pcts(d_orc), "6_oracle_reprice": pcts(rp),
                      "6_oracle_amortised": pcts(amort), "batch_size": per_batch},
        "scale": (None if not scale_out else
                  [{"label": b["label"], "nodes": b["nodes"], "edges": b["edges"],
                    **{k: pcts(b[k]) for k in
                       ("togo dijkstra (miss)", "enc ctx (numpy)", "ENCODE (E-GAT)",
                        "1 static Dijkstra")}} for b in scale_out]),
    }
    name = args.out or f"latency_{device}{'_s3' if closure is not None else ''}.json"
    path = C.HERE / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
