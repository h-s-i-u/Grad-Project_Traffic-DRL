#!/usr/bin/env python3
"""Resident router for the interactive demo -- mode A' of the SUMO handover doc, section 4.

WHAT THIS IS
    One long-lived object holding the road graph and the trained agent, answering a
    single question on demand:

        given the vehicles on the road right now, how full each road is right now, and
        which roads a visitor just closed -- where should each vehicle go next?

    Construction costs ~3.4 s (imports + graph + agent). Answering costs 0.25-0.46 s
    for the analytic policies and ~14 ms per vehicle for the DRL agent (measured, log
    section 13.25). That ratio is the entire reason this is a resident object rather
    than a script: a booth demo that ran a script per click would spend 3.4 of every
    4 seconds on imports the visitor is watching.

WHAT THIS IS NOT
    It imports neither TraCI nor SUMO, and never will. It takes plain dicts and lists
    and returns a plain dict, so `controller.py` can be developed against it before the
    simulator works at all. Run this file directly to exercise every path against
    fabricated state with no simulator present:

        python reroute_service.py                          # analytic policies only
        python reroute_service.py --drl checkpoints/taichung/drl_fusion_togo25.pt

IT DOES NOT MODIFY policies.py, AND MOSTLY DOES NOT REPLACE IT EITHER
    On the default path every policy here IS the function run_compare calls --
    policy_static, policy_prediction_greedy, policy_global_penalty, policy_drl -- with the
    same arguments. Nothing is re-implemented, so nothing can drift.

    Only the `seed_load=` opt-in needs more, because starting from a non-empty network is
    the one thing policies.py has no parameter for. The natural fix would be
    `init_load=` on RoutingEnv and policy_incremental -- a no-op for every existing
    caller -- and it is still not taken: policies.py backs numbers that are already
    reported (log 13.23 ATT -36.0%/-55.8%, 13.24, 13.27), reproduced by cloning the
    repository and running run_compare.py, and a demo feature does not get to edit that
    file. "Harmlessly" is an argument, and a reader would have to take it on trust.

    So that path is served locally: `_SeededEnv` subclasses RoutingEnv for policy 7, and
    `Router._incremental` transcribes policy_incremental for policy 6. The transcription
    is kept honest by a test rather than by care -- with an empty seed it must return
    exactly what pol.policy_global_penalty returns, and the self-test asserts it on every
    run, so a change to eq.4 fails loudly here instead of letting the demo and the report
    drift apart.

THE CONTRACT  (handover doc section 4.2 -- both sides speak SUMO edge ids)

    controller -> Router
        reroute(active, closed)                                 what to compute
            active = [(veh_id, current_edge_id, dest_osmid), ...]
            closed = [edge_id, ...]                             what the visitor shut
        network_state(load)                                     what to display
            load   = {edge_id: vehicles}                        SUMO's actual occupancy

    🔴 SUMO's load goes to the DISPLAY, not to the router. See the next section.

    active rows may carry a FOURTH element, the vehicle's remaining edge ids
    (getRoute() sliced at getRouteIndex()). With it, reroute(only_affected=True) touches
    only the vehicles a closure actually reaches -- less churn, and a small enough fleet
    to afford beam decoding.

    Router -> controller
        routes = {veh_id: [edge_id, ...]}                       new routes, ready for
                                                                traci.vehicle.setRoute()

    Every returned route BEGINS with the vehicle's current edge, which setRoute()
    requires. A vehicle absent from `routes` was not re-routed and must be left alone:
    absence means "keep going", never "no route exists".

WHY THE ROUTER DOES NOT READ SUMO'S LOAD
    The obvious design is to seed the re-route with the occupancy SUMO reports. It was
    built that way first, and measured (800 vehicles, the same fleet and graph both
    times, only the starting load different):

        starting load          policy 7 served      policies 1/4/6
        SUMO occupancy         345/800 greedy       774/800
                               632/800 beam-8       774/800
        empty (this design)    737/800 greedy       774/800
                               774/800 beam-8       774/800

    Two reasons, and the second is the real one.

    1. eq.4 couples vehicles THROUGH the load they leave behind -- "the load PERSISTS
       across them, so the penalty couples vehicles; that coupling is what suppresses the
       herding effect" (policies.RoutingEnv). Starting from zero, vehicle 50 finds the
       corridor already carrying rho = 0.35 and a spread penalty of ~58 s against an edge
       time of ~21 s, so it diverts. That feedback loop IS the mechanism being
       demonstrated, and it is strongest when the load it reads is the load it made.

    2. The model has no clock, so the two quantities cannot be added. Offline, load[e] is
       one consistent thing: traversals of e over the study period. Seeding adds SUMO's
       "entries in the last ~154 s" to an assignment covering each vehicle's whole
       remaining trip (~500 s) -- two different spans summed as if they were one. That is
       the granularity mismatch the handover flagged in section 3.2, and it does not have
       a clean fix at this level; it has one in mode B, where SUMO's state becomes the
       agent's observation and the BPR model goes away entirely.

    What is given up: the router reacts to the NETWORK changing (a visitor shutting a
    road), not to congestion the micro-simulation produces on its own. That is exactly the
    setting the reported numbers come from, so the demo and the report compute the same
    thing. `seed_load=` still exists for anyone who wants the other behaviour, with the
    cost above.

WHY THERE IS NO CLOCK HERE
    The assignment model has no time axis -- the S3 closure is defined on the dispatch
    ORDER, not on a wall clock (log section 13.17). So "who is still on the road" and
    "how full is that road" are questions only the simulator can answer, and they arrive
    as arguments. This module answers exactly one question: given that state, where to.

WHAT `load` HAS TO MEAN  (it still matters -- the displayed rho is compared to reported rho)
    Offline, `load[e]` counts every traversal of edge e over the whole dispatch, and
    `cap` is the CSV's veh/h times TAICHUNG_CAPACITY_SCALE. Those units only agree if
    the scale is read as an observation window:

        cap = veh/h * 0.0429  ==  vehicles per (0.0429 h)  ==  vehicles per 154 s

    so rho = load/cap is a flow ratio over ~154 s, and the live quantity that matches it
    is ENTRIES INTO THE EDGE OVER THE LAST ~154 s -- not the instantaneous vehicle count
    traci.edge.getLastStepVehicleNumber() returns. Hand `network_state()` the
    instantaneous count and the worst-rho on screen sits near zero while the report
    quotes 3.33; nothing errors, the panel is simply measuring something else.
    `LoadWindow` below maintains the right quantity from per-step observations.

    Honest caveat: 0.0429 was calibrated to land the herding baseline at a target
    worst-rho for 800 vehicles (handover doc section 6, item 4), not derived from a
    window. 154 s is therefore the window that scale IMPLIES, and it moves if the demo
    changes the vehicle count and recalibrates.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import deque

import networkx as nx
import numpy as np

import closure as clo
import config as C
import metrics as M
import network as net
import policies as pol
# Imported, not re-implemented. The premise of the whole A' integration is that our
# edge ids and SUMO's are the same string; two definitions of it in two files is exactly
# how that stops being true six weeks from now.
from export_sumo import edge_id

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# Seconds of edge entries that `load` should count. See the module docstring.
LOAD_WINDOW_S = 3600.0 * C.TAICHUNG_CAPACITY_SCALE

# Policies worth exposing live. 4 and 7 side by side ARE the demo (handover 5.3);
# 6 belongs on screen as a dashed upper bound, not as a competitor (handover 5.4).
POLICIES = ("static", "herding", "oracle", "drl")

# 臺灣大道一段 -> 臺灣大道. Both Chinese numerals and digits appear in OSM name tags.
_SEGMENT = re.compile(r"^(.+?)(?:[一二三四五六七八九十百]+|\d+)段$")


class LoadWindow:
    """Rolling count of edge ENTRIES over the last `window` seconds.

    The controller already calls traci.vehicle.getRoadID() for every vehicle to build
    `active`; hand the same snapshot here and this turns it into the quantity `rho` is
    actually defined against (module docstring). An edge entry is recorded when a
    vehicle's road id changes, so this counts flow, not standing vehicles.

        lw = LoadWindow()
        ...每步...  lw.observe(traci.simulation.getTime(), {v: traci.vehicle.getRoadID(v)
                                                            for v in traci.vehicle.getIDList()})
        state  = router.network_state(lw.counts())    # the panel
        routes = router.reroute(active, closed)       # the routing

    Internal edges (SUMO writes junction ids as ":something") are ignored: they are not
    edges of our graph and would only ever be dropped later.
    """

    def __init__(self, window=LOAD_WINDOW_S):
        self.window = float(window)
        self._events = deque()      # (t, edge_id), oldest first
        self._where = {}            # veh_id -> edge_id it was last seen on
        self.t = 0.0

    def observe(self, t, on_edge):
        """Record one simulation step. `on_edge` is {veh_id: edge_id}."""
        self.t = float(t)
        for veh, eid in on_edge.items():
            if not eid or eid.startswith(":"):
                continue            # junction-internal edge: not ours
            if self._where.get(veh) != eid:
                self._where[veh] = eid
                self._events.append((self.t, eid))
        gone = [v for v in self._where if v not in on_edge]
        for v in gone:
            del self._where[v]      # vehicle left the simulation
        self._expire()

    def _expire(self):
        cut = self.t - self.window
        while self._events and self._events[0][0] < cut:
            self._events.popleft()

    def counts(self):
        """{edge_id: entries within the window} -- the `load` argument to reroute()."""
        self._expire()
        out = {}
        for _, eid in self._events:
            out[eid] = out.get(eid, 0.0) + 1.0
        return out

    def __len__(self):
        return len(self._events)


class _SeededEnv(pol.RoutingEnv):
    """RoutingEnv that starts from the load SUMO reports instead of an empty network.

    WHY A SUBCLASS AND NOT A PARAMETER
        The obvious change is `RoutingEnv(..., init_load=None)` in policies.py. It would
        be a no-op for every existing caller -- but policies.py is the file backing the
        numbers already reported (log 13.23, 13.24, 13.27), and someone cloning the repo
        reproduces them by running it. A demo feature does not get to edit that file,
        even harmlessly. Everything the live loop needs lives here instead.

    WHERE THE HOOK GOES
        reset() clears load / _rho_sum and then calls _start_next_vehicle(), which is
        what snapshots the encoder context for vehicle 0. That call is the ONE instant
        where the network is empty and nothing has read it yet, so it is where the load
        gets seeded. Seeding after reset() returns would be too late: vehicle 0 would
        already have been encoded, and with togo_refresh its to-go Dijkstra would already
        have been run, on an empty network.

    WHAT MUST BE SEEDED TOGETHER
        _rho_sum is a RUNNING sum of per-edge saturation, not a derived quantity. A
        non-empty self.load with _rho_sum = 0 leaves _mean_rho -- and therefore eq.4's
        entire spread term, for the whole episode -- silently wrong rather than raising.
    """

    def __init__(self, g, demand, seed_load=None, **kw):
        self._seed_load = {k: float(v) for k, v in (seed_load or {}).items()}
        self._seeded = False
        super().__init__(g, demand, **kw)     # calls reset() -> _start_next_vehicle()

    def reset(self):
        self._seeded = False                  # re-arm: reset() clears the load again
        obs = super().reset()
        if not self._seeded:
            # The hook is a subclass reaching into a base class's control flow, so it is
            # only as durable as that flow. If reset() ever stops routing through
            # _start_next_vehicle(), the seeding silently does not happen and policy 7
            # re-routes on a network it believes is empty -- which looks like a working
            # demo. Fail here instead.
            raise RuntimeError(
                "_SeededEnv: RoutingEnv.reset() no longer calls _start_next_vehicle(), "
                "so the load was never seeded. Re-site the hook in reroute_service.py.")
        return obs

    def _start_next_vehicle(self):
        if not self._seeded:
            self._seeded = True
            self.load = {e: v for e, v in self._seed_load.items() if e in self._cap}
            self.total_load = float(sum(self.load.values()))
            self._rho_sum = float(sum(v / self._cap[e] for e, v in self.load.items()))
        super()._start_next_vehicle()


class _Banned:
    """A Closure that also refuses one extra edge -- the per-vehicle U-turn ban.

    Carries Closure's WHOLE interface, not just `blocked`. The analytic policies only
    call blocked() through the Dijkstra weight, but RoutingEnv also calls active() and
    cutoff() to decide when the closure lands; a partial stand-in would work for
    policies 1/4/6 and raise on 7. `at = 0` semantics: in force from the first vehicle,
    because a visitor's road is shut NOW.
    """

    def __init__(self, base, extra):
        self.base, self.extra = base, extra
        self.edges = frozenset({extra}) | (base.edges if base is not None else frozenset())
        self.at, self.label = 0.0, "uturn-ban"

    def blocked(self, u, v):
        return (u, v) == self.extra or (self.base is not None and self.base.blocked(u, v))

    def cutoff(self, n):
        return 0

    def active(self, vehicle_index, n):
        return True


class Router:
    """The graph, the agent, and one re-routing call. Build once, keep forever.

    Parameters mirror run_compare's so a demo number can be traced back to a reported
    one; `drl` is the checkpoint path and its `.meta.json` sidecar is read for
    `togo_refresh`, which changes what the policy OBSERVES and therefore may not be
    guessed (log section 13.16, item 6).
    """

    def __init__(self, drl=None, capacity_scale=None, max_hops=None, device=None,
                 beam=0, verbose=True):
        t0 = time.perf_counter()
        self.g, self.info = net.build_graph_for(
            "taichung", capacity_scale=capacity_scale, verbose=verbose)
        self.o2i = self.g.graph["osmid_to_idx"]
        self.i2o = self.g.graph["idx_to_osmid"]
        self.max_hops = max_hops or net.default_max_hops("taichung")
        self.edge_ids = {e: edge_id(self.g, *e) for e in self.g.edges()}
        self.by_id = {v: k for k, v in self.edge_ids.items()}
        self.agent = None
        self.device = device
        # Greedy by default: 13.9 ms/vehicle against beam-8's 90-118 (log 13.25 item 2).
        # Beam mainly buys served% back -- greedy dead-ends because _valid_neighbors
        # excludes visited nodes and the vehicle can corner itself. Raise it if the booth
        # can afford the wall clock; the two are different SETTINGS, so say which is on
        # screen rather than quoting the report's beam-8 numbers over a greedy run.
        self.beam = int(beam or 0)
        self.togo_refresh = 0
        self.trained_worst_rho = None     # what the checkpoint saw; from its meta sidecar
        self.last_stats = {}
        self._drl_fail = {"dead_end": 0, "max_hops": 0, "trivial": 0}
        self._roads = None
        if drl:
            self._load_agent(drl, verbose)
        self.startup_s = time.perf_counter() - t0
        if verbose:
            print(f"[router] ready in {self.startup_s:.2f} s -- "
                  f"{self.g.number_of_nodes():,} nodes / {self.g.number_of_edges():,} edges, "
                  f"policies: {', '.join(self.available())}")

    # ---------------------------------------------------------------- setup ---
    def _load_agent(self, path, verbose=True):
        meta_path = os.path.splitext(path)[0] + ".meta.json"
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        else:
            print(f"[router] WARNING no {os.path.basename(meta_path)} beside the "
                  f"checkpoint; togo_refresh defaults to 0, which is probably wrong "
                  f"and will degrade the agent silently rather than raise.")
        self.togo_refresh = int(meta.get("togo_refresh", 0) or 0)
        self.trained_worst_rho = (meta.get("eval") or {}).get("worst_rho")
        # capacity_scale enters rho, rho enters the observation. A mismatch does not
        # raise anywhere downstream -- it just makes the agent read a network it was
        # never trained on.
        trained_scale = meta.get("capacity_scale")
        live_scale = self.info.get("capacity_scale")
        if trained_scale is not None and abs(float(trained_scale) - float(live_scale)) > 1e-9:
            print(f"[router] WARNING capacity_scale mismatch: checkpoint trained at "
                  f"{trained_scale}, router running at {live_scale}. rho feeds the "
                  f"observation, so the agent is reading a different network.")
        self.agent = pol.make_drl_agent(path, self.g)
        # 🔴 make_drl_agent loads with map_location="cpu" and leaves it there. Offline
        # that is a choice; at a booth it is a 1.5x latency penalty nobody asked for --
        # 21.5 ms/vehicle against 13.9 (log 13.25), so 800 vehicles is 17 s instead of
        # 11. Nothing errors, the demo is just slower than the number in the handover.
        if self.device is None:
            self.device = ("cuda" if pol.torch.cuda.is_available() else "cpu")
        if self.device != "cpu":
            self.agent.to(self.device)
        if verbose:
            print(f"[router] agent {os.path.basename(path)} on {self.device} "
                  f"(togo_refresh={self.togo_refresh}, beam={self.beam or 'greedy'}, "
                  f"trained on {meta.get('nodes', '?')} nodes / "
                  f"{meta.get('edges', '?')} edges)")

    def available(self):
        return [p for p in POLICIES if p != "drl" or self.agent is not None]

    def warmup(self):
        """Pay the first-call costs now: CUDA context, the first reverse Dijkstra.

        Without this the first visitor click is several seconds slower than every one
        after it, which on a booth reads as "it crashed" rather than "it is warming up".
        """
        u, v = next(iter(self.g.edges()))
        dest = self.i2o[max(self.g.nodes(), key=lambda n: self.g.in_degree(n))]
        fake = [("__warmup__", self.edge_ids[(u, v)], dest)]
        for p in self.available():
            try:
                self.reroute(fake, (), policy=p)
            except Exception as exc:                  # a warm-up must never be fatal
                print(f"[router] warmup: {p} raised {type(exc).__name__}: {exc}")
        return self

    # ------------------------------------------------------------ id mapping ---
    def parse_edge(self, eid):
        """SUMO edge id -> (u, v) graph indices, or None if it is not ours.

        SUMO junction-internal edges (':...') and anything netconvert invented land
        here; returning None rather than raising keeps one stray id from killing a
        whole re-route.
        """
        return self.by_id.get(eid)

    def road_edges(self, prefix):
        """Road name (or prefix) -> [edge_id]. This is what POST /close sends us.

        Prefix, not equality: the arena names roads per segment, so '臺灣大道' takes the
        whole corridor and '臺灣大道三段' just the throat. Which of those to close is a
        measured question, not an aesthetic one -- the table is in closure.py's
        docstring, and the segment carrying the LEAST load does the most damage.
        """
        return [self.edge_ids[e] for e in clo.edges_by_road(self.g, prefix)]

    def roads(self, limit=12, min_edges=4):
        """Closable roads for the touch panel, with what each one would sever.

        sumo-gui has no API for clicking the canvas to shut a road (handover 4.5), so the
        visitor picks from a short list. `scc_frac` is the share of the arena still
        strongly connected afterwards: below ~0.85 the closure demolishes the network
        rather than diverting it, and every policy's served% collapses for a reason that
        has nothing to do with routing.

        WHOLE CORRIDORS ARE INCLUDED, not just segments. The arena names roads per
        segment (臺灣大道一段 … 四段), so grouping by the raw name offers the visitor
        `臺灣大道四段` and never `臺灣大道` -- and the whole corridor IS the demo (handover
        5.2). Aggregates are marked `corridor`; `road_edges()` takes either, since it
        matches by prefix.

        🔴 THE ORDER HERE MEANS NOTHING. Rows are sorted by edge count because something
        has to be first, and neither length nor load predicts disruption: 臺灣大道二段 is
        36 edges carrying 6.3% of baseline load and moves ATT by +4.7%, while 三段 is 20
        edges at 2.2% and moves it by +102.9%. The measured table is in closure.py's
        docstring -- pick the booth's buttons from THAT, not from this listing.
        """
        if self._roads is None:
            by_road = {}
            for u, v, d in self.g.edges(data=True):
                if d.get("road_name"):
                    by_road.setdefault(d["road_name"], []).append((u, v))
            corridor = {}
            for name, edges in by_road.items():
                m = _SEGMENT.match(name)
                if m:
                    corridor.setdefault(m.group(1), []).extend(edges)
            groups = [(n, e, False) for n, e in by_road.items()]
            groups += [(n, e, True) for n, e in corridor.items() if n not in by_road]
            n_nodes = self.g.number_of_nodes()
            rows = []
            for name, edges, is_corridor in groups:
                if len(edges) < min_edges:
                    continue
                big = max(nx.strongly_connected_components(
                    nx.restricted_view(self.g, [], edges)), key=len)
                rows.append({"road": name, "edges": len(edges),
                             "km": sum(self.g.edges[e]["length"] for e in edges) / 1000.0,
                             "corridor": is_corridor, "scc_frac": len(big) / n_nodes,
                             "safe": len(big) / n_nodes >= clo.MIN_SCC_FRACTION})
            self._roads = sorted(rows, key=lambda r: -r["edges"])
        return self._roads[:limit]

    # ------------------------------------------------------------- the call ---
    def reroute(self, active, closed=(), policy="drl", no_uturn=True,
                only_affected=False, seed_load=None):
        """{veh_id: [edge_id, ...]} for every vehicle that could be given a new route.

        `active` is [(veh_id, current_edge_id, dest_osmid), ...] and `closed` is
        [edge_id, ...]. SUMO's occupancy is deliberately NOT an argument here -- it goes
        to network_state() for the display. The module docstring has the measurement
        behind that (policy 7: 737/800 served from an empty start, 345/800 from a seeded
        one) and `seed_load=` is the opt-in for the other behaviour.

        Vehicles are OMITTED, never given an empty route, when they cannot or should not
        be re-routed: unknown edge, unknown destination, already on their final edge, or
        no path at all after the closure. `self.last_stats` breaks the omissions down --
        one "did not re-route" count cannot distinguish a bad id from a severed network,
        and on a booth those two look identical until you check.

        `only_affected` re-routes just the vehicles whose remaining route uses a closed
        edge. It needs a FOURTH element per row -- the vehicle's remaining edge ids, which
        the controller has from getRoute() + getRouteIndex() -- and a row without one is
        treated as affected, so the conservative case is the default. Two reasons to want
        it: rewriting the route of a car the closure never touched is churn a real
        navigation system would not produce, and it cuts the fleet enough to afford beam
        decoding (see _drl and the log, 13.23).

        ⚠️ `only_affected` shrinks WHAT IS INJECTED, and it also shrinks what is computed
        -- which weakens eq.4's coupling, since the vehicles left out no longer push the
        others off the corridor. Prefer computing the whole fleet and injecting only the
        affected ones, unless the wall clock forces otherwise.
        """
        if policy not in self.available():
            raise ValueError(f"unknown policy {policy!r}; have {self.available()}")
        t_start = time.perf_counter()
        self._drl_fail = {"dead_end": 0, "max_hops": 0, "trivial": 0}

        # --- closure ------------------------------------------------------------
        # at=0.0 makes it active from vehicle 0: a visitor's road is shut NOW, whereas
        # S3's `at` staggers it through the dispatch order. Masked, never removed --
        # the agent's edge_index is a buffer sized from this graph.
        cl_edges, bad_closed = [], 0
        for eid in closed or ():
            e = self.parse_edge(eid)
            if e is None:
                bad_closed += 1
            else:
                cl_edges.append(e)
        closure = clo.Closure(cl_edges, at=0.0, label="live") if cl_edges else None

        # --- optional starting load (NOT the default; see the module docstring) ---
        seed, bad_load = {}, 0
        for eid, n in (seed_load or {}).items():
            e = self.parse_edge(eid)
            if e is None:
                bad_load += 1
            else:
                seed[e] = float(n)

        # --- demand -------------------------------------------------------------
        demand, meta = [], []
        skip = {"bad_edge": 0, "bad_dest": 0, "at_dest": 0, "unaffected": 0}
        shut = set(closed or ()) if only_affected else set()
        for row in active:
            veh, cur_id, dest = row[0], row[1], row[2]
            e = self.parse_edge(cur_id)
            if e is None:
                skip["bad_edge"] += 1
                continue
            d = self.o2i.get(int(dest))
            if d is None:
                skip["bad_dest"] += 1
                continue
            # No remaining route supplied -> assume it is affected. Skipping a car that
            # IS heading into a closed road strands it; re-routing one that is not merely
            # costs time, so the ambiguous case takes the harmless side.
            if shut and len(row) > 3 and row[3] is not None and not (shut & set(row[3])):
                skip["unaffected"] += 1
                continue
            # The vehicle is ON (u, v), so its next decision is made AT v. Routing from
            # u instead would hand it a route it has already half-driven.
            if e[1] == d:
                skip["at_dest"] += 1        # on its final edge; leave it alone
                continue
            demand.append((e[1], d))
            meta.append((veh, e, cur_id))

        routes, no_path, uturn = {}, 0, 0
        if demand:
            paths = self._route(demand, policy, closure, seed)
            for (veh, e, cur_id), p in zip(meta, paths):
                if not p or len(p) < 2:
                    no_path += 1
                    continue
                if no_uturn and p[1] == e[0]:
                    # An immediate reversal is physically implausible and SUMO may simply
                    # refuse the route at the junction. Re-route this one vehicle with the
                    # reversal banned rather than dropping it -- dropping would leave it
                    # driving into the road the visitor just closed.
                    #
                    # 🔴 Under the SAME policy. Falling back to the forecast weight (as
                    # this first did) hands a share of policy 6's and 7's vehicles a
                    # policy-4 route -- 13-16% of them in the self-test -- which dilutes
                    # the exact contrast the demo exists to show, and does it invisibly.
                    p = self._route([(e[1], p[-1])], policy,
                                    _Banned(closure, (e[1], e[0])), seed, n_batches=1)[0]
                    uturn += 1
                    if not p or len(p) < 2:
                        no_path += 1
                        continue
                routes[veh] = [cur_id] + [self.edge_ids[(a, b)]
                                          for a, b in zip(p[:-1], p[1:])]

        # The saturation the AGENT is being asked to read, before it adds anything. Its
        # checkpoint records what it saw in training (meta eval.worst_rho = 2.26); far
        # past that it is extrapolating, and greedy decoding fails by cornering itself
        # rather than by routing badly. Reported rather than warned on, because a warning
        # on every click is noise -- the controller can put it on screen once.
        seed_rho = max((v / self.g.edges[e]["cap"] for e, v in seed.items()), default=0.0)
        self.last_stats = {
            "policy": policy, "active": len(active), "routed": len(routes),
            "no_path": no_path, "uturn_fixed": uturn, "closed_edges": len(cl_edges),
            "unknown_closed": bad_closed, "unknown_load": bad_load,
            "load_edges": len(seed), "seed_worst_rho": float(seed_rho),
            "trained_worst_rho": self.trained_worst_rho,
            "seconds": time.perf_counter() - t_start, **skip, **self._drl_fail,
        }
        return routes

    def plan(self, demand, policy="drl", closed=(), seed_load=None):
        """Route a whole fleet from scratch: [(origin, dest)] -> [node path or None].

        Node INDICES, not osmids, and node paths rather than edge ids -- this is the
        offline contract (`run_compare`'s), not the live one. `reroute()` is the live
        counterpart and does the id translation.

        Use this when the fleet is being assigned as a batch, which is the setting eq. 4
        was built for: vehicle k sees the load vehicles 0..k-1 left behind, and that
        coupling is what suppresses herding. Routing a trickle of one or two vehicles at a
        time through the same policy gets none of it -- every rho the agent reads is still
        zero -- so a demo that replaces arrivals individually shows policy 7 behaving like
        policy 1, whatever function it calls.
        """
        if policy not in self.available():
            raise ValueError(f"unknown policy {policy!r}; have {self.available()}")
        self._drl_fail = {"dead_end": 0, "max_hops": 0, "trivial": 0}
        cl = [e for e in (self.parse_edge(x) for x in closed or ()) if e]
        closure = clo.Closure(cl, at=0.0, label="live") if cl else None
        seed = {e: float(n) for e, n in
                ((self.parse_edge(k), v) for k, v in (seed_load or {}).items()) if e}
        t0 = time.perf_counter()
        paths = self._route(demand, policy, closure, seed)
        self.last_stats = {"policy": policy, "active": len(demand),
                           "routed": sum(1 for p in paths if p),
                           "closed_edges": len(cl), "uturn_fixed": 0,
                           "seconds": time.perf_counter() - t0, **self._drl_fail}
        return paths

    def _route(self, demand, policy, closure, seed, n_batches=None):
        if policy == "static":
            return pol.policy_static(self.g, demand, closure)
        if policy == "herding":
            # Load-independent BY DESIGN: every vehicle reads the same forecast and picks
            # the same fastest road. That is the herding mechanism, not a missing feature,
            # which is why `seed` is not passed here -- and why policy 4 needs nothing
            # from this file at all.
            return pol.policy_prediction_greedy(self.g, demand, closure=closure)
        if policy == "oracle":
            if seed:
                return self._incremental(demand, closure, seed, n_batches=n_batches)
            if n_batches:
                # The single-vehicle U-turn fallback. policy_global_penalty is fixed at
                # C.N_BATCHES, so for one vehicle it sweeps all 1,690 edge costs 30 times
                # for one Dijkstra -- measured at 4.5 s across 155 reversed vehicles.
                # policy_incremental is the same function one layer down and takes the
                # batch count. Equivalent, not an approximation: a batch with no vehicles
                # adds no load, so the next cost pass recomputes the same numbers.
                return pol.policy_incremental(self.g, demand, use_penalty=True,
                                              n_batches=n_batches, closure=closure)
            # With no starting load this IS policy_global_penalty, so call the real one.
            # The default demo path therefore runs the same function run_compare runs, and
            # the transcription below is reached only by the seed_load= opt-in.
            return pol.policy_global_penalty(self.g, demand, closure)
        return self._drl(demand, closure, seed)

    def _drl(self, demand, closure, seed):
        """Roll the agent out, starting from `seed` rather than an empty network.

        Line-for-line policies.policy_drl, with _SeededEnv in place of RoutingEnv.
        Greedy unless Router was built with beam>1: the report quotes beam-8 for quality,
        but greedy is what runs in 13.9 ms/vehicle and keeps a booth responsive (log
        13.25 item 2). Label the demo accordingly rather than quoting the report's
        beam-8 numbers over a greedy run.
        """
        # 🔴 The agent caches its graph encoding keyed by VEHICLE INDEX, and policy_drl
        # does not clear it. Offline that is harmless -- one episode runs 0..799 and the
        # next starts at 0 while the cache still says 799, so it re-encodes. Here every
        # call restarts at vehicle 0, so a call that routed exactly ONE vehicle (warmup,
        # or a lone diverted car) leaves _cv = 0 and the NEXT call's vehicle 0 silently
        # reuses an encoding of the network as it was before the visitor closed the road.
        if hasattr(self.agent, "reset_cache"):
            self.agent.reset_cache()
        fail = self._drl_fail
        if not seed:
            # Same reasoning as the oracle branch: with no starting load _SeededEnv is
            # RoutingEnv, so run policies.policy_drl itself. policy_drl's `stats` is an
            # update(), not an accumulate, so it lands in a scratch dict first -- the
            # U-turn guard re-enters this once per reversed vehicle.
            st = {}
            paths = pol.policy_drl(self.g, demand, self.agent, max_hops=self.max_hops,
                                   stats=st, togo_refresh=self.togo_refresh,
                                   closure=closure, beam=self.beam)
            for k in ("dead_end", "max_hops", "trivial"):
                fail[k] += st.get(k, 0)
            return paths
        env = _SeededEnv(self.g, demand, seed_load=seed, use_penalty=True,
                         max_hops=self.max_hops, togo_refresh=self.togo_refresh,
                         closure=closure)
        obs = env.reset()
        if self.beam > 1:
            # Same policy, wider decoding. beam=1 deliberately falls through to greedy:
            # the two must agree, and routing it here would hide a divergence.
            while not env.done:
                env.commit(*env.beam_route(self.agent, self.beam, self.max_hops))
        else:
            while not env.done and obs is not None:
                obs, _, _, _ = env.step(self.agent.act(obs, greedy=True))
        # 🔴 "Not served" has three causes needing three different fixes, and one number
        # cannot tell them apart (log 13.23). A dead end means the policy cornered itself
        # and wants a wider beam; max_hops means it wandered and wants a longer budget or
        # a to-go estimate it can still read; trivial means the trip was never routable.
        # Accumulated, not assigned: the U-turn guard re-enters this per vehicle.
        fail["dead_end"] += env.n_deadend
        fail["max_hops"] += env.n_maxhops
        fail["trivial"] += env.n_trivial
        return env.paths

    def _incremental(self, demand, closure, seed, use_penalty=True, n_batches=None):
        """policies.policy_incremental, starting from `seed` rather than an empty network.

        🔴 TRANSCRIBED ON PURPOSE, and the duplication is the lesser evil. The alternative
        was an `init_load=` parameter threaded through policies.py -- but policies.py now
        backs published numbers (log 13.23, 13.24, 13.27), and a demo feature is not worth
        even a no-op edit to the file a reader clones to reproduce them.

        The duplication is made safe by a test rather than by care: with an EMPTY seed
        this must return exactly what pol.policy_global_penalty returns, and
        `python reroute_service.py` asserts it. Change eq.4 in policies.py and this
        file's self-test fails -- loudly, in the demo, instead of the demo and the report
        quietly drifting apart.

        The one changed line is marked below.
        """
        g = self.g
        # Capped at the vehicle count only so the single-vehicle U-turn fallback does not
        # sweep all 1,690 edge costs 30 times for one Dijkstra. Equivalent, not an
        # approximation: a batch with no vehicles in it adds no load, so the cost pass at
        # the top of the next iteration recomputes the same numbers.
        n_batches = min(n_batches or C.N_BATCHES, max(1, len(demand)))
        edges = list(g.edges())
        t0 = {e: g.edges[e]["t0"] for e in edges}
        cap = {e: g.edges[e]["cap"] for e in edges}
        load = {e: float(seed.get(e, 0.0)) for e in edges}      # <-- the only difference
        t_ref = float(np.mean([t0[e] for e in edges])) if edges else 1.0
        cut = closure.cutoff(len(demand)) if closure is not None else len(demand)
        w_closed = pol._closed_weight(closure, "cost") if closure is not None else None

        paths = [None] * len(demand)
        for batch in np.array_split(np.arange(len(demand)), n_batches):
            rho = {e: load[e] / cap[e] for e in edges}
            mean_rho = float(np.mean([rho[e] for e in edges])) if edges else 0.0
            for e in edges:
                cost = C.ALPHA * pol._bpr(t0[e], load[e], cap[e])
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

    # ------------------------------------------------------------ read-outs ---
    def network_state(self, load, top=None):
        """What GET /state should report about the ROADS, from SUMO's own load.

        worst_rho, gini_load and frac_saturated are pure functions of the load vector,
        so they are honest here. ATT is NOT: offline it is the realised BPR time of the
        paths the model assigned, and live the only true answer is what SUMO measured
        (tripinfo, or arrival minus depart). Reporting a BPR-modelled ATT next to a
        running simulation would put a number on screen that the visible cars contradict.

        The Gini reference set is EVERY arena edge, fixed forever. The report's Gini uses
        the union of edges the seven policies touched in that run (log section 13.27),
        so the two are not comparable -- do not print this number beside a reported one.
        """
        rho, veh = {}, {}
        for e in self.g.edges():
            n = float((load or {}).get(self.edge_ids[e], 0.0))
            veh[e] = n
            rho[e] = n / self.g.edges[e]["cap"]
        vals = list(rho.values()) or [0.0]
        items = sorted(rho.items(), key=lambda kv: -kv[1])
        if top:
            items = items[:top]
        return {
            "worst_rho": float(max(vals)),
            "gini_load": M.gini(list(veh.values())),
            "frac_saturated": float(np.mean([r > C.RHO_THRESHOLD for r in vals])),
            "vehicles": float(sum(veh.values())),
            "gini_ref": "all arena edges (NOT the report's per-run union)",
            "edges": {self.edge_ids[e]: {"veh": veh[e], "rho": round(r, 4)}
                      for e, r in items if veh[e] > 0},
        }

    def geometry(self, detail=None):
        """{edge_id: [[lat, lon], ...]} -- enough for a Leaflet polyline layer.

        Sent once at page load; after that only network_state() goes over the wire.

        Two points per edge by default, because an arena edge is a MERGED CHAIN and only
        its endpoints survive into arena_edges_taichung.csv. That is accurate for 95% of
        edges (real length / chord is 1.000 at the median) and visibly wrong for the long
        ones -- 十甲東路 is 4,279.5 m of road drawn as a 3,391 m straight line.

        `detail` is an optional {edge_id: [[lat, lon], ...]} that replaces the chord where
        the real shape is known; demo/build_geometry.py recovers it. Purely cosmetic:
        `length` and `t0` come from the CSV's measured length_m, and no routing or metric
        code reads coordinates.
        """
        out = {}
        for u, v in self.g.edges():
            eid = self.edge_ids[(u, v)]
            shape = (detail or {}).get(eid)
            out[eid] = shape or [[self.g.nodes[u]["lat"], self.g.nodes[u]["lon"]],
                                 [self.g.nodes[v]["lat"], self.g.nodes[v]["lon"]]]
        return out


# --------------------------------------------------------------------------- #
# self-test: everything above, against fabricated state, with no simulator
# --------------------------------------------------------------------------- #
def _fake_state(router, n, rng):
    """Put `n` vehicles partway along routes to the hotspot hubs.

    Origins come from the largest SCC so the trips are routable, and destinations from
    the four highest-in-degree nodes -- the same funnel `run_compare.make_demand` uses,
    because that concentration is what produces herding in the first place.

    🔴 Each vehicle is placed PARTWAY ALONG a route to its hub, not on a random edge.
    A car in the demo has been driving toward its destination, so the shortest path from
    where it is rarely doubles back through the edge it is on. Random placement (which
    this did first) reversed 20% of the fleet and charged the U-turn fallback for every
    one of them -- 155 single-vehicle re-routes, which is a property of the fabrication,
    not of the demo.

    Rows carry the fourth contract element (remaining edge ids), so only_affected works.
    """
    scc = sorted(net.largest_scc(router.g))
    hubs = sorted(scc, key=lambda x: router.g.in_degree(x), reverse=True)[:C.N_HOTSPOTS]
    dests = [int(rng.choice(hubs)) for _ in range(n)]
    paths = pol.policy_prediction_greedy(
        router.g, [(int(rng.choice(scc)), d) for d in dests])
    active, on = [], {}
    for i, (p, d) in enumerate(zip(paths, dests)):
        if not p or len(p) < 3:
            continue                       # too short to be partway through anything
        edges = [router.edge_ids[e] for e in zip(p[:-1], p[1:])]
        k = int(rng.integers(0, len(edges) - 1))     # never the last edge: that is at_dest
        active.append((f"v{i}", edges[k], router.i2o[d], edges[k:]))
        on[f"v{i}"] = edges[k]
    return active, on, paths


def _fake_load(router, paths, rng, window=LOAD_WINDOW_S):
    """Edge entries within ONE window, which is what SUMO would report -- not whole trips.

    🔴 Counting every edge of every route (which this first did) overstates the load by
    roughly trip_duration / window. At 800 vehicles it produced 26,040 traversals and a
    seeded worst-rho of 3.12 -- already the report's END state for the herding baseline --
    and the router was then asked to lay 800 more trips on top of it. Policy 7 duly
    collapsed to 122/800 served while the analytic policies, which have no learned
    regime to leave, did not notice.

    A vehicle contributes only the edges it entered in the last `window` seconds, so it
    covers about len(path) * window / trip_time of its route. Free-flow t0 is used for
    the trip time, which slightly overestimates the slice (a congested trip is slower and
    therefore covers fewer edges) -- deliberately the conservative direction.

    Still an upper bound in one respect: every vehicle is treated as on the road at once,
    whereas a real run has some finished and some not yet departed.
    """
    load = {}
    for p in paths:
        if not p or len(p) < 2:
            continue
        edges = list(zip(p[:-1], p[1:]))
        t_trip = sum(router.g.edges[e]["t0"] for e in edges)
        k = max(1, min(len(edges), round(len(edges) * window / max(t_trip, 1e-9))))
        s = int(rng.integers(0, len(edges) - k + 1))
        for e in edges[s:s + k]:
            eid = router.edge_ids[e]
            load[eid] = load.get(eid, 0.0) + 1.0
    return load


def main():
    ap = argparse.ArgumentParser(description="Exercise Router with fabricated state.")
    ap.add_argument("--drl", default=None, metavar="CKPT.pt")
    ap.add_argument("--vehicles", type=int, default=100)
    ap.add_argument("--close-road", default="臺灣大道")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="default: cuda when available. On CPU the agent runs at "
                         "21.5 ms/vehicle instead of 13.9 (log 13.25)")
    ap.add_argument("--beam", type=int, default=0, metavar="W",
                    help="beam width for policy 7 (default greedy). Buys served%% back "
                         "from greedy's dead-ends at 6-8x the wall clock")
    cli = ap.parse_args()
    rng = np.random.default_rng(cli.seed)

    print(f"\n{'=' * 84}\nreroute_service self-test -- no SUMO, no TraCI\n{'=' * 84}")
    r = Router(drl=cli.drl, device=cli.device, beam=cli.beam).warmup()
    pols = r.available()

    # 1. edge ids agree with what the exporter wrote into the .net.xml
    sample = list(r.g.edges())[:200]
    assert all(r.parse_edge(r.edge_ids[e]) == e for e in sample), "edge id round trip"
    print(f"\n[1] edge ids: round trip ok on {len(sample)} edges; "
          f"'{r.edge_ids[sample[0]]}' -> {sample[0]}")

    # 2. the closable-road menu
    print(f"\n[2] closable roads (scc_frac < {clo.MIN_SCC_FRACTION:.2f} = demolition; "
          f"order is NOT impact -- see closure.py):")
    for row in r.roads(limit=6):
        print(f"      {clo._pad(row['road'], 14)}{row['edges']:>4} edges "
              f"{row['km']:>7.2f} km  scc {row['scc_frac']:>5.1%}  "
              f"{'ok' if row['safe'] else 'UNSAFE':<6} "
              f"{'<- whole corridor' if row['corridor'] else ''}")

    # 3. one re-route per policy. `load` here is what SUMO would be reporting -- what the
    #    herding baseline has put on the network by the time the visitor clicks -- and it
    #    is used for the DISPLAY panel only, never handed to the router.
    active, on, fleet = _fake_state(r, cli.vehicles, rng)
    dem = [(r.parse_edge(e)[1], r.o2i[int(d)]) for _, e, d, _ in active]
    load = _fake_load(r, fleet, rng)

    # 🔴 The transcription guard. _incremental() is a copy of policy_incremental with
    # one line changed, taken so that policies.py -- which backs the reported numbers --
    # needs no edit for the demo. With an empty seed the copy must BE the original. If
    # eq.4 ever changes in policies.py, this fails here rather than letting the demo and
    # the report drift apart without anyone noticing.
    assert r._incremental(dem, None, {}) == pol.policy_global_penalty(r.g, dem), \
        "reroute_service._incremental has drifted from policies.policy_global_penalty"
    print(f"\n[3a] _incremental with an empty seed == policy_global_penalty: ok "
          f"({len(dem)} trips, eq.4 unchanged)")

    def row(label, note="", **kw):
        s = r.last_stats
        print(f"      {label:<11} routed {s['routed']:>4}/{s['active']:<4} "
              f"no_path {s['no_path']:>3} (dead_end {s['dead_end']:>3} "
              f"max_hops {s['max_hops']:>3})  uturn {s['uturn_fixed']:>3}  "
              f"{s['seconds'] * 1000:>7.1f} ms   {note}")

    shown = r.network_state(load)
    print(f"\n[3] {len(active)} vehicles; the display panel would show "
          f"{len(load)} occupied edges, {sum(load.values()):.0f} entries in "
          f"{LOAD_WINDOW_S:.0f} s, worst-rho {shown['worst_rho']:.3f}. "
          f"The router is NOT given that load:")
    base = {}
    for p in pols:
        base[p] = r.reroute(active, (), policy=p)
        row(p)

    # 🔴 The rejected alternative, kept as a measured row rather than a claim. Seeding the
    # router with SUMO's occupancy is the design this file started with; it costs policy 7
    # most of its served% and buys nothing the display does not already give. Policies
    # 1/4/6 are unmoved, which is why the row is worth printing next to them.
    if r.agent is not None:
        r.reroute(active, (), policy="drl", seed_load=load)
        row("drl seeded", "<- rejected: router fed SUMO's load (see the docstring)")

    # 4. every route must start on the vehicle's current edge -- setRoute demands it
    first = pols[0]
    bad = [v for v, route in base[first].items() if route[0] != on[v]]
    assert not bad, f"{len(bad)} routes do not start on the vehicle's current edge"
    ok = all(r.parse_edge(a)[1] == r.parse_edge(b)[0]
             for route in base[first].values()
             for a, b in zip(route[:-1], route[1:]))
    assert ok, "a route is not edge-contiguous"
    print(f"\n[4] every route starts on the vehicle's current edge and is contiguous: ok")

    # 5. close a road and confirm the answer actually changes
    shut = r.road_edges(cli.close_road)
    print(f"\n[5] closing {cli.close_road} ({len(shut)} edges):")
    for p in pols:
        after = r.reroute(active, shut, policy=p)
        s = r.last_stats
        changed = sum(1 for v in after if v in base[p] and after[v] != base[p][v])
        via = sum(1 for route in after.values() if set(route[1:]) & set(shut))
        print(f"      {p:<11} routed {s['routed']:>4}  changed {changed:>4}  "
              f"still using a closed edge {via:>3}  {s['seconds'] * 1000:>7.1f} ms")
        assert via == 0, f"{p} routed {via} vehicles onto a closed edge"

    # 5a. only_affected: how much of the fleet a closure actually reaches. This is the
    #     number that decides whether beam decoding is affordable at the booth -- and
    #     rewriting the route of a car the incident never touched is churn no real
    #     navigation system would produce. Each row's fourth element is the rest of the
    #     herding path the vehicle was following, which is what it would be in the demo.
    r.reroute(active, shut, policy=pols[0], only_affected=True)
    s = r.last_stats
    touched = s["active"] - s["unaffected"]
    print(f"\n[5a] only_affected: {touched}/{s['active']} vehicles "
          f"({touched / max(1, s['active']):.0%}) have a closed edge still ahead of them; "
          f"the other {s['unaffected']} keep their route")

    # 6. LoadWindow: entries are counted once per edge and expire out of the window
    lw = LoadWindow(window=10.0)
    lw.observe(0.0, on)                       # every vehicle enters its edge
    n_enter = len(lw)
    lw.observe(1.0, on)                       # nobody moved -> no new entries
    same = len(lw)
    lw.observe(20.0, {})                      # 20 s later, all of it is out of window
    print(f"\n[6] LoadWindow(10 s): {n_enter} entries recorded, "
          f"{same} after a step where nothing moved, {len(lw)} after the window passed")
    assert n_enter == same == len(on) and len(lw) == 0, "LoadWindow accounting"

    # 7. what GET /state would return
    st = r.network_state(load, top=3)
    print(f"\n[7] network_state: worst_rho {st['worst_rho']:.3f}  "
          f"gini {st['gini_load']:.4f}  saturated {st['frac_saturated']:.1%}  "
          f"({st['vehicles']:.0f} vehicles, {len(st['edges'])} edges shown)")
    print(f"      live load window = {LOAD_WINDOW_S:.0f} s of edge entries "
          f"(capacity_scale {C.TAICHUNG_CAPACITY_SCALE} read as an observation window)")
    print(f"      geometry: {len(r.geometry()):,} polylines for the Leaflet layer")

    print(f"\nall checks passed. controller.py can be written against this today; "
          f"nothing here needs SUMO.\n")


if __name__ == "__main__":
    main()
