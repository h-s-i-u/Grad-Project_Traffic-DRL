"""Mid-run arterial closure — scenario S3 of 實驗設計 §4.3.

The proposal opens with this situation: an incident shuts an arterial, every
navigation app hands its users the same new fastest route at the same moment, and
the detour jams. S2 (the hotspot funnel) produces herding from demand alone; S3
produces the SECOND wave, where the trigger is a change in the network itself.

Timing is what makes it S3 rather than "a different road network": the closure
lands after a fraction of the demand has already been dispatched, so vehicles that
set off earlier keep the route they were given and only later ones see the new
graph. Closing from t=0 would just be a smaller arena.

    Measured on the arena (800 vehicles, seed 42, herding baseline, common
    766-trip subset, closed from the start = upper bound):

        closure           edges   load%   ATT      dATT      Gini
        (open)                0   0.00%    723.3    +0.0%   0.6688
        臺灣大道              83   8.81%   1479.8  +104.6%   0.7074
        臺灣大道二段          36   6.32%    757.3    +4.7%   0.6820
        臺灣大道三段          20   2.24%   1467.7  +102.9%   0.6982

    Load share does NOT predict disruption: the busiest segment (二段) has
    parallel roads and barely matters, while 三段 is a throat carrying a quarter
    of that load and accounts for almost the whole effect. Pick with `--close-list`,
    not by intuition.

MASK, DO NOT REMOVE. `EGATActorCritic` registers `edge_index` and `edge_static` as
BUFFERS sized from the graph it is built with, and `make_drl_agent` builds the model
from `g` before loading the checkpoint. Handing it a graph with 83 fewer edges is a
state-dict shape mismatch. Masking also keeps the load that wave-1 vehicles left on
the closed road visible in `edge_rho`, which is what actually happened.

Three call sites consume a Closure (see policies.py):
    _route_fixed          policies (1)(2)(3)(4)   -- callable Dijkstra weight
    policy_incremental    policies (5)(6)         -- callable Dijkstra weight
    RoutingEnv            policy (7)              -- action mask AND to-go

The third is the one that silently breaks: `_dist_to_dest` runs a reverse Dijkstra
that must route AROUND the closure, or the agent's only sense of "how far is left"
points down a road that no longer exists.
"""
import unicodedata

import networkx as nx

# A closed road has to carry real traffic in the undisturbed baseline, or "closing"
# it is not a disruption and S3 measures nothing. 臺灣大道四段 is 7.7 km of arena
# edge that no route uses: closing it moves ATT by 0.0%.
MIN_LOAD_SHARE = 0.02
# Closing any whole road disconnects part of the arena (the arena is a SUBGRAPH: the
# roads that would still connect those blocks were pruned). Below this the closure is
# demolishing the network rather than diverting it.
MIN_SCC_FRACTION = 0.85


class Closure:
    """A set of edges that becomes untraversable partway through the demand.

    `at` is a FRACTION of the dispatch sequence, not a wall-clock time: vehicle k
    sees the closure iff k >= round(at * n). The assignment model has no clock, and
    dispatch order is the only ordering it has.
    """

    def __init__(self, edges, at=0.5, label="closure"):
        # Out of range does not raise on its own: at > 1 makes cutoff exceed the
        # demand so the closure never fires, and at < 0 makes it fire from vehicle 0.
        # Either way the run completes and prints a table that silently answers a
        # different question than the flag asked.
        if not 0.0 <= float(at) <= 1.0:
            raise ValueError(f"--close-at is a fraction of the dispatch sequence and "
                             f"must be in [0, 1]; got {at}")
        self.edges = frozenset(edges)
        self.at = float(at)
        self.label = label

    def __len__(self):
        return len(self.edges)

    def cutoff(self, n):
        return int(round(self.at * n))

    def active(self, vehicle_index, n):
        return vehicle_index >= self.cutoff(n)

    def blocked(self, u, v):
        return (u, v) in self.edges

    def view(self, g):
        """Read-only view of `g` with the closed edges gone (no copy).

        For connectivity questions and demand feasibility only. Do NOT build an
        agent on it -- see the module docstring.
        """
        return nx.restricted_view(g, [], list(self.edges))


# --------------------------------------------------------------------------- #
# choosing what to close
# --------------------------------------------------------------------------- #
def _road_names(g):
    """{(u, v): road_name}. Empty if the graph carries no names (METR-LA)."""
    return {(u, v): d["road_name"] for u, v, d in g.edges(data=True) if d.get("road_name")}


def edges_by_road(g, prefix):
    """Edges whose road_name starts with `prefix`.

    PREFIX, not equality: the arena names roads per segment (臺灣大道一段 ...
    臺灣大道四段), so `臺灣大道` closes the whole corridor and `臺灣大道三段`
    closes just that segment. One flag, both granularities.
    """
    names = _road_names(g)
    if not names:
        raise ValueError("this graph carries no road_name (only --graph taichung does)")
    hits = [e for e, rn in names.items() if rn.startswith(prefix)]
    if not hits:
        near = sorted({rn for rn in names.values() if prefix[:2] in rn})[:8]
        raise ValueError(f"no road matches {prefix!r}"
                         + (f"; did you mean: {', '.join(near)}" if near else ""))
    return hits


def edges_by_endpoints(g, spec):
    """--close-edge FROM,TO given in ORIGINAL OSM ids (what the CSV and map show)."""
    try:
        a, b = (int(x) for x in spec.split(","))
    except ValueError:
        raise ValueError(f"--close-edge wants 'FROM,TO' in OSM ids, got {spec!r}")
    o2i = g.graph.get("osmid_to_idx", {})
    u, v = o2i.get(a), o2i.get(b)
    if u is None or v is None or not g.has_edge(u, v):
        raise ValueError(f"({a}, {b}) is not an edge of this graph")
    return [(u, v)]


def baseline_load(g, demand, policy):
    """Edge load left by `policy` on the UNDISTURBED graph -> the load-share guard."""
    import metrics as M
    load, _ = M.edge_loads(g, policy(g, demand))
    return load, max(1.0, sum(load.values()))


def survey(g, demand, load, total, hubs=(), limit=20):
    """Per-road impact table for --close-list.

    Reachability here uses "both endpoints inside the post-closure SCC", which is a
    sufficient condition rather than an exact one (a node outside the SCC may still
    reach into it). It is used for the MENU only; the demand filter that decides
    real numbers uses exact reachability. Being slightly pessimistic in a listing is
    fine; being wrong in the filter is not.
    """
    names = _road_names(g)
    by_road = {}
    for e, rn in names.items():
        by_road.setdefault(rn, []).append(e)
    rows = []
    n = g.number_of_nodes()
    for rn, edges in by_road.items():
        share = sum(load.get(e, 0.0) for e in edges) / total
        if share <= 0:
            continue
        h = nx.restricted_view(g, [], edges)
        big = max(nx.strongly_connected_components(h), key=len)
        rows.append({
            "road": rn,
            "edges": len(edges),
            "km": sum(g.edges[e]["length"] for e in edges) / 1000.0,
            "load_share": share,
            "scc": len(big),
            "scc_frac": len(big) / n,
            "hubs_ok": all(x in big for x in hubs),
            "infeasible": sum(1 for a, b in demand if a not in big or b not in big),
        })
    rows.sort(key=lambda r: -r["load_share"])
    return rows[:limit]


def busiest_road(g, demand, load, total, hubs=()):
    """Data-driven pick: heaviest road that survives both guards.

    The guards are not decoration. The single busiest road in the arena is 松竹路
    (20.5% of baseline load) and closing it severs all four hotspot hubs, leaving
    269 of 799 trips with no route at all -- every policy's served% collapses for a
    reason that has nothing to do with the policies. The heaviest road is often the
    one that must stay open.
    """
    for r in survey(g, demand, load, total, hubs, limit=10_000):
        if r["hubs_ok"] and r["scc_frac"] >= MIN_SCC_FRACTION:
            return r["road"]
    raise ValueError("no road passes the hub / connectivity guards")


# --------------------------------------------------------------------------- #
# guards + demand handling
# --------------------------------------------------------------------------- #
def inspect(g, closure, demand, load, total, hubs=()):
    """Everything a run should print about the closure before trusting its numbers."""
    h = closure.view(g)
    big = max(nx.strongly_connected_components(h), key=len)
    n = g.number_of_nodes()
    infeasible = _infeasible(h, demand)
    return {
        "label": closure.label,
        "n_edges": len(closure),
        "km": sum(g.edges[e]["length"] for e in closure.edges) / 1000.0,
        "load_share": sum(load.get(e, 0.0) for e in closure.edges) / total,
        "at": closure.at,
        "scc": len(big),
        "scc_frac": len(big) / n,
        "isolated": n - len(big),
        "hubs_ok": all(x in big for x in hubs),
        "infeasible": len(infeasible),
        "infeasible_frac": len(infeasible) / max(1, len(demand)),
    }


def _infeasible(h, demand):
    """Indices of trips with no route in the closed graph `h`.

    Grouped by destination: the hotspot scenario has 4 of them, so this is 4 reverse
    BFS instead of one per vehicle.
    """
    reach = {}
    bad = []
    for k, (o, d) in enumerate(demand):
        anc = reach.get(d)
        if anc is None:
            anc = reach[d] = nx.ancestors(h, d) | {d}
        if o not in anc:
            bad.append(k)
    return bad


def select_demand(g, closure, demand, mode):
    """Apply the S3 demand policy. Returns (demand, info).

    Closing an arterial splits the arena into 780 + 60 nodes, and ~3.8% of S2's
    trips lose every route. That is an artefact of the arena, not of the closure:
    57 of those 60 nodes are still connected on the FULL 9,904-node network (which
    only drops to 99.2% under the same closure). Scoring those trips as failures
    would report our own graph pruning as a consequence of the incident -- and it
    would push policy (7) under the 95% served threshold, flagging the whole row
    NOT COMPARABLE for a reason unrelated to the agent.

        filter    (default) drop trips that are infeasible after the closure.
                  The demand stays independent of --close-at, so a sweep over
                  closure timing compares the same vehicles.
        wave      drop them only if dispatched AFTER the closure. Physically
                  tighter (an early vehicle really did get through) and keeps ~2%
                  more trips, but the demand then moves with --close-at.
        resample  handled in make_demand: draw origins from the post-closure SCC.
        none      no filtering; infeasible trips count as unserved.
    """
    if closure is None or mode in ("none", "resample"):
        return demand, {"mode": mode, "dropped": 0}
    bad = set(_infeasible(closure.view(g), demand))
    if mode == "wave":
        cut = closure.cutoff(len(demand))
        bad = {k for k in bad if k >= cut}
    elif mode != "filter":
        raise ValueError(f"unknown --close-demand {mode!r}")
    kept = [d for k, d in enumerate(demand) if k not in bad]
    return kept, {"mode": mode, "dropped": len(bad),
                  "dropped_frac": len(bad) / max(1, len(demand))}


def post_closure_pool(g, closure):
    """Origins for --close-demand resample: the SCC that survives the closure."""
    return sorted(max(nx.strongly_connected_components(closure.view(g)), key=len))


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _pad(s, width):
    """Left-justify to a DISPLAY width.

    Road names are Chinese, so `{:<16}` pads by character count and every column
    after it walks left by one space per glyph. These tables are read to make a
    decision; a misaligned one gets misread.
    """
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(0, width - w)


def fmt_survey(rows):
    head = (f"  {_pad('road', 18)}{'edges':>6}{'km':>7}{'load%':>8}"
            f"{'SCC after':>11}{'hubs':>8}{'infeas':>8}")
    out = [head, "  " + "-" * (len(head) - 2)]
    for r in rows:
        out.append(f"  {_pad(r['road'], 18)}{r['edges']:>6}{r['km']:>7.2f}"
                   f"{r['load_share']:>7.1%}{r['scc_frac']:>10.1%}"
                   f"{'ok' if r['hubs_ok'] else 'BROKEN':>8}{r['infeasible']:>8}")
    out += ["",
            "  load% = share of the herding baseline's edge traversals on the open network.",
            "  It does NOT predict disruption -- a busy road with good parallels barely",
            "  matters, a quiet throat can double ATT. hubs=BROKEN means the closure cuts",
            "  the hotspot destinations off, which collapses served% for every policy."]
    return "\n".join(out)


def fmt_inspect(info):
    lines = [
        f"[S3] closing {info['label']}: {info['n_edges']} edges / {info['km']:.1f} km, "
        f"{info['load_share']:.2%} of the herding baseline's load, "
        f"at {info['at']:.0%} of dispatch",
        f"[S3] after closure: largest SCC {info['scc']} "
        f"({info['scc_frac']:.1%}), {info['isolated']} nodes isolated, "
        f"hubs {'intact' if info['hubs_ok'] else 'BROKEN'}",
    ]
    if info["load_share"] < MIN_LOAD_SHARE:
        lines.append(f"  ⚠ that road carries under {MIN_LOAD_SHARE:.0%} of baseline load — "
                     f"closing it is not a disruption and the deltas will be noise.")
    if not info["hubs_ok"]:
        lines.append("  ⚠ the closure cuts off hotspot destinations: every policy's "
                     "served% drops for reasons unrelated to routing. Pick another road.")
    elif info["scc_frac"] < MIN_SCC_FRACTION:
        lines.append(f"  ⚠ only {info['scc_frac']:.0%} of the arena remains connected; "
                     f"this is demolition rather than diversion.")
    return lines


def fmt_demand(info, n_before):
    if not info.get("dropped"):
        return []
    return [
        f"[S3] demand: {info['dropped']}/{n_before} trips ({info['dropped_frac']:.1%}) "
        f"have no route after the closure and were removed (--close-demand {info['mode']}).",
        "     Those origins are isolated in the ARENA only — 95% of them stay connected "
        "on the full",
        "     9,904-node network, which the same closure leaves 99.2% connected. Scoring "
        "them as",
        "     failures would report the arena's pruning as a result of the incident.",
    ]
