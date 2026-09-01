#!/usr/bin/env python3
"""Recover the road shape of each arena edge, for drawing only.

    cd demo && python build_geometry.py          # writes arena_geometry.json

WHY THIS EXISTS
    An arena edge is a MERGED CHAIN. build_simplified_network.py collapses runs of
    pass-through nodes (39,920 -> 9,904) and build_arena.py simplifies again after taking
    the subgraph (1,442 -> 840, log 13.13). Only the two endpoints survive into
    arena_edges_taichung.csv -- there is no geometry column -- so anything drawing the
    network from that file draws every road as a straight chord.

    Measured over all 1,690 arena edges, real length / chord length is 1.000 at the
    median and 1.012 on average, so most are visually fine. The exceptions are the long
    ones, which are also the conspicuous ones:

        十甲東路     4,279.5 m of road drawn as a 3,391 m straight line   1.26x
        環中路三段   1,687.5 m drawn as 1,167 m                           1.45x
        雷中街       2,039.1 m drawn as 1,710 m                           1.19x

    A 4.3 km straight line across the city does not look like a road, and reads as a data
    error when it is a rendering one.

WHAT IT DOES NOT AFFECT
    Nothing but the picture. `length_m` is the measured road length and is what `t0` is
    built from; export_sumo.py writes an explicit `length` per edge so SUMO uses the
    measured value rather than the drawn shape; routing, rho, ATT and Gini never read
    coordinates at all.

HOW IT RECOVERS THE SHAPE
    🔴 NOT by shortest path. The chain that was merged is not generally the shortest route
    between its endpoints -- on a 4 km corridor the full network almost always offers
    something shorter -- so a plain Dijkstra returns a different road and the length check
    (rightly) throws it away. Measured: that approach recovered 72% of edges and failed on
    essentially every edge over 1 km, which is the entire set worth fixing.

    The actual constraint is exact and comes from how simplification works. A node is
    merged away precisely when it is a pass-through, so the interior of a merged edge is
    made of nodes that exist in the PARENT graph and not in the CHILD:

        interior of an arena edge       subset of  simplified nodes - arena nodes
        interior of a simplified edge   subset of  Map_fined nodes  - simplified nodes

    So each stage is a Dijkstra that may not route THROUGH any node of the child graph.
    An alternative route would have to pass some junction, junctions survive
    simplification, and it is therefore excluded by construction.

    Two stages, undoing one merge each, in the reverse of the order they were applied:

        arena edge (u, v)  ->  chain through simplified_*  ->  chain through Map_fined/

    The result is still CHECKED against the arena edge's own length_m and only accepted
    within --tol. The constraint makes a wrong answer unlikely; the check makes it
    visible. Silently accepting one would paint a corridor's load onto the wrong streets,
    which is exactly what the heat map is read for.

    Anything that fails keeps its straight chord and is listed at the end.

INPUTS ARE READ-ONLY
    Map/ is the data pipeline's output and is not touched. The result lands here, in
    demo/, and the demo is its only consumer.
"""
import argparse
import heapq
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MAP = HERE.parent / "Map"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


# build_simplified_network.expand_directed's convention, reproduced exactly.
ONEWAY_FORWARD = {"yes", "true", "1"}
ONEWAY_REVERSE = {"reverse", "-1"}


def load_graph(edges_csv, expand=False):
    """{node: {succ: length}} on original OSM ids.

    🔴 The two layers do NOT use the same convention, and reading them as if they did is
    what broke the first two attempts at this script.

        Map_fined      one row per SEGMENT, direction carried in `oneway`
                       (measured: 0% of its (u,v) pairs have (v,u) present; 29,906 of
                       43,711 rows are `no`, i.e. two-way)
        simplified_*   the merge's OUTPUT, already one row per DIRECTION (84.3%)
        arena_*        likewise (52.5%; the rest are genuine one-ways)

    Reading Map_fined without expanding leaves every two-way road one-directional, so a
    chain breaks the moment it meets one -- and the more hops a chain has the likelier
    that is, which is why the failures were exactly the long edges the script exists for.

    `expand=True` reproduces build_simplified_network.expand_directed, `reverse` included.
    Parallel segments keep the shorter length, as that script also does.
    """
    df = pd.read_csv(edges_csv, encoding="utf-8-sig")
    g = {}

    def add(a, b, ln):
        if a == b:
            return
        succ = g.setdefault(a, {})
        g.setdefault(b, {})
        if ln < succ.get(b, float("inf")):
            succ[b] = ln

    ow = (df["oneway"].astype(str).str.strip().str.lower().to_numpy()
          if expand and "oneway" in df.columns else None)
    for i, (a, b, ln) in enumerate(zip(df.from_node.astype("int64"),
                                       df.to_node.astype("int64"),
                                       df.length_m.astype(float))):
        a, b, ln = int(a), int(b), float(ln)
        if ow is None:
            add(a, b, ln)
        elif ow[i] in ONEWAY_REVERSE:
            add(b, a, ln)
        elif ow[i] in ONEWAY_FORWARD:
            add(a, b, ln)
        else:
            add(a, b, ln)
            add(b, a, ln)
    return g


def restricted_path(g, a, b, blocked, cap):
    """Shortest a -> b whose INTERMEDIATE nodes all avoid `blocked`.

    `blocked` is the child graph's node set: every junction that survived the merge. The
    chain we are looking for has none of them inside it, and any alternative route has at
    least one, so this is a filter rather than a preference.
    """
    if a not in g or b not in g:
        return None
    dist, prev, pq, seen = {a: 0.0}, {}, [(0.0, a)], set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == b:
            break
        if len(seen) > cap:
            return None                    # runaway; caller falls back
        for w, ln in g[u].items():
            if w != b and w in blocked:
                continue                   # a junction: cannot be interior to a chain
            nd = d + ln
            if nd < dist.get(w, float("inf")):
                dist[w], prev[w] = nd, u
                heapq.heappush(pq, (nd, w))
    if b not in dist:
        return None
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1], dist[b]


def length_targeted(g, a, b, blocked, want, tol, cap):
    """A path a -> b of length within `tol` of `want`, interiors avoiding `blocked`.

    restricted_path returns the SHORTEST such path, which for a handful of arena edges is
    not the chain that was merged: build_arena.py simplified a SUBGRAPH of simplified_*,
    and a shortcut the subgraph did not contain is still present in the full graph. So the
    search is over the wrong graph, and minimising picks the shortcut.

    Enumerating instead of minimising fixes it without loosening the length check -- which
    would be the wrong trade, since a route that is 5% short is a route down different
    streets, and the heat map is read to see which streets are loaded. The search stays
    small because every interior node must be absent from the child graph, and because
    nothing longer than `want` is ever extended.
    """
    lo, hi = want * (1.0 - tol), want * (1.0 + tol)
    if a not in g or b not in g:
        return None
    stack, visits = [(a, 0.0, (a,))], 0
    while stack:
        u, acc, path = stack.pop()
        visits += 1
        if visits > cap:
            return None
        for w, ln in g[u].items():
            na = acc + ln
            if na > hi or w in path:
                continue
            if w == b:
                if na >= lo:
                    return list(path) + [w], na
            elif w not in blocked:
                stack.append((w, na, path + (w,)))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=0.02, metavar="FRAC",
                    help="accept a recovered shape when its length is within this "
                         "fraction of the edge's own length_m (default 2%%)")
    ap.add_argument("--precision", type=int, default=5,
                    help="decimal places for lat/lon; 5 is ~1 m and halves the file")
    ap.add_argument("--cap", type=int, default=40000, metavar="N",
                    help="node budget per Dijkstra")
    ap.add_argument("--dfs-cap", type=int, default=400000, metavar="N",
                    help="step budget for the length-targeted enumeration, which only "
                         "runs for edges the shortest chain got wrong")
    ap.add_argument("--loose", action="store_true",
                    help="for edges no chain of the right length was found for, draw the "
                         "best real-road path anyway instead of a straight chord. Every "
                         "use is listed with its length ratio")
    ap.add_argument("--out", default=str(HERE / "arena_geometry.json"))
    cli = ap.parse_args()

    print(f"\n{'=' * 82}\nrecovering arena edge shapes (drawing only)\n{'=' * 82}")
    arena = pd.read_csv(MAP / "arena_edges_taichung.csv", encoding="utf-8-sig")
    arena_nodes = set(pd.read_csv(MAP / "arena_nodes_taichung.csv",
                                  encoding="utf-8-sig").node_id.astype("int64"))
    simp_nodes = set(pd.read_csv(MAP / "simplified_nodes_taichung.csv",
                                 encoding="utf-8-sig").node_id.astype("int64"))
    nodes = pd.read_csv(MAP / "Map_fined" / "graph_nodes_taichung.csv",
                        encoding="utf-8-sig")
    lat = dict(zip(nodes.node_id.astype("int64"), nodes.latitude.astype(float)))
    lon = dict(zip(nodes.node_id.astype("int64"), nodes.longitude.astype(float)))

    simp = load_graph(MAP / "simplified_edges_taichung.csv")
    fine = load_graph(MAP / "Map_fined" / "graph_edges_taichung.csv", expand=True)
    print(f"  arena       {len(arena):>7,} edges / {len(arena_nodes):,} nodes")
    print(f"  simplified  {sum(len(v) for v in simp.values()):>7,} edges / "
          f"{len(simp):,} nodes")
    print(f"  Map_fined   {sum(len(v) for v in fine.values()):>7,} directed edges / "
          f"{len(fine):,} nodes  (expanded from 43,711 rows by `oneway`)")
    print(f"  interior of an arena edge avoids {len(arena_nodes):,} arena nodes; "
          f"of a simplified edge, {len(simp_nodes):,} simplified nodes")

    cache = {}
    out, stats = {}, {"ok": 0, "enumerated": 0, "unrestricted": 0, "loose": 0,
                      "chord": 0}
    residual, failed, loose = [], [], []

    def expand(chain):
        """One chain of simplified edges -> the fine-graph points and their total length."""
        pts, got = [chain[0]], 0.0
        for x, y in zip(chain[:-1], chain[1:]):
            key = (x, y)
            if key not in cache:
                cache[key] = restricted_path(fine, x, y, simp_nodes, cli.cap)
            sub = cache[key]
            if sub is None:
                return None, 0.0, "no chain in Map_fined"
            pts += sub[0][1:]
            got += sub[1]
        return pts, got, ""

    for r in arena.itertuples():
        a, b, want = int(r.from_node), int(r.to_node), float(r.length_m)
        eid, why = f"{a}_{b}", ""
        pts = got = None
        tol_m = cli.tol * max(want, 1.0)

        # stage 1: the chain of simplified edges this arena edge was merged from
        near = None                        # best-effort shape, kept for --loose
        s = restricted_path(simp, a, b, arena_nodes, cli.cap)
        if s is None:
            why = "no chain in simplified"
        else:
            pts, got, why = expand(s[0])                 # stage 2, through Map_fined
            if pts is not None and abs(got - want) > tol_m:
                near = (pts, got)
                pts, why = None, f"length {got:.1f} vs {want:.1f} ({got / want:.3f}x)"

        how = "ok" if pts is not None else None

        # The shortest restricted chain was the wrong one -- build_arena.py simplified a
        # SUBGRAPH, so a shortcut it never contained is still in the full graph. Enumerate
        # for one of the right length instead of loosening the check.
        if pts is None and why.startswith("length"):
            alt = length_targeted(simp, a, b, arena_nodes, want, cli.tol, cli.dfs_cap)
            if alt is not None:
                p2, g2, w2 = expand(alt[0])
                if p2 is not None and abs(g2 - want) <= tol_m:
                    pts, got, how = p2, g2, "enumerated"

        if pts is None:
            # Last resort: one unconstrained search across the fine graph. Only accepted
            # if the length still matches, so it cannot introduce a wrong road silently.
            d = restricted_path(fine, a, b, set(), cli.cap)
            if d is not None and abs(d[1] - want) <= tol_m:
                pts, got, how = d[0], d[1], "unrestricted"

        if pts is None and cli.loose and near is not None:
            # Off-length, but it is still a run of real roads between the right two
            # endpoints, and it is closer to the truth than the alternative: 十甲東路's
            # best chain is 0.944x its length_m where the straight chord is 0.79x. Only
            # reachable behind a flag, and every use is listed, because a path of the
            # wrong length is a path down at least partly the wrong streets -- which is
            # what the heat map is read for.
            pts, got, how = near[0], near[1], "loose"
            loose.append((want, getattr(r, "road_name", "") or "", got / want))

        if pts is None:
            stats["chord"] += 1
            failed.append((want, getattr(r, "road_name", "") or "", eid, why))
            continue
        stats[how] += 1

        if len(pts) > 2:                   # 2 points is the chord the demo already draws
            out[eid] = [[round(lat[p], cli.precision), round(lon[p], cli.precision)]
                        for p in pts if p in lat]
        residual.append(abs(got - want) / max(want, 1.0))

    with open(cli.out, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))

    n = len(arena)
    size = Path(cli.out).stat().st_size / 1024
    print(f"\nrecovered:")
    print(f"  chain through both layers   {stats['ok']:>5} ({stats['ok'] / n:5.1%})")
    print(f"  length-targeted enumeration {stats['enumerated']:>5} "
          f"({stats['enumerated'] / n:5.1%})")
    print(f"  unconstrained fallback      {stats['unrestricted']:>5} "
          f"({stats['unrestricted'] / n:5.1%})")
    if stats["loose"]:
        print(f"  best-effort (--loose)       {stats['loose']:>5} "
              f"({stats['loose'] / n:5.1%})")
    print(f"  kept the straight chord     {stats['chord']:>5} ({stats['chord'] / n:5.1%})")
    if residual:
        rs = sorted(residual)
        print(f"\n  length residual vs length_m: median {rs[len(rs) // 2]:.4%}, "
              f"worst accepted {rs[-1]:.4%}  (tolerance {cli.tol:.1%})")
    print(f"\nwrote {cli.out}")
    print(f"  {len(out):,} edges with a real shape, "
          f"{sum(len(v) for v in out.values()):,} points, {size:,.0f} KB")
    print(f"  the demo draws a chord for any edge absent from this file, so the "
          f"{n - len(out):,} others\n  are either straight already or fell back")

    if loose:
        print(f"\n{len(loose)} edges drawn BEST-EFFORT (--loose): real roads between the "
              f"right two endpoints,\nbut not a chain matching length_m, so part of each "
              f"shape may follow the wrong street.")
        for want, road, ratio in sorted(loose, reverse=True):
            print(f"  {want:>9.1f} m  {road or '(no road_name)':<14} {ratio:.3f}x")

    if failed:
        print(f"\n{len(failed)} edges kept the chord:")
        for want, road, eid, why in sorted(failed, reverse=True)[:15]:
            print(f"  {want:>9.1f} m  {road or '(no road_name)':<14} {why}")
        if len(failed) > 15:
            print(f"  ... and {len(failed) - 15} more")
        by_why = {}
        for _, _, _, why in failed:
            key = why.split(" (")[0] if why.startswith("length") else why
            by_why["length mismatch" if key.startswith("length") else key] = \
                by_why.get("length mismatch" if key.startswith("length") else key, 0) + 1
        print("  reasons: " + ", ".join(f"{k} x{v}" for k, v in sorted(by_why.items())))


if __name__ == "__main__":
    main()
