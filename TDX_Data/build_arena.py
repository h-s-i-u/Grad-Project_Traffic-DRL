# -*- coding: utf-8 -*-
"""
build_arena.py
──────────────
Carve the DRL routing arena out of the full Taichung OSM export.

Why (the reason is prediction coverage, not compute):
    The full export is 7,489 nodes / 20,347 edges, but only 524 of those edges (2.6%)
    carry a TDX prediction. Policies (2)(3)(4) route by Dijkstra on `tpred`, so
    wherever tpred == t0 their shortest path IS the free-flow shortest path -- i.e.
    policy (1). At 2.6% that is almost everywhere, and baselines (2)(3)(4) collapse
    onto (1). Baseline (4) is the denominator of every delta in the report, and
    "herding" means every vehicle chasing the SAME forecast, so a forecast that
    cannot tell roads apart leaves nothing to suppress.

    Vectorising _compute_enc_ctx would not move that ratio by a single point. Only
    shrinking the arena does.

This also brings the network back in line with the proposal. §4.2 asks for the
"主要道路網" of the 東海大學–台中車站 region at "約 100 個 intersection nodes"; the
current export is every alley and service road, 75x that node count. Filtering to the
road hierarchy is a correction toward the proposal, not a departure from it.

Composition (four sources, unioned then reduced to one strongly connected component):
    seed        edges covered by the TDX sections that survived build_speed.py's
                missing-rate filter -- the roads that RECEIVE diverted traffic, which
                is where herding and the second-wave congestion physically happen
    connect     shortest paths stitching the seed's disconnected fragments together.
                A TDX section is one arterial stretch, so the 524 seed edges alone
                have a largest SCC of 43 nodes; without this step most of them fall
                out of the final component and their sections are wasted
    corridor    the shortest path each way between the two demo endpoints. 台灣大道
                carries no TDX sensor (TDX instruments the roads feeding INTO it),
                but S3 must be able to CLOSE it, and closing a road needs the road in
                the graph, not a speed forecast for it
    backbone    every edge with at least --min-lanes lanes -- the parallel
                alternatives diverted traffic can actually take

Node ids stay the original OSM ids, so an arena route renders directly on the full
Map/graph_*_taichung.csv with no translation. The full export is never modified.

Outputs (identical schema to the inputs, so taichung_loader.py reads them unchanged):
    Map/arena_nodes_taichung.csv    node_id, latitude, longitude
    Map/arena_edges_taichung.csv    from_node, to_node, length_m,
                                    free_flow_speed_kmh, lanes, capacity
    Map/arena_meta.json             every number below, for the report

Usage:
    cd TDX_Data
    python build_arena.py                  # compares lanes>=1..4, writes lanes>=3
    python build_arena.py --min-lanes 2
    python build_arena.py --no-corridor    # TDX grid only; drops 台灣大道, so S3 has
                                           # no arterial to close and the demo route
                                           # leaves the arena
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import networkx as nx

import build_simplified_network as bsn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DIR = os.path.join(ROOT_DIR, "Map")

# Demo endpoints named in proposal §4.2. Pulling the corridor between them in also
# guarantees both are routable inside the arena.
DEMO_POINTS = {"東海大學": (24.1817, 120.6021), "台中車站": (24.1369, 120.6869)}


def osm_ids(df, col):
    """Read an OSM id column as int64.

    Never use .astype(int) on these: numpy's default int is int32 on Windows and
    84.7% of these ids exceed 2^31-1, so it wraps SILENTLY -- no warning, no error,
    just ids that match nothing. That bug wrote an arena node file holding 150 of
    its 440 rows before it was caught.
    """
    return df[col].to_numpy(dtype="int64")


def nearest_node(nodes, lat, lon):
    """Nearest node id by planar distance. Equirectangular is exact enough here."""
    dy = (nodes["latitude"].to_numpy() - lat) * 110_540.0
    dx = (nodes["longitude"].to_numpy() - lon) * 111_320.0 * np.cos(np.radians(lat))
    d = np.hypot(dx, dy)
    i = int(np.argmin(d))
    return int(osm_ids(nodes, "node_id")[i]), float(d[i])


def connect_fragments(g_full, seed, rounds=3):
    """Stitch the seed's disconnected fragments together with shortest paths.

    Multi-source Dijkstra from the fragment already attached, in both the forward and
    the reversed graph -- a one-way street would otherwise give a way in but no way
    back, and the arena has to be STRONGLY connected because run_compare samples OD
    pairs in both directions.

    Returns (edges added, fragment count, size of the largest fragment).
    """
    s = nx.DiGraph()
    s.add_edges_from(seed)
    comps = sorted(nx.weakly_connected_components(s), key=len, reverse=True)
    if len(comps) <= 1:
        return set(), len(comps), (len(comps[0]) if comps else 0)

    hub, added = set(comps[0]), set()
    rg = g_full.reverse(copy=False)
    for _ in range(rounds):
        rest = [c for c in comps if not (c & hub)]
        if not rest:
            break
        for forward, gg in ((True, g_full), (False, rg)):
            dist, path = nx.multi_source_dijkstra(gg, hub, weight="length")
            for c in rest:
                reach = [n for n in c if n in dist]
                if not reach:
                    continue
                p = path[min(reach, key=lambda n: dist[n])]
                pe = list(zip(p[:-1], p[1:]))
                added |= set(pe) if forward else {(b, a) for a, b in pe}
        for c in rest:
            hub |= c
    return added & set(g_full.edges()), len(comps), len(comps[0])


def build_arena(g_full, edge_sets):
    """Union the sources, then keep the largest strongly connected component."""
    keep = set().union(*edge_sets) & set(g_full.edges())
    h = nx.DiGraph()
    h.add_edges_from(keep)
    if h.number_of_nodes() < 2:
        return None
    scc = max(nx.strongly_connected_components(h), key=len)
    return g_full.subgraph(scc).copy()


def resimplify(g, seed, edges, row_of, protect=()):
    """Merge the pass-through chains that carving the arena creates.

    build_simplified_network.py already removed the full network's pass-through nodes,
    but taking a subgraph cuts edges and turns former junctions back into one-in
    one-out nodes -- 684 of 1,442 here (47.4%). Each costs the routing policy a
    decision step at which it has no decision to make: in a rollout 54.5% of the
    transitions had exactly one candidate, contributing zero policy gradient while
    stretching the credit-assignment chain to 46 hops.

    Two things are protected from merging, both because something outside this file
    addresses them BY NODE OR EDGE ID:
      * chains containing a SEED edge -- section_to_edges.csv attaches predictions by
        exactly the (u, v) key, so merging a covered edge would silently detach its
        TDX signal, and a half-covered merged edge is not something make_drl_input
        can express anyway;
      * the nodes in `protect` -- the demo endpoints. 台中車站 happened to be a
        pass-through node in the arena and was merged away on the first run, leaving
        the demo with no destination.
    Measured cost: 81 of the 684 chains stay, 603 still merge.

    Returns (merged_graph, n_merged). Attributes are aggregated with the same rules as
    build_simplified_network.aggregate, so free-flow travel time is preserved exactly.
    """
    protect = set(protect)

    def passthrough(n):
        pred, succ = set(g.predecessors(n)), set(g.successors(n))
        if n in protect:
            return False                       # named node, must stay addressable
        if any(e in seed for e in list(g.in_edges(n)) + list(g.out_edges(n))):
            return False                       # keep every seed edge addressable
        if g.in_degree(n) == 1 and g.out_degree(n) == 1 and pred != succ:
            return True
        return (g.in_degree(n) == 2 and g.out_degree(n) == 2
                and pred == succ and len(pred) == 2)

    through = {n for n in g.nodes() if passthrough(n)}
    if not through:
        return g, 0
    keep = set(g.nodes()) - through

    merged, walked = {}, set()
    for a in keep:
        for b in list(g.successors(a)):
            rows, prev, cur, hops = [row_of[(a, b)]], a, b, 0
            while cur in through and hops < 10_000:
                walked.add(cur)
                nxt = bsn.step(g, cur, prev)
                if nxt is None:
                    break
                rows.append(row_of[(cur, nxt)])
                prev, cur, hops = cur, nxt, hops + 1
            if cur == a:
                continue                       # chain looped back on itself
            attrs = bsn.aggregate(rows, edges)
            if (a, cur) in merged and attrs["length_m"] >= merged[(a, cur)]["length_m"]:
                continue                       # parallel chain: keep the shorter
            merged[(a, cur)] = attrs

    # Same trap as in build_simplified_network: a ring made only of pass-through nodes
    # is never reached from a junction and would vanish silently.
    for n in sorted(through - walked):
        for m in g.successors(n):
            merged.setdefault((n, m), bsn.aggregate([row_of[(n, m)]], edges))
        keep.add(n)

    h = nx.DiGraph()
    h.add_nodes_from(keep)
    for (u, v), a in merged.items():
        h.add_edge(u, v, **{k: val for k, val in a.items() if not k.startswith("_")})
    return h, len(through)


def summarize(g, seed, length_of):
    edges = set(g.edges())
    covered = seed & edges
    total_len = sum(length_of[e] for e in edges)
    cov_len = sum(length_of[e] for e in covered)
    return {
        "nodes": g.number_of_nodes(), "edges": len(edges),
        "covered_edges": len(covered),
        "coverage": len(covered) / len(edges) if edges else 0.0,
        "coverage_by_length": cov_len / total_len if total_len else 0.0,
        # Hop count overstates the decision load: at a degree-2 node the policy has
        # no choice to make. This is how many real branch points exist.
        "branch_nodes": sum(1 for n in g.nodes() if g.out_degree(n) >= 2),
    }


def sample_hops(g, n_pairs, rng):
    nodes = list(g.nodes())
    if len(nodes) < 2:
        return float("nan"), 0
    hops = []
    for _ in range(n_pairs):
        a, b = rng.choice(len(nodes), size=2, replace=False)
        try:
            hops.append(nx.shortest_path_length(g, nodes[a], nodes[b]))
        except nx.NetworkXNoPath:
            pass
    return (float(np.mean(hops)), int(np.max(hops))) if hops else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map-dir", default=MAP_DIR)
    # The road network must be the SAME one build_network.py mapped the TDX sections
    # onto, otherwise the seed edges refer to node pairs that do not exist here and the
    # arena is built from nothing. Defaults track build_network.py's own defaults.
    ap.add_argument("--nodes-csv",
                    default=os.path.join(MAP_DIR, "simplified_nodes_taichung.csv"),
                    help="node CSV (default: the simplified network)")
    ap.add_argument("--edges-csv",
                    default=os.path.join(MAP_DIR, "simplified_edges_taichung.csv"),
                    help="edge CSV, one row per directed edge "
                         "(default: the simplified network)")
    ap.add_argument("--min-lanes", type=int, default=3,
                    help="backbone threshold for the arena that gets written")
    ap.add_argument("--compare", default="1,2,3,4",
                    help="lane thresholds to tabulate before writing")
    ap.add_argument("--no-corridor", action="store_true")
    ap.add_argument("--no-resimplify", action="store_true",
                    help="skip merging the pass-through nodes the subgraph cut creates; keeps every original edge id")
    ap.add_argument("--no-connect", action="store_true",
                    help="skip fragment stitching (most TDX sections then drop out)")
    ap.add_argument("--connect-rounds", type=int, default=3)
    ap.add_argument("--od-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="arena")
    cli = ap.parse_args()

    md = cli.map_dir
    src = {"nodes": cli.nodes_csv, "edges": cli.edges_csv,
           "s2e": os.path.join(md, "section_to_edges.csv"),
           "index": os.path.join(md, "taichung_section_index.csv")}
    for name, p in src.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{name} not found: {p}\n"
                                    f"Run build_network.py and build_speed.py first.")

    nodes = pd.read_csv(src["nodes"])
    edges = pd.read_csv(src["edges"])
    s2e = pd.read_csv(src["s2e"], encoding="utf-8-sig")
    index = pd.read_csv(src["index"], encoding="utf-8-sig")

    # --- full graph ---
    g_full = nx.DiGraph()
    for nid, lat, lon in zip(osm_ids(nodes, "node_id"),
                            nodes["latitude"], nodes["longitude"]):
        g_full.add_node(int(nid), latitude=float(lat), longitude=float(lon))
    keys = [(int(a), int(b)) for a, b in
            zip(osm_ids(edges, "from_node"), osm_ids(edges, "to_node"))]
    length_of, row_of = {}, {}
    for i, ((u, v), L) in enumerate(zip(keys, edges["length_m"])):
        if u in g_full and v in g_full:
            g_full.add_edge(u, v, length=float(L))
            length_of[(u, v)] = float(L)
            row_of[(u, v)] = i          # lets resimplify() reuse bsn.aggregate()
    full_edges = set(g_full.edges())

    # --- seed ---
    # Membership uses an explicit comprehension, NOT pandas .isin(): on an object
    # column of tuples isin() coerces to a 2-D array and returns wrong matches (it
    # reported 63 of these 524 edges during development).
    kept_sections = set(index["SectionID"])
    s2e_kept = s2e[s2e["SectionID"].isin(kept_sections)]
    mapped = {(int(a), int(b)) for a, b in
              zip(osm_ids(s2e_kept, "from_node"), osm_ids(s2e_kept, "to_node"))}
    seed = mapped & full_edges
    # A low hit rate means the mapping was produced against a DIFFERENT network -- the
    # single easiest way to get a silently empty arena, since node ids are OSM ids and
    # so a stale mapping still "looks" plausible.
    hit = len(seed) / len(mapped) if mapped else 0.0
    if hit < 0.5:
        raise SystemExit(
            f"ERROR  only {len(seed)}/{len(mapped)} ({hit:.0%}) of the mapped edges "
            f"exist in this network.\n"
            f"  section_to_edges.csv was built against a different graph. Re-run "
            f"build_network.py\n  pointing at the same files as --nodes-csv / "
            f"--edges-csv:\n    {cli.nodes_csv}\n    {cli.edges_csv}")

    print("=== inputs ===")
    print(f"  nodes  {cli.nodes_csv}")
    print(f"  edges  {cli.edges_csv}")
    print(f"  full network   {g_full.number_of_nodes():,} nodes / {len(full_edges):,} edges")
    print(f"  seed (TDX)     {len(seed)} edges from {len(kept_sections)} sections "
          f"= {len(seed) / len(full_edges):.1%} of the full network")

    link = set()
    if not cli.no_connect:
        link, n_frag, biggest = connect_fragments(g_full, seed, cli.connect_rounds)
        print(f"  connect        {len(link)} edges stitching {n_frag} seed fragments "
              f"(largest fragment {biggest} nodes)")

    corridor, endpoints = set(), {}
    if not cli.no_corridor:
        for name, (la, lo) in DEMO_POINTS.items():
            nid, dist = nearest_node(nodes, la, lo)
            endpoints[name] = {"node": nid, "snap_distance_m": dist}
        a, b = (v["node"] for v in endpoints.values())
        for s, t in ((a, b), (b, a)):
            try:
                p = nx.shortest_path(g_full, s, t, weight="length")
                corridor |= set(zip(p[:-1], p[1:]))
            except nx.NetworkXNoPath:
                print(f"  WARNING  no path {s} -> {t}; corridor incomplete")
        print(f"  corridor       {len(corridor)} edges "
              f"({sum(length_of[e] for e in corridor) / 1000:.1f} km, both ways); "
              f"TDX covers {len(seed & corridor)} of them")

    # --- compare lane thresholds ---
    lanes = edges["lanes"].fillna(0).to_numpy()
    print(f"\n=== backbone threshold ===")
    print(f"  {'lanes>=':>7}{'nodes':>8}{'edges':>8}{'coverage':>10}{'by len':>9}"
          f"{'seed kept':>12}{'branch':>8}{'sections':>10}")
    print("  " + "-" * 74)
    table = {}
    for L in sorted({int(x) for x in cli.compare.split(",")} | {cli.min_lanes}):
        backbone = {k for k, n in zip(keys, lanes) if n >= L} & full_edges
        g = build_arena(g_full, [seed, link, corridor, backbone])
        if g is None:
            continue
        s = summarize(g, seed, length_of)
        kept_e = set(g.edges())
        s["sections_in_arena"] = int(s2e_kept[
            [(int(a), int(b)) in kept_e for a, b in
             zip(osm_ids(s2e_kept, "from_node"), osm_ids(s2e_kept, "to_node"))]
        ]["SectionID"].nunique())
        table[L] = (s, g)
        print(f"  {L:>7}{s['nodes']:>8,}{s['edges']:>8,}{s['coverage']:>9.1%}"
              f"{s['coverage_by_length']:>9.1%}"
              f"{s['covered_edges']:>7,}/{len(seed):<4}{s['branch_nodes']:>8,}"
              f"{s['sections_in_arena']:>6}/{len(kept_sections):<4}"
              f"{'  <- writing' if L == cli.min_lanes else ''}")

    if cli.min_lanes not in table:
        raise SystemExit(f"ERROR  lanes>={cli.min_lanes} gave a degenerate arena")
    stats, g = table[cli.min_lanes]

    # --- merge the chains that carving the subgraph created ---
    n_merged = 0
    if not cli.no_resimplify:
        before_n, before_e = g.number_of_nodes(), g.number_of_edges()
        rng0 = np.random.default_rng(cli.seed)
        hops_before, _ = sample_hops(g, cli.od_samples, rng0)
        g, n_merged = resimplify(g, seed, edges, row_of,
                                 protect={v["node"] for v in endpoints.values()})
        # (u, v) as a tuple: `for *e, d in ...` binds e to a LIST, which cannot be a
        # dict key.
        length_of = {(u, v): float(d["length_m"]) for u, v, d in g.edges(data=True)}
        stats = summarize(g, seed, length_of)
        kept_e0 = set(g.edges())
        stats["sections_in_arena"] = int(s2e_kept[
            [(int(a), int(b)) in kept_e0 for a, b in
             zip(osm_ids(s2e_kept, "from_node"), osm_ids(s2e_kept, "to_node"))]
        ]["SectionID"].nunique())
        rng0 = np.random.default_rng(cli.seed)
        hops_after, _ = sample_hops(g, cli.od_samples, rng0)
        print("")
        print("=== re-simplify ===")
        print(f"  merged {n_merged:,} pass-through nodes left by the subgraph cut "
              f"(chains touching a seed edge are kept so predictions stay attached)")
        print(f"  {before_n:,} -> {g.number_of_nodes():,} nodes, "
              f"{before_e:,} -> {g.number_of_edges():,} edges")
        print(f"  mean path {hops_before:.1f} -> {hops_after:.1f} hops "
              f"({(hops_after - hops_before) / hops_before:+.0%}) -- every remaining "
              f"node is a real junction")

    # --- the chosen arena ---
    rng = np.random.default_rng(cli.seed)
    mean_hops, max_hops = sample_hops(g, cli.od_samples, rng)
    kept_e = set(g.edges())
    alive = s2e_kept[[(int(a), int(b)) in kept_e for a, b in
                      zip(osm_ids(s2e_kept, "from_node"), osm_ids(s2e_kept, "to_node"))]]
    roads = (index[index["SectionID"].isin(set(alive["SectionID"]))]["RoadName"]
             .astype(str).str.replace(r"[一二三四五六七八九十]?段.*$", "", regex=True)
             .str.strip().value_counts())

    print(f"\n=== arena (lanes>={cli.min_lanes}) ===")
    print(f"  {stats['nodes']:,} nodes / {stats['edges']:,} edges "
          f"(from {g_full.number_of_nodes():,} / {len(full_edges):,}); "
          f"proposal §4.2 asked for ~100 nodes")
    print(f"  prediction coverage {stats['coverage']:.1%} by edge, "
          f"{stats['coverage_by_length']:.1%} by length "
          f"(full network {len(seed) / len(full_edges):.1%})")
    print(f"  TDX sections represented {stats['sections_in_arena']} of {len(kept_sections)}")
    print(f"  path {mean_hops:.1f} hops mean / {max_hops} max, but only "
          f"{stats['branch_nodes']:,} of {stats['nodes']:,} nodes are branch points")
    print(f"  roads with prediction: "
          f"{', '.join(f'{r}({n})' for r, n in roads.head(8).items())}")

    if endpoints:
        print(f"\n=== demo endpoints (proposal §4.2) ===")
        for name, info in endpoints.items():
            info["in_arena"] = info["node"] in g
            print(f"  {name}  node {info['node']}  snapped {info['snap_distance_m']:.0f} m"
                  f"  {'IN the arena' if info['in_arena'] else 'NOT in the arena'}")
        if not all(v["in_arena"] for v in endpoints.values()):
            print("  -> an endpoint is missing. If re-simplify ran it should have "
                  "been protected; otherwise it fell out with the SCC -- try a "
                  "lower --min-lanes.")

    # --- write, preserving the input schema ---
    # Built from the GRAPH, not by filtering the source rows: after re-simplify a
    # merged edge is a synthesised chain with no row of its own in `edges`.
    keep_nodes = set(g.nodes())
    out_nodes = nodes[[int(n) in keep_nodes for n in osm_ids(nodes, "node_id")]].copy()
    cols = ["length_m", "free_flow_speed_kmh", "lanes", "capacity", "road_name",
            "district", "oneway", "lanes_imputed", "speed_imputed",
            "road_name_missing", "district_matched_by"]
    rows_out = []
    for u, v, d in g.edges(data=True):
        if "length_m" in d:                       # merged edge: attributes on the graph
            rows_out.append({"from_node": u, "to_node": v,
                             **{c: d.get(c) for c in cols}})
        else:                                     # untouched edge: copy its source row
            rows_out.append({"from_node": u, "to_node": v,
                             **{c: edges[c].iat[row_of[(u, v)]] for c in cols}})
    out_edges = pd.DataFrame(rows_out).sort_values(["from_node", "to_node"])
    if len(out_nodes) != stats["nodes"] or len(out_edges) != stats["edges"]:
        raise SystemExit(f"ERROR  wrote {len(out_nodes)} nodes / {len(out_edges)} edges "
                         f"but the arena has {stats['nodes']} / {stats['edges']} -- "
                         f"an id lookup lost rows")
    p_nodes = os.path.join(md, f"{cli.prefix}_nodes_taichung.csv")
    p_edges = os.path.join(md, f"{cli.prefix}_edges_taichung.csv")
    out_nodes.to_csv(p_nodes, index=False, encoding="utf-8-sig")
    out_edges.to_csv(p_edges, index=False, encoding="utf-8-sig")

    meta = {
        "min_lanes": cli.min_lanes, "corridor_included": bool(corridor),
        "connect_included": bool(link), "connect_edges": len(link),
        "corridor_edges": len(corridor),
        "full_nodes": g_full.number_of_nodes(), "full_edges": len(full_edges),
        "arena_nodes": stats["nodes"], "arena_edges": stats["edges"],
        "branch_nodes": stats["branch_nodes"],
        "seed_edges_total": len(seed), "seed_edges_in_arena": stats["covered_edges"],
        "coverage_by_edge": stats["coverage"],
        "coverage_by_length": stats["coverage_by_length"],
        "coverage_full_network": len(seed) / len(full_edges),
        "sections_kept": len(kept_sections),
        "sections_in_arena": stats["sections_in_arena"],
        "mean_hops": mean_hops, "max_hops": max_hops,
        "demo_endpoints": endpoints,
        "comparison": {str(L): {k: v for k, v in s.items()} for L, (s, _) in table.items()},
        "note": ("Node ids are the original OSM ids, so arena routes render on the "
                 "full Map/graph_*_taichung.csv unchanged. 台灣大道 is in the arena "
                 "via the corridor but carries no TDX prediction -- S3 closes it, "
                 "which needs the road, not a forecast for it."),
    }
    p_meta = os.path.join(md, f"{cli.prefix}_meta.json")
    with open(p_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nOK  {p_nodes}  [{len(out_nodes)}]")
    print(f"OK  {p_edges}  [{len(out_edges)}]")
    print(f"OK  {p_meta}")
    print(f"\nNext:")
    print(f"  1. integration/taichung_loader.py -- point the router at the arena:")
    print(f"       DEFAULT_EDGES = _ROOT / 'Map' / '{cli.prefix}_edges_taichung.csv'")
    print(f"       DEFAULT_NODES = _ROOT / 'Map' / '{cli.prefix}_nodes_taichung.csv'")
    print(f"  2. cd integration && python calibrate_taichung.py")
    print(f"     capacity_scale was tuned on the full graph; it does NOT carry over")
    print(f"  3. make_drl_input.py needs no change -- it writes all {len(seed)} covered")
    print(f"     edges and the loader ignores those outside the arena")


if __name__ == "__main__":
    main()
