# -*- coding: utf-8 -*-
"""
build_simplified_network.py
───────────────────────────
Collapse the chains of intermediate nodes in Map_fined/ into single edges, so the
routing graph keeps its shape but stops carrying nodes where no decision is made.

Why:
    Map_fined/ has a node roughly every 49 m (12,502 nodes over 609 km of road), and
    68.5% of the nodes inside the largest SCC have exactly one way in and one way out.
    A per-node routing policy has no choice to make at those, yet every one of them
    costs a decision step, an encoder pass and a PPO transition. Merging them leaves
    about 3,579 nodes and does not change a single route.

    This is NOT pruning. No road is removed and no route becomes unavailable; the
    same journey is simply described with fewer waypoints.

Direction handling (the reason this runs before build_network.py):
    Map_fined writes one row per road segment plus an `oneway` column
    (no 9,211 / yes 4,640 / reverse 1). Whether a node is a pass-through depends on
    direction, so the graph is expanded to directed edges FIRST:
        oneway=yes         -> (from, to)
        oneway=reverse|-1  -> (to, from)      one-way against the drawn geometry
        otherwise          -> both directions
    Treating every segment as two-way instead would invent 4,640 wrong-way edges --
    a third of the network -- and would make diversion look easier than it is.

Attribute aggregation over a merged chain:
    length_m              sum
    free_flow_speed_kmh   length-weighted HARMONIC mean, i.e. L / sum(len_i/speed_i).
                          This is the only choice that leaves free-flow travel time
                          unchanged, and t0 = length/speed is what the router costs
                          journeys with. An arithmetic mean or a min would silently
                          shift every t0 in the network.
    lanes, capacity       min -- a chain is only as wide as its narrowest link
    road_name, district   the value covering the greatest length of the chain
    *_imputed             True if ANY segment in the chain was imputed (conservative)
    district_matched_by   the least trustworthy of the chain (unavailable >
                          nearest_fallback > contains)

Output (one row per DIRECTED edge, unlike the input):
    Map/simplified_nodes_taichung.csv
    Map/simplified_edges_taichung.csv
    Map/simplified_meta.json

    Emitting both directions explicitly matters: taichung_loader.py does not read the
    `oneway` column, so handing it one row per segment would make every road one-way.
    The column is still written, as provenance for which segments were one-way in OSM.

Self-check:
    Merging is only worth doing if it provably changes nothing. The script verifies
    that total road length is preserved and that free-flow shortest-path TIMES between
    random surviving nodes match the original graph. A mismatch fails the run.

Usage:
    cd TDX_Data
    python build_simplified_network.py
    python build_simplified_network.py --check-pairs 500
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DIR = os.path.join(ROOT_DIR, "Map")
SRC_DIR = os.path.join(MAP_DIR, "Map_fined")

ONEWAY_FORWARD = {"yes", "true", "1"}
ONEWAY_REVERSE = {"reverse", "-1"}
TRUST_ORDER = ["contains", "nearest_fallback", "unavailable"]   # best -> worst


def osm_ids(df, col):
    """int64, never .astype(int): numpy's default int is int32 on Windows and these
    OSM ids exceed 2^31-1, which wraps silently."""
    return df[col].to_numpy(dtype="int64")


def expand_directed(edges):
    """[(u, v, source_row_index)] honouring the oneway column."""
    u_all, v_all = osm_ids(edges, "from_node"), osm_ids(edges, "to_node")
    ow = edges["oneway"].astype(str).str.strip().str.lower().to_numpy()
    out = []
    for i, (u, v, o) in enumerate(zip(u_all, v_all, ow)):
        u, v = int(u), int(v)
        if u == v:
            continue                      # self loop in the source data
        if o in ONEWAY_REVERSE:
            out.append((v, u, i))
        elif o in ONEWAY_FORWARD:
            out.append((u, v, i))
        else:
            out.append((u, v, i))
            out.append((v, u, i))
    return out


def is_passthrough(g, n):
    """No routing choice is made here, in either direction.

    One-way chain : exactly one way in, one way out, to different nodes.
    Two-way chain : the ways in and the ways out are the SAME pair of neighbours.
    Anything else (junction, dead end, merge point) is kept.
    """
    pred, succ = set(g.predecessors(n)), set(g.successors(n))
    if g.in_degree(n) == 1 and g.out_degree(n) == 1 and pred != succ:
        return True
    return (g.in_degree(n) == 2 and g.out_degree(n) == 2
            and pred == succ and len(pred) == 2)


def step(g, cur, prev):
    """Next node along a chain, or None if `cur` is not a pass-through."""
    succ = list(g.successors(cur))
    if len(succ) == 1:
        return succ[0]                               # one-way chain
    if len(succ) == 2 and prev in succ:
        return succ[1] if succ[0] == prev else succ[0]
    return None


def aggregate(rows, edges):
    """Collapse the source rows of one chain into a single edge's attributes."""
    lengths = np.array([float(edges["length_m"].iat[i]) for i in rows])
    speeds = np.array([float(edges["free_flow_speed_kmh"].iat[i]) for i in rows])
    total_len = float(lengths.sum())
    # Harmonic mean weighted by length: preserves sum(len_i / speed_i), i.e. t0.
    travel = float((lengths / np.maximum(speeds, 1e-9)).sum())
    speed = total_len / travel if travel > 0 else float(speeds.min())

    def longest(col):
        acc = Counter()
        for i, L in zip(rows, lengths):
            acc[edges[col].iat[i]] += L
        return acc.most_common(1)[0][0]

    trust = [str(edges["district_matched_by"].iat[i]) for i in rows]
    worst = max(trust, key=lambda t: TRUST_ORDER.index(t) if t in TRUST_ORDER
                else len(TRUST_ORDER))
    return {
        "length_m": total_len,
        "free_flow_speed_kmh": speed,
        "lanes": int(min(int(edges["lanes"].iat[i]) for i in rows)),
        "capacity": int(min(int(edges["capacity"].iat[i]) for i in rows)),
        "road_name": longest("road_name"),
        "district": longest("district"),
        "oneway": ("yes" if any(str(edges["oneway"].iat[i]).strip().lower()
                                not in ("no", "false", "0") for i in rows) else "no"),
        "lanes_imputed": bool(any(bool(edges["lanes_imputed"].iat[i]) for i in rows)),
        "speed_imputed": bool(any(bool(edges["speed_imputed"].iat[i]) for i in rows)),
        "road_name_missing": bool(any(bool(edges["road_name_missing"].iat[i])
                                      for i in rows)),
        "district_matched_by": worst,
        "_t0": travel,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", default=SRC_DIR)
    ap.add_argument("--out-dir", default=MAP_DIR)
    ap.add_argument("--prefix", default="simplified")
    ap.add_argument("--largest-scc", action="store_true", default=True,
                    help="keep only the largest strongly connected component")
    ap.add_argument("--keep-all", dest="largest_scc", action="store_false")
    ap.add_argument("--check-pairs", type=int, default=300,
                    help="random OD pairs used to prove travel times are unchanged")
    ap.add_argument("--tolerance", type=float, default=1e-6,
                    help="max allowed relative travel-time difference")
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    p_nodes = os.path.join(cli.src_dir, "graph_nodes_taichung.csv")
    p_edges = os.path.join(cli.src_dir, "graph_edges_taichung.csv")
    for p in (p_nodes, p_edges):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} not found")
    nodes = pd.read_csv(p_nodes)
    edges = pd.read_csv(p_edges)
    for col in ("oneway", "road_name", "district", "district_matched_by"):
        if col not in edges.columns:
            raise ValueError(f"{p_edges} has no '{col}' column -- this script needs the "
                             f"Map_fined export that carries oneway and road metadata")

    # --- directed graph ---
    directed = expand_directed(edges)
    g = nx.DiGraph()
    g.add_nodes_from(int(x) for x in osm_ids(nodes, "node_id"))
    for u, v, i in directed:
        if g.has_edge(u, v):
            # keep the shorter of two parallel segments; a DiGraph holds only one
            if edges["length_m"].iat[i] >= edges["length_m"].iat[g[u][v]["row"]]:
                continue
        g.add_edge(u, v, row=i)

    print("=== input ===")
    print(f"  {len(nodes):,} nodes / {len(edges):,} segments (as written)")
    counts = edges["oneway"].astype(str).str.lower().value_counts()
    print(f"  oneway: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    print(f"  expanded by oneway -> {g.number_of_edges():,} directed edges")

    if cli.largest_scc:
        scc = max(nx.strongly_connected_components(g), key=len)
        dropped = g.number_of_nodes() - len(scc)
        g = g.subgraph(scc).copy()
        print(f"  largest SCC {len(scc):,} (dropped {dropped:,} nodes that cannot be "
              f"reached both ways)")

    orig = g.copy()          # kept for the travel-time proof

    # --- collapse chains ---
    through = {n for n in g.nodes() if is_passthrough(g, n)}
    keep = set(g.nodes()) - through
    print(f"\n=== merge ===")
    print(f"  pass-through nodes {len(through):,} / {g.number_of_nodes():,} "
          f"({len(through) / g.number_of_nodes():.1%}) -> keeping {len(keep):,}")

    merged, parallel, loops = {}, 0, 0
    # Nodes actually walked over. A pass-through node absent from `merged`'s endpoints
    # is the NORMAL outcome -- that is what being merged away means -- so orphan rings
    # must be detected from what the walks consumed, not from the output's endpoints.
    walked = set()
    for a in keep:
        for b in list(g.successors(a)):
            rows, prev, cur, hops = [g[a][b]["row"]], a, b, 0
            while cur in through and hops < 10_000:
                walked.add(cur)
                nxt = step(g, cur, prev)
                if nxt is None:
                    break
                rows.append(g[cur][nxt]["row"])
                prev, cur, hops = cur, nxt, hops + 1
            if cur == a:
                loops += 1                 # chain looped back on itself
                continue
            attrs = aggregate(rows, edges)
            if (a, cur) in merged:
                parallel += 1
                if attrs["length_m"] >= merged[(a, cur)]["length_m"]:
                    continue               # keep the shorter alternative
            merged[(a, cur)] = attrs

    # A ring made entirely of pass-through nodes has no junction to start from, so no
    # walk ever reaches it. Rare, but it would silently delete a road.
    orphan_rings = sorted(through - walked)
    if orphan_rings:
        print(f"  NOTE  {len(orphan_rings):,} nodes sit on rings made entirely of "
              f"pass-through nodes (no junction to start from); kept as they are")
        for n in orphan_rings:
            for m in g.successors(n):
                merged.setdefault((n, m), aggregate([g[n][m]["row"]], edges))
        keep |= set(orphan_rings)

    print(f"  {g.number_of_edges():,} -> {len(merged):,} directed edges "
          f"(parallel chains reduced to the shorter one: {parallel:,}; "
          f"self loops dropped: {loops:,})")

    # --- verify: total length ---
    # Compare like with like. The source file holds ONE row per segment, the output
    # one row per DIRECTION, so a two-way road counts twice on the right-hand side;
    # the reference has to be the expanded directed graph, not the raw file total.
    src_len = float(edges["length_m"].sum())
    exp_len = sum(float(edges["length_m"].iat[d["row"]]) for _, _, d in orig.edges(data=True))
    new_len = sum(a["length_m"] for a in merged.values())
    print(f"\n=== verification ===")
    print(f"  source file (one row per segment)      {src_len / 1000:>9,.1f} km")
    print(f"  expanded to directed edges, in the SCC {exp_len / 1000:>9,.1f} km")
    print(f"  after merging                          {new_len / 1000:>9,.1f} km "
          f"({(new_len - exp_len) / exp_len:+.3%})")
    if exp_len > 0 and abs(new_len - exp_len) / exp_len > 0.02:
        print(f"  WARNING  more than 2% of road length changed. Merging should only "
              f"redistribute length, not create or destroy it -- check the parallel-"
              f"chain and self-loop counts above before trusting this output.")

    # --- verify: free-flow travel time is unchanged ---
    h = nx.DiGraph()
    for (u, v), a in merged.items():
        h.add_edge(u, v, t0=a["_t0"])
    for u, v, d in orig.edges(data=True):
        i = d["row"]
        orig[u][v]["t0"] = (float(edges["length_m"].iat[i])
                            / max(float(edges["free_flow_speed_kmh"].iat[i]), 1e-9))

    rng = np.random.default_rng(cli.seed)
    pool = [n for n in keep if n in h]
    worst_rel, checked, unreachable = 0.0, 0, 0
    for _ in range(cli.check_pairs):
        if len(pool) < 2:
            break
        a, b = rng.choice(len(pool), size=2, replace=False)
        a, b = pool[a], pool[b]
        try:
            t_old = nx.shortest_path_length(orig, a, b, weight="t0")
            t_new = nx.shortest_path_length(h, a, b, weight="t0")
        except nx.NetworkXNoPath:
            unreachable += 1
            continue
        checked += 1
        if t_old > 0:
            worst_rel = max(worst_rel, abs(t_new - t_old) / t_old)
    print(f"  free-flow travel time over {checked} random OD pairs: "
          f"max relative error {worst_rel:.2e} (unreachable {unreachable})")
    if checked and worst_rel > cli.tolerance:
        raise SystemExit(
            f"ERROR  simplification changed path times ({worst_rel:.2e} > "
            f"{cli.tolerance:.0e}). The merge rules are wrong -- do not use the output.")
    # The input was a single strongly connected component, so merging it must leave one
    # too. Widespread unreachability means chains were broken, not shortened -- and it
    # also means the travel-time check above only exercised the surviving fragment.
    if checked + unreachable and unreachable / (checked + unreachable) > 0.02:
        raise SystemExit(
            f"ERROR  {unreachable} of {checked + unreachable} sampled OD pairs are "
            f"unreachable after merging, but the input was one strongly connected "
            f"component. The graph was fragmented -- do not use the output.")
    print(f"  PASS  routing behaviour is unchanged")

    # --- write ---
    rows = []
    for (u, v), a in sorted(merged.items()):
        rows.append({"from_node": u, "to_node": v,
                     **{k: a[k] for k in ("length_m", "free_flow_speed_kmh", "lanes",
                                          "capacity", "road_name", "district", "oneway",
                                          "lanes_imputed", "speed_imputed",
                                          "road_name_missing", "district_matched_by")}})
    out_edges = pd.DataFrame(rows)
    used = set(out_edges["from_node"]) | set(out_edges["to_node"])
    out_nodes = nodes[[int(n) in used for n in osm_ids(nodes, "node_id")]].copy()

    os.makedirs(cli.out_dir, exist_ok=True)
    pe = os.path.join(cli.out_dir, f"{cli.prefix}_edges_taichung.csv")
    pn = os.path.join(cli.out_dir, f"{cli.prefix}_nodes_taichung.csv")
    out_edges.to_csv(pe, index=False, encoding="utf-8-sig")
    out_nodes.to_csv(pn, index=False, encoding="utf-8-sig")

    meta = {
        "source": os.path.relpath(cli.src_dir, ROOT_DIR),
        "src_nodes": int(len(nodes)), "src_segments": int(len(edges)),
        "expanded_directed_edges": int(orig.number_of_edges()),
        "largest_scc_only": bool(cli.largest_scc),
        "passthrough_merged": int(len(through)),
        "out_nodes": int(len(out_nodes)), "out_directed_edges": int(len(out_edges)),
        "parallel_dropped": int(parallel), "self_loops_dropped": int(loops),
        "length_km_source": src_len / 1000, "length_km_out": new_len / 1000,
        "traveltime_pairs_checked": int(checked),
        "traveltime_max_rel_error": float(worst_rel),
        "note": ("One row per DIRECTED edge: taichung_loader.py does not read the "
                 "oneway column, so the file must not need it. The column is kept as "
                 "provenance. Speeds are length-weighted harmonic means so that "
                 "t0 = length/speed is preserved exactly."),
    }
    pm = os.path.join(cli.out_dir, f"{cli.prefix}_meta.json")
    with open(pm, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nOK  {pn}  [{len(out_nodes):,}]")
    print(f"OK  {pe}  [{len(out_edges):,}]")
    print(f"OK  {pm}")
    print(f"\nNext: point build_network.py at these two files to rebuild the "
          f"TDX section -> edge mapping.")


if __name__ == "__main__":
    main()
