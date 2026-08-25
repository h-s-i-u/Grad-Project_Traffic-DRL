"""Load the Taichung OSM road network (nodes + edges CSV) into a routing-ready graph.

Drop-in replacement for `network.build_graph()`'s output: the returned NetworkX
DiGraph plugs straight into the existing decision pipeline (policies / metrics /
EGATActorCritic / RoutingEnv), so the agent can run on the real road network
instead of the METR-LA kernel.

Input CSV schema (from the data team):
    graph_nodes_taichung.csv : node_id, latitude, longitude
    graph_edges_taichung.csv : from_node, to_node, length_m,
                               free_flow_speed_kmh, lanes, capacity

Output graph:
  * Nodes are relabelled to a contiguous 0..N-1 index. This is REQUIRED: the E-GAT
    encoder indexes node-embedding rows by node id (H[node_id]) and builds its
    edge_index from the node ids, so they must be 0..N-1, not raw OSM ids.
  * Each edge carries:
        length : road length in metres (from the CSV)
        t0     : free-flow travel time in SECONDS = length_m / free_flow_speed
        tpred  : current/predicted travel time in seconds. Defaults to t0 until a
                 live-speed source (e.g. TDX) is wired in via `current_speed`.
        cap    : capacity (CSV veh/h, times `capacity_scale`)
        lanes, free_flow_speed_kmh : kept for reference / later calibration
  * Node attributes keep the original `osmid`, `lat`, `lon`.
  * Graph attributes `osmid_to_idx` / `idx_to_osmid` let you map TDX SectionIDs
    (matched to OSM node ids) back to graph indices later.

CAVEAT — per-edge capacity: `RoutingEnv` currently uses a single uniform capacity
(config.EDGE_CAPACITY). To honour the per-edge `cap` stored here it must be changed
to read g.edges[e]["cap"]; that is a separate edit (metrics.py already does).

CAVEAT — capacity units: METR-LA used cap=18 (abstract per-link load), whereas the
CSV capacity is veh/h (~1360-4079). With the same vehicle count the network never
congests, so herding never appears. Rescale via `capacity_scale` and/or raise the
demand volume until worst-rho > 1 (see `__main__` for a quick check).
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:                                     # reuse config paths when imported inside integration/
    import config as C
    _ROOT = C.ROOT
except Exception:                        # ...but stay runnable standalone
    _ROOT = Path(__file__).resolve().parent.parent

# The ARENA, not the full OSM export (TDX_Data/build_arena.py writes both files).
# On the full 7,489-node export only 524 of 20,347 edges (2.6%) carry a prediction, so
# `tpred == t0` almost everywhere and the Dijkstra-on-tpred baselines (2)(3)(4) come
# out identical to (1) -- measured: worst-rho 3.0882 for all four, Gini 0.8429~0.8436.
# Baseline (4) is the denominator of every delta in the report, so that collapse would
# make the herding comparison meaningless. The arena is 1,224 nodes / 2,342 edges at
# 14.8% coverage. Node ids are the original OSM ids, so routes still render on the
# full map; swap these two lines back to compare against the full network.
DEFAULT_EDGES = _ROOT / "Map" / "arena_edges_taichung.csv"
DEFAULT_NODES = _ROOT / "Map" / "arena_nodes_taichung.csv"
# written by STGCN/run_infer_taichung.py: from_node, to_node, speed_kmh
DEFAULT_PRED_EDGES = Path(__file__).resolve().parent / "taichung_pred_edges.csv"

MPS_PER_KMH = 1000.0 / 3600.0            # km/h -> m/s


def load_current_speed(path=DEFAULT_PRED_EDGES, verbose=True):
    """Read make_drl_input.py's edge-level predictions.

    Returns (current, variants):
        current  = {(from_osmid, to_osmid): km/h} from the ensemble  -> `tpred`
        variants = {"stgcn": {...}, "stgat": {...}}                  -> `tpred_stgcn/stgat`

    The variants are what let the proposal's baselines (2) pure-STGCN and (3) pure-STGAT
    route on their own prediction, mirroring `network.build_graph`'s `speed_variants`.

    Both come back empty if the file is absent, in which case every edge keeps its
    free-flow time: `tpred` collapses onto `t0` and the prediction-greedy baseline
    becomes identical to the static one.
    """
    path = Path(path)
    if not path.is_file():
        if verbose:
            print(f"[taichung] no predictions at {path}; tpred will fall back to t0. "
                  f"Run the two run_infer_taichung.py, then make_drl_input.py.")
        return {}, {}

    df = pd.read_csv(path, encoding="utf-8-sig")
    edges = list(zip(df["from_node"].astype("int64"), df["to_node"].astype("int64")))

    def column(name):
        return {e: float(s) for e, s in zip(edges, df[name])} if name in df else {}

    current = column("speed_hybrid")
    variants = {k: column(f"speed_{k}") for k in ("stgcn", "stgat")}
    variants = {k: v for k, v in variants.items() if v}

    if verbose:
        lo, hi = df["speed_hybrid"].min(), df["speed_hybrid"].max()
        print(f"[taichung] loaded {len(current)} predicted edge speeds "
              f"({lo:.1f}-{hi:.1f} km/h); variants: {', '.join(variants) or 'none'}")
    return current, variants


def load_taichung_graph(edges_csv=DEFAULT_EDGES, nodes_csv=DEFAULT_NODES,
                        default_speed_kmh=30.0, capacity_scale=1.0,
                        current_speed=None, speed_variants=None, bidirectional=False,
                        largest_scc_only=False, tpred_fallback="network_mean",
                        verbose=True):
    """Build a routing-ready DiGraph from the Taichung node/edge CSVs.

    Parameters
    ----------
    default_speed_kmh : fallback speed for edges whose free_flow_speed_kmh is blank.
    capacity_scale    : multiply CSV capacity by this (use to bring veh/h into the
                        abstract load scale the BPR / eq.4 reward was tuned on).
    current_speed     : optional {(from_osmid, to_osmid): speed_kmh} of predicted or
                        live speeds. When given, an edge's `tpred` uses it instead of
                        the free-flow speed. Missing edges fall back to free-flow.
    speed_variants    : optional {name: {(from,to): km/h}} of per-model speeds; each
                        adds a `tpred_<name>` edge attribute. Mirrors
                        network.build_graph's argument of the same name, and is what
                        lets the single-model baselines route on their own prediction.
    bidirectional     : add a reverse edge for any one-way edge (helps connectivity
                        if the CSV lists each road only once).
    largest_scc_only  : keep only the largest strongly connected component, then
                        relabel to a contiguous 0..M-1 index (guarantees every
                        sampled OD pair is routable — recommended for training).
    tpred_fallback    : what an edge with no prediction is assumed to be doing.
                        "network_mean" (default) scales its free-flow time by the
                        mean slowdown observed on the edges that DO have a
                        prediction; "free_flow" leaves tpred = t0, which biases
                        every prediction-following policy away from the
                        instrumented roads. See the long note further down.
    """
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    current_speed = current_speed or {}
    speed_variants = speed_variants or {}

    # --- contiguous node index (0..N-1), keyed by the original OSM node id ---
    osmid_to_idx = {int(osmid): i for i, osmid in enumerate(nodes["node_id"].tolist())}

    g = nx.DiGraph()
    for osmid, lat, lon in zip(nodes["node_id"], nodes["latitude"], nodes["longitude"]):
        g.add_node(osmid_to_idx[int(osmid)], osmid=int(osmid),
                   lat=float(lat), lon=float(lon))

    skipped, defaulted = 0, 0
    has_road_name = "road_name" in edges.columns
    # (u, v) -> {attr names that got a REAL prediction}. Needed for the fallback pass
    # below; recovering it afterwards by comparing tpred against t0 would misread any
    # edge whose predicted speed happens to equal its free-flow speed.
    real = {}
    for row in edges.itertuples(index=False):
        u = osmid_to_idx.get(int(row.from_node))
        v = osmid_to_idx.get(int(row.to_node))
        if u is None or v is None or u == v:
            skipped += 1                 # endpoint not in the node set, or a self-loop
            continue

        length_m = float(row.length_m)

        speed = row.free_flow_speed_kmh
        if pd.isna(speed) or float(speed) <= 0:
            speed = default_speed_kmh    # blank/invalid speed -> urban default
            defaulted += 1
        speed = float(speed)

        # capacity: CSV veh/h; fall back to lanes*1500 if blank, then scale.
        cap = row.capacity
        if pd.isna(cap):
            cap = (float(row.lanes) if not pd.isna(row.lanes) else 1.0) * 1500.0
        cap = max(float(cap) * capacity_scale, 1.0)

        # tpred uses a predicted speed for this edge if provided, else free-flow
        # (revisited by the fallback pass after the loop -- see below).
        key = (int(row.from_node), int(row.to_node))
        preds = {}
        cur = current_speed.get(key)
        got = cur is not None and float(cur) > 0
        preds["tpred"] = (float(cur) if got else speed, got)
        for name, table in speed_variants.items():
            s = table.get(key)
            got = s is not None and float(s) > 0
            preds[f"tpred_{name}"] = (float(s) if got else speed, got)

        attrs = {
            "length": length_m,
            "t0": length_m / (speed * MPS_PER_KMH),          # free-flow time (seconds)
            "cap": cap,
            "lanes": float(row.lanes) if not pd.isna(row.lanes) else 0.0,
            "free_flow_speed_kmh": speed,
            # Carried on the graph rather than re-read from the CSV by closure.py:
            # this function knows which file it actually loaded (edges_csv can be
            # overridden), so an independent reader could silently name roads from a
            # DIFFERENT export than the one being routed on. S3 selects what to close
            # by name, so that mismatch would close the wrong edges without erroring.
            # Absent from the pre-Map_fined export (graph_edges_taichung.csv), hence
            # the column check rather than a bare row.road_name.
            "road_name": ("" if not has_road_name or pd.isna(row.road_name)
                          else str(row.road_name)),
        }
        for name, (s, got) in preds.items():
            attrs[name] = length_m / (s * MPS_PER_KMH)       # current/predicted time
        # Explicit flag rather than testing tpred != t0: once the fallback pass runs,
        # EVERY edge differs from t0 and that test would report 100% coverage.
        attrs["observed"] = preds["tpred"][1]
        g.add_edge(u, v, **attrs)
        real[(u, v)] = {n for n, (_, got) in preds.items() if got}

    # --- what an UNOBSERVED edge is assumed to be doing ---------------------------
    # Leaving tpred = t0 there assumes every road without a sensor is at FREE FLOW,
    # which is not a neutral assumption. TDX reports real urban speeds (mean 24.8 km/h
    # on this network) while t0 comes from the 50 km/h limit, so every instrumented
    # road looks twice as slow as every uninstrumented one. Dijkstra-on-tpred then
    # routes systematically AWAY from the arterials that have data and into side
    # streets -- measured as baselines (2)(3)(4) finishing 18% SLOWER than free-flow
    # (1), which reads like herding but is an artefact of mixing two scales.
    #
    # "network_mean" instead assumes an unobserved road is as congested as the average
    # observed one: scale its free-flow time by the mean observed slowdown. Still an
    # assumption, and it must be disclosed, but it puts both groups on one scale.
    # Each attribute gets its OWN ratio -- STGCN and STGAT predict different means, and
    # sharing one ratio would leak one model's bias into the other's baseline.
    if tpred_fallback not in ("network_mean", "free_flow"):
        raise ValueError(f"tpred_fallback must be 'network_mean' or 'free_flow', "
                         f"got {tpred_fallback!r}")
    if tpred_fallback == "network_mean":
        names = ["tpred"] + [f"tpred_{n}" for n in speed_variants]
        for name in names:
            # t0/tpred == observed speed / free-flow speed, per edge
            obs = [d["t0"] / d[name] for u, v, d in g.edges(data=True)
                   if name in real.get((u, v), ())]
            if not obs:
                continue                       # no observations -> nothing to infer
            k = float(np.mean(obs))
            n_scaled = 0
            for u, v, d in g.edges(data=True):
                if name not in real.get((u, v), ()):
                    d[name] = d["t0"] / k
                    n_scaled += 1
            if verbose:
                print(f"[taichung] {name}: {len(obs)} observed edges average "
                      f"{k:.2f}x free-flow; {n_scaled} unobserved edges scaled to match")

    if bidirectional:
        for u, v, data in list(g.edges(data=True)):
            if not g.has_edge(v, u):
                g.add_edge(v, u, **data)

    if largest_scc_only and g.number_of_nodes():
        keep = max(nx.strongly_connected_components(g), key=len)
        g = g.subgraph(keep).copy()
        # re-contiguate to 0..M-1 (osmid stays as a node attribute)
        g = nx.convert_node_labels_to_integers(g, ordering="sorted")

    # (re)build the id maps against the final node set
    idx_to_osmid = {n: g.nodes[n]["osmid"] for n in g.nodes()}
    g.graph["idx_to_osmid"] = idx_to_osmid
    g.graph["osmid_to_idx"] = {o: i for i, o in idx_to_osmid.items()}

    if verbose:
        n, m = g.number_of_nodes(), g.number_of_edges()
        scc = max(nx.strongly_connected_components(g), key=len) if n else set()
        print(f"[taichung] {n} nodes / {m} edges "
              f"(skipped {skipped} edges, {defaulted} used default speed)")
        print(f"[taichung] avg out-degree {m / max(1, n):.2f}; "
              f"largest SCC = {len(scc)}/{n}")
    return g


if __name__ == "__main__":
    # Quick sanity check on the FULL graph (no pruning) — verifies fields, id
    # remapping, connectivity, units, and whether the demand scale will congest.
    import random

    g = load_taichung_graph(bidirectional=False, largest_scc_only=False)

    caps = [d["cap"] for *_, d in g.edges(data=True)]
    t0s = [d["t0"] for *_, d in g.edges(data=True)]
    if caps:
        print(f"[taichung] cap (veh/h)  min/mean/max: "
              f"{min(caps):.0f} / {np.mean(caps):.0f} / {max(caps):.0f}")
        print(f"[taichung] t0 (seconds) min/mean/max: "
              f"{min(t0s):.1f} / {np.mean(t0s):.1f} / {max(t0s):.1f}")

    # sample point-to-point hop counts inside the largest SCC (route length feel)
    scc = sorted(max(nx.strongly_connected_components(g), key=len)) if g.number_of_nodes() else []
    rng = random.Random(42)
    hops = []
    for _ in range(50):
        if len(scc) < 2:
            break
        a, b = rng.choice(scc), rng.choice(scc)
        if a != b:
            try:
                hops.append(nx.shortest_path_length(g, a, b))
            except nx.NetworkXNoPath:
                pass
    if hops:
        print(f"[taichung] sample shortest-path hops (n={len(hops)}): "
              f"min {min(hops)} / mean {np.mean(hops):.1f} / max {max(hops)}")

    print("[taichung] Tip: if the largest SCC is much smaller than the node count, "
          "re-run with bidirectional=True; for training use largest_scc_only=True.")
