"""Build the road network as a graph (NOT a 2D embedding).

We keep the problem ON THE GRAPH and do not project the cost matrix into 2D,
because (a) a distance/speed cost matrix is generally non-metric (asymmetric
speeds violate the triangle inequality) so a 2D embedding is lossy, and (b)
navigation is a graph routing problem, not a Euclidean tour. Every edge carries a
real travel time and a capacity, which is what lets us model congestion and the
herding effect.
"""
import pickle

import networkx as nx
import numpy as np

import config as C


def load_adjacency():
    """METR-LA Gaussian-kernel adjacency (directed, self-loops = 1)."""
    with open(C.ADJ_PKL, "rb") as f:
        try:
            _, _, adj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            _, _, adj = pickle.load(f, encoding="latin1")
    return np.asarray(adj, dtype=float)


def load_ensemble_speed():
    """STGCN+STGAT ensemble node speed (mph), clamped to the physical range.

    Returns (speed[N], info). The clamp to [SPEED_MIN, SPEED_MAX] is now just a
    guard: both models predict within 0-70 mph. `info` still reports each model's
    raw range and flags any out-of-range drift, so a future regression would be
    caught.
    """
    stgcn = np.load(C.STGCN_PRED).flatten()
    stgat = np.load(C.STGAT_PRED).flatten()

    def clamp(s):
        return np.clip(s, C.SPEED_MIN, C.SPEED_MAX)

    speed = C.W_STGCN * clamp(stgcn) + C.W_STGAT * clamp(stgat)
    info = {
        "stgcn_range": (float(stgcn.min()), float(stgcn.max())),
        "stgat_range": (float(stgat.min()), float(stgat.max())),
        "stgat_out_of_range": bool(stgat.max() > C.SPEED_MAX * 1.5),
        # per-model speeds kept separately so the proposal's baselines (2) "pure STGCN
        # + Dijkstra" and (3) "pure STGAT + Dijkstra" can each route on their OWN
        # prediction, instead of all three sharing the ensemble.
        "variants": {"stgcn": clamp(stgcn), "stgat": clamp(stgat)},
    }
    return speed, info


def recover_length(adj, eps=1e-6):
    """Gaussian kernel adj = exp(-d^2/sigma^2)  ->  relative distance d ∝ sqrt(-ln adj).

    One consistent convention across the whole pipeline.
    """
    with np.errstate(divide="ignore"):
        d = np.sqrt(-np.log(np.clip(adj, eps, 1.0)))
    return d


def build_graph(adj, speed, knn=None, speed_variants=None):
    """Directed road graph. Each edge gets:
        length : relative road length (from the kernel)
        t0     : free-flow travel time          = length / SPEED_MAX
        tpred  : predicted (uncongested) time    = length / mean(predicted speed)
        cap    : capacity proxy

    `speed_variants` is an optional {name: speed[N]} of per-model predictions; each
    adds a `tpred_<name>` edge attribute. That is what lets the proposal's separate
    baselines (2) pure-STGCN and (3) pure-STGAT each route on their own prediction
    while (4) routes on the ensemble.

    k-NN sparsification (knn>0, default C.KNN): each node keeps only its `knn`
    strongest-adjacency (= nearest) out-neighbours, turning the dense Gaussian-kernel
    graph (~76 edges/node) into a realistic sparse road network. knn=0 keeps all edges.
    """
    n = adj.shape[0]
    length = recover_length(adj)
    knn = C.KNN if knn is None else knn
    variants = speed_variants or {}
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        cand = [j for j in range(n) if j != i and adj[i, j] > C.ADJ_THRESHOLD]
        if knn and len(cand) > knn:
            cand = sorted(cand, key=lambda j: adj[i, j], reverse=True)[:knn]
        for j in cand:
            edge_speed = max(0.5 * (speed[i] + speed[j]), C.SPEED_MIN)
            attrs = {
                "length": float(length[i, j]),
                "t0": float(length[i, j] / C.SPEED_MAX),
                "tpred": float(length[i, j] / edge_speed),
                "cap": float(C.EDGE_CAPACITY),
            }
            for name, sp in variants.items():
                v = max(0.5 * (sp[i] + sp[j]), C.SPEED_MIN)
                attrs[f"tpred_{name}"] = float(length[i, j] / v)
            g.add_edge(i, j, **attrs)
    return g


def largest_scc(g):
    """Largest strongly connected component — guarantees sampled OD pairs are routable."""
    return max(nx.strongly_connected_components(g), key=len)


GRAPHS = ("metr-la", "taichung")


def build_graph_for(name="metr-la", capacity_scale=None, verbose=True):
    """Build the road graph for a named dataset -> (graph, info).

    Single entry point shared by run_compare.py and train_drl.py, so switching
    datasets is one flag rather than a different code path in each script.

    "metr-la"  : Gaussian-kernel sensor graph (207 nodes); edge speeds come from the
                 STGCN+STGAT ensemble predictions (stg{cn,at}_pred.npy). Times are in
                 RELATIVE units (the kernel's sigma is unknown).
    "taichung" : real OSM road network (Map/graph_*_taichung.csv); `t0` is in SECONDS
                 from the posted speed limit and each edge carries its own capacity.
                 `tpred` falls back to `t0` until a live-speed source (TDX) is wired
                 into taichung_loader's `current_speed` — until then the
                 prediction-greedy baseline is identical to the static one.

    `info` always carries "dataset"; the METR-LA keys (stgcn_range / stgat_range /
    stgat_out_of_range) are absent for taichung, so read them with .get().
    """
    if name == "metr-la":
        adj = load_adjacency()
        speed, info = load_ensemble_speed()
        g = build_graph(adj, speed, speed_variants=info.get("variants"))
        info["dataset"] = name
        info["speed_range"] = (float(speed.min()), float(speed.max()))
        return g, info

    if name == "taichung":
        # local import: keeps pandas optional for the METR-LA path
        from taichung_loader import load_taichung_graph, load_current_speed
        scale = C.TAICHUNG_CAPACITY_SCALE if capacity_scale is None else capacity_scale
        # Predicted edge speeds from make_drl_input.py; empty until the models have run,
        # in which case tpred falls back to t0 and baselines (2)(3)(4) degrade to (1).
        current, variants = load_current_speed(verbose=verbose)
        g = load_taichung_graph(
            default_speed_kmh=C.TAICHUNG_DEFAULT_SPEED_KMH,
            capacity_scale=scale,
            current_speed=current,
            speed_variants=variants,
            largest_scc_only=True,          # every sampled OD pair must be routable
            tpred_fallback=C.TAICHUNG_TPRED_FALLBACK,
            verbose=verbose,
        )
        return g, {"dataset": name, "capacity_scale": scale,
                   "predicted_edges": len(current),
                   "tpred_fallback": C.TAICHUNG_TPRED_FALLBACK}

    raise ValueError(f"unknown graph '{name}' (expected one of {GRAPHS})")


def default_max_hops(name):
    """Hop budget per dataset: city routes are far longer than METR-LA's dense kernel."""
    return C.TAICHUNG_MAX_HOPS if name == "taichung" else 60
