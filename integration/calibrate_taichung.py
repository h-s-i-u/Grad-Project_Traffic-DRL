#!/usr/bin/env python3
"""Calibrate the Taichung graph so the herding effect is actually observable.

The CSV capacity is in veh/h (~1360-8000), but a run only routes a few hundred/
thousand vehicles, so at the real capacity every edge sits near rho~0 and nothing
congests -> no herding to suppress. This script finds a (n_vehicles, capacity_scale)
where the naive shortest-path ("herding") baseline already congests (worst-rho > 1),
which is the regime where the global penalty / DRL agent can show a benefit.

Part 1 : fast sweep over vehicle count (shortest-path baseline only) -> where it congests.
Part 2 : full 5-policy comparison at a chosen setting -> validates per-edge capacity
         end-to-end (metrics + policy_incremental + RoutingEnv via the placeholder agent)
         and shows the real-data herding story.

    cd integration && python calibrate_taichung.py
"""
import numpy as np
import networkx as nx

import config as C
import metrics as M
import policies as pol
from taichung_loader import load_taichung_graph

SEED = 42
SWEEP_N = [500, 1000, 2000, 4000]     # vehicles to try in the fast sweep
TARGET_RHO = 3.0                       # aim the full comparison at ~this worst-rho


def hotspot_demand(scc, hubs, n, rng):
    """n vehicles: random origins in the SCC, destinations funnelled to the hubs."""
    origins = rng.choice(scc, size=n)
    dests = rng.choice(hubs, size=n)
    return [(int(o), int(d)) for o, d in zip(origins, dests) if o != d]


def herding_loads(g, demand, hub_trees):
    """Route every vehicle on its free-flow shortest path (the herding baseline) using
    precomputed per-hub shortest-path trees, and return the per-edge load dict."""
    load = {}
    served = 0
    for o, h in demand:
        path = hub_trees[h].get(o)
        if not path:
            continue
        served += 1
        for e in zip(path[:-1], path[1:]):
            load[e] = load.get(e, 0.0) + 1.0
    return load, served


def worst_rho_and_gini(g, load):
    """max(load/cap) and Gini of the edge-load distribution (over used edges)."""
    worst = max((v / g.edges[e]["cap"] for e, v in load.items()), default=0.0)
    gini = M.gini(list(load.values())) if load else 0.0
    return worst, gini


def main():
    print("Loading Taichung graph (largest SCC, capacity_scale=1.0)...")
    g = load_taichung_graph(largest_scc_only=True, capacity_scale=1.0, verbose=True)
    scc = sorted(g.nodes())
    hubs = sorted(scc, key=lambda n: g.in_degree(n), reverse=True)[:C.N_HOTSPOTS]
    print(f"hubs (top in-degree) = {hubs}\n")

    # Per-hub shortest-path trees on the reversed graph: paths[o] = o -> hub (free-flow).
    rev = g.reverse(copy=False)
    hub_trees = {}
    for h in hubs:
        _, paths = nx.single_source_dijkstra(rev, h, weight="t0")
        hub_trees[h] = {o: p[::-1] for o, p in paths.items()}   # reverse -> o..h in original

    # ---- Part 1: fast sweep -------------------------------------------------
    print("=== Part 1: congestion sweep (herding baseline) ===")
    print(f"{'vehicles':>9} | {'served':>6} | {'worst-rho@scale1':>16} | {'Gini(load)':>10} "
          f"| {'scale for rho=3':>15}")
    print("-" * 76)
    rng = np.random.default_rng(SEED)
    for n in SWEEP_N:
        demand = hotspot_demand(scc, hubs, n, rng)
        load, served = herding_loads(g, demand, hub_trees)
        worst, gini = worst_rho_and_gini(g, load)
        scale_for_3 = worst / TARGET_RHO if worst > 0 else float("nan")
        print(f"{n:>9} | {served:>6} | {worst:>16.4f} | {gini:>10.3f} | {scale_for_3:>15.4f}")
    print("\nRead: pick a vehicle count, then set config.KNN aside and use capacity_scale\n"
          "= (worst-rho@scale1 / desired worst-rho) so the herding baseline actually congests.\n")

    # ---- Part 2: full 5-policy comparison at a congesting setting -----------
    n2 = 800
    demand = hotspot_demand(scc, hubs, n2, np.random.default_rng(SEED + 1))
    load, _ = herding_loads(g, demand, hub_trees)
    worst1, _ = worst_rho_and_gini(g, load)
    scale = max(worst1 / TARGET_RHO, 1e-4)     # bring herding worst-rho to ~TARGET_RHO
    print(f"=== Part 2: full comparison at n={n2}, capacity_scale={scale:.4f} "
          f"(targets worst-rho~{TARGET_RHO}) ===")

    gs = load_taichung_graph(largest_scc_only=True, capacity_scale=scale, verbose=False)
    runs = {
        "static": pol.policy_static(gs, demand),
        "prediction-greedy (HERDING)": pol.policy_prediction_greedy(gs, demand),
        "load-aware": pol.policy_load_aware(gs, demand),
        "global-penalty (oracle)": pol.policy_global_penalty(gs, demand),
        # max_hops matters here: city routes run 30-70 hops, so the 60-hop default
        # abandons most trips before they arrive and understates the rollout.
        "drl-placeholder": pol.policy_drl(gs, demand, pol.make_drl_agent("placeholder", gs),
                                          max_hops=C.TAICHUNG_MAX_HOPS),
    }
    ref = set()
    for paths in runs.values():
        l, _ = M.edge_loads(gs, paths)
        ref |= {e for e, v in l.items() if v > 0}
    ref = sorted(ref)

    print(f"{'policy':>28} | {'ATT(s)':>8} | {'worst-rho':>9} | {'Gini':>6} | {'served':>6}")
    print("-" * 72)
    base = None
    for name, paths in runs.items():
        m = M.evaluate(gs, paths, ref)
        if name.startswith("prediction-greedy"):
            base = m
        print(f"{name:>28} | {m['att']:>8.1f} | {m['worst_rho']:>9.3f} | "
              f"{m['gini_load']:>6.3f} | {m['served']:>6}")
    if base:
        print("\nHerding suppression vs the HERDING baseline (negative = better):")
        for name, paths in runs.items():
            if name.startswith("prediction-greedy"):
                continue
            m = M.evaluate(gs, paths, ref)
            dg = 100 * (m["gini_load"] - base["gini_load"]) / base["gini_load"] if base["gini_load"] else 0
            print(f"  {name:>28} : Gini {dg:+6.1f}%")
    print("\n(drl-placeholder = analytic A*-greedy stand-in; it also validates that "
          "RoutingEnv now reads per-edge capacity.)")


if __name__ == "__main__":
    main()
