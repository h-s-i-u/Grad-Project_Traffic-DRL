#!/usr/bin/env python3
"""Why is the DRL agent's ATT worse than the analytic policies'?

Three explanations are usually confused with each other, and they call for different
fixes, so this separates them:

  detour       the agent picks physically longer routes. If it detours MORE than the
               oracle does, the eq.4 penalty terms are over-weighted for a myopic
               policy and lambda tuning is justified.
  congestion   the routes are a similar length but run through busier links -- the
               agent is spreading load badly, which is a capability gap, not a
               weighting one.
  attrition    the ~2.5% of trips the agent abandons are the hard ones, so its ATT is
               computed over an easier subset than the oracle's 100%. Comparing the
               headline ATT across policies with different served rates is not
               apples to apples at all.

The last one is handled by scoring every policy on the COMMON subset of trips that
all of them completed, which is the only fair comparison.

Reference point established without the agent (seed 42, 800 vehicles):
    static      detour 1.000x   ATT ~848
    load-aware  detour 1.035x   ATT ~476
    oracle      detour 1.133x   ATT ~468      <- detours the MOST and is still fastest
So detouring per se is not the problem; the question is whether the agent detours
further than the oracle without earning it back.

🔴 COMPARING TWO AGENTS NEEDS ONE RUN, NOT TWO. Pass --drl twice. Each agent abandons a
different set of trips, so a separate run computes its ATT over a different common
subset -- measured, two runs of the same demand differed by 8 trips and static's
ATT(common) moved 758.8 -> 786.7 (3.7%) between them, which is bigger than the effect
being looked for. Scored together, every agent shares one subset and the numbers mean
what they appear to mean.

Usage:
    cd integration
    python diagnose_agent.py --drl checkpoints/taichung/drl_agent_f10_800it.pt
    # two agents, one comparable subset
    python diagnose_agent.py --drl checkpoints/taichung/drl_agent_f10_800it.pt \\
                             --drl checkpoints/taichung/drl_togo25_f10_800it.pt
    # under the S3 arterial closure, same flags as run_compare.py
    python diagnose_agent.py --drl ... --close-road 臺灣大道 --close-at 0.50
"""
import argparse
import json
import os

import networkx as nx
import numpy as np

import closure as clo
import config as C
import metrics as M
import network as net
import policies as pol


def path_time(g, path, attr="t0"):
    return sum(g.edges[e][attr] for e in zip(path[:-1], path[1:]))


def realized_times(g, paths):
    """Per-vehicle realised travel time under this policy's own load, index-aligned
    with `paths` (None where the trip failed).

    metrics.evaluate() cannot be reused here: it drops the failed trips while
    building its list, so the positions no longer line up with the demand and a
    common-subset comparison would silently compare different vehicles.
    """
    load, _ = M.edge_loads(g, paths)
    tt = {}
    for e in g.edges():
        cap = g.edges[e]["cap"]
        tt[e] = g.edges[e]["t0"] * (1.0 + C.BPR_A * (load[e] / cap) ** C.BPR_B)
    return [None if not p else sum(tt[e] for e in zip(p[:-1], p[1:])) for p in paths]


def _demand(g, scc, hubs, n_vehicles, seed):
    """Same draw run_compare.make_demand produces, so the two tools agree."""
    rng = np.random.default_rng(seed)
    o = rng.choice(scc, n_vehicles)
    d = rng.choice(hubs, n_vehicles)
    return [(int(a), int(b)) for a, b in zip(o, d) if a != b], hubs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="taichung", choices=net.GRAPHS)
    ap.add_argument("--drl", required=True, action="append", metavar="CKPT",
                    help="trained checkpoint. Repeat the flag to score several agents "
                         "IN ONE RUN -- ATT(common) is only comparable between agents "
                         "evaluated on the same subset of trips (see below)")
    ap.add_argument("--vehicles", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--close-road", metavar="PREFIX", default=None,
                    help="S3: close every edge whose road_name starts with PREFIX")
    ap.add_argument("--close-edge", metavar="FROM,TO", default=None)
    ap.add_argument("--close-busiest", action="store_true")
    ap.add_argument("--close-at", type=float, default=0.5,
                    help="fraction of the demand dispatched before the closure")
    ap.add_argument("--close-demand", choices=["filter", "wave", "resample", "none"],
                    default="filter")
    cli = ap.parse_args()

    C.N_VEHICLES = cli.vehicles
    g, _ = net.build_graph_for(cli.graph, verbose=False)
    scc = sorted(net.largest_scc(g))
    hubs = sorted(scc, key=lambda n: g.in_degree(n), reverse=True)[:C.N_HOTSPOTS]

    # The setting each checkpoint was TRAINED with. Reading this is not optional:
    # togo_refresh changes what the policy observes, so evaluating a togo_refresh=25
    # agent at 0 degrades it silently. Measured cost of getting this wrong: served
    # 96.4% -> 51.3% on the same checkpoint, same seed, same graph.
    agents = []
    for ckpt in cli.drl:
        togo_refresh, tag = 0, ""
        meta_path = os.path.splitext(ckpt)[0] + ".meta.json"
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            togo_refresh = int(meta.get("togo_refresh", 0) or 0)
            tag = (f"iter {meta.get('iteration')}, trained served "
                   f"{meta.get('served', float('nan')):.1%}, "
                   f"togo_refresh={togo_refresh}, "
                   f"capacity_scale={meta.get('capacity_scale')}")
        elif ckpt not in ("placeholder", "oracle"):
            tag = ("⚠ no sidecar -- assuming togo_refresh=0. If it was trained with a "
                   "non-zero value, every number below is measured on observations the "
                   "agent never saw")
        name = os.path.basename(ckpt)
        agents.append({"ckpt": ckpt, "togo": togo_refresh, "name": name})
        print(f"  {name}: {tag}")

    closure = None
    if cli.close_road or cli.close_edge or cli.close_busiest:
        probe, _ = _demand(g, scc, hubs, cli.vehicles, cli.seed)
        load, total = clo.baseline_load(g, probe, pol.policy_prediction_greedy)
        try:
            if cli.close_busiest:
                label = clo.busiest_road(g, probe, load, total, hubs)
                edges = clo.edges_by_road(g, label)
            elif cli.close_road:
                label, edges = cli.close_road, clo.edges_by_road(g, cli.close_road)
            else:
                label, edges = cli.close_edge, clo.edges_by_endpoints(g, cli.close_edge)
            closure = clo.Closure(edges, at=cli.close_at, label=label)
        except ValueError as e:
            raise SystemExit(f"error: {e}")
        for line in clo.fmt_inspect(clo.inspect(g, closure, probe, load, total, hubs)):
            print(line)

    demand, _ = _demand(g, scc, hubs, cli.vehicles, cli.seed)
    n_drawn = len(demand)
    demand, dem_info = clo.select_demand(g, closure, demand, cli.close_demand)

    # Denominator of the detour ratio: the free-flow optimum on the network THAT
    # VEHICLE ACTUALLY SAW. Using the open-network optimum for post-closure vehicles
    # would fold the cost of the closure into the policies' detour figure, which is
    # what this tool exists to separate.
    cut = closure.cutoff(len(demand)) if closure is not None else len(demand)

    def free_flow_optima(view, indices):
        out = {}
        for od in {demand[i] for i in indices}:
            try:
                out[od] = nx.shortest_path_length(view, od[0], od[1], weight="t0")
            except nx.NetworkXNoPath:
                pass
        return out

    before = free_flow_optima(g, range(cut))
    after = (free_flow_optima(closure.view(g), range(cut, len(demand)))
             if closure is not None else {})
    ref_time = [(before if i < cut else after).get(demand[i])
                for i in range(len(demand))]

    max_hops = net.default_max_hops(cli.graph)
    runs = {
        "1 static": pol.policy_static(g, demand, closure),
        "4 hybrid (herding)": pol.policy_prediction_greedy(g, demand, closure=closure),
        "5 load-aware": pol.policy_load_aware(g, demand, closure),
        "6 oracle": pol.policy_global_penalty(g, demand, closure),
    }
    # Every agent is rolled out in the SAME run, so the common subset below is shared.
    # Scoring them in separate runs does not work: each agent abandons different trips,
    # so each run computes ATT over a different set. Measured -- the two agents' runs
    # differed by 8 trips and static's ATT(common) moved 758.8 -> 786.7 (3.7%) between
    # them, which is larger than the difference being looked for.
    for a in agents:
        a["stats"] = {}
        label = f"7 DRL {a['name']}" if len(agents) > 1 else "7 DRL agent"
        a["label"] = label
        runs[label] = pol.policy_drl(g, demand, pol.make_drl_agent(a["ckpt"], g),
                                     max_hops=max_hops, togo_refresh=a["togo"],
                                     closure=closure, stats=a["stats"])

    # Trips every policy completed. Anything else makes the ATT columns incomparable.
    common = [i for i in range(len(demand))
              if ref_time[i] and ref_time[i] > 0 and all(p[i] for p in runs.values())]
    print(f"\n=== setup ===")
    print(f"  {cli.graph} | {g.number_of_nodes():,} nodes / {g.number_of_edges():,} edges"
          f" | {len(demand)} vehicles | seed {cli.seed}"
          + (f" | closure {closure.label} at {closure.at:.0%}" if closure else ""))
    if dem_info.get("dropped"):
        print(f"  {dem_info['dropped']}/{n_drawn} trips have no route after the closure "
              f"and were removed (--close-demand {dem_info['mode']})")
    print(f"  trips completed by EVERY policy: {len(common)} / {len(demand)} "
          f"({len(common) / len(demand):.1%}) -- the fair comparison set")
    if len(agents) > 1:
        print(f"  {len(agents)} agents scored on that ONE subset, so their "
              f"ATT(common) are directly comparable")
    for a in agents:
        s = a["stats"]
        print(f"  {a['name']} failures: dead-end {s.get('dead_end', 0)}, "
              f"max-hops {s.get('max_hops', 0)}, unroutable {s.get('trivial', 0)}")

    ref = sorted({e for paths in runs.values()
                  for e, v in M.edge_loads(g, paths)[0].items() if v > 0})

    rows = {}
    for name, paths in runs.items():
        m = M.evaluate(g, paths, ref)
        det = np.array([path_time(g, paths[i]) / ref_time[i] for i in common])
        rt = realized_times(g, paths)          # ATT on the common set only, so
        rows[name] = {                          # attrition cannot flatter anyone
            "served": m["served"], "det": det.mean(),
            "p90": float(np.percentile(det, 90)),
            "hops": float(np.mean([len(paths[i]) - 1 for i in common])),
            "att_all": m["att"],
            "att_c": float(np.mean([rt[i] for i in common])),
        }
    # The oracle is the yardstick: "did the agent move toward it" is the question the
    # pre-registered criterion asks, and a raw ATT cannot answer it across runs.
    oracle = rows["6 oracle"]["att_c"]
    w = max(len(n) for n in rows) + 2

    print(f"\n=== detour and realised cost ===")
    print(f"  {'policy':<{w}}{'served':>8}{'detour':>9}{'p90':>8}{'hops':>7}"
          f"{'ATT(all)':>10}{'ATT(common)':>13}{'vs oracle':>11}")
    print("  " + "-" * (w + 66))
    for name, r in rows.items():
        gap = 100 * (r["att_c"] - oracle) / oracle
        print(f"  {name:<{w}}{r['served']:>8}{r['det']:>8.3f}x{r['p90']:>7.3f}x"
              f"{r['hops']:>7.1f}{r['att_all']:>10.1f}{r['att_c']:>13.1f}"
              f"{gap:>10.1f}%")

    if len(agents) > 1:
        print(f"\n=== agent vs agent (same {len(common)} trips) ===")
        base = rows[agents[0]["label"]]
        for a in agents[1:]:
            r = rows[a["label"]]
            d_att = 100 * (r["att_c"] - base["att_c"]) / base["att_c"]
            closed = ((base["att_c"] - r["att_c"]) / (base["att_c"] - oracle) * 100
                      if base["att_c"] > oracle else float("nan"))
            print(f"  {a['name']} vs {agents[0]['name']}:")
            print(f"    ATT(common) {base['att_c']:.1f} -> {r['att_c']:.1f} "
                  f"({d_att:+.1f}%), closing {closed:.0f}% of the gap to the oracle")
            print(f"    detour      {base['det']:.3f}x -> {r['det']:.3f}x "
                  f"(oracle {rows['6 oracle']['det']:.3f}x)")
            print(f"    served      {base['served']} -> {r['served']}")

    print(f"\n=== how to read it ===")
    print("  detour(DRL) > detour(oracle)  -> the agent avoids busy links too eagerly.")
    print("     The eq.4 penalty is too strong for a policy that only sees one hop")
    print("     ahead; lowering config.LAMBDA_VAR (0.8) or PENALTY_SCALE (12.0) is")
    print("     then a justified experiment.")
    print("  detour(DRL) ~ detour(oracle) but ATT(common) worse -> same length, busier")
    print("     roads. That is a capability gap (myopic vs global Dijkstra), and")
    print("     re-weighting the reward will not close it -- the oracle already beats")
    print("     load-aware on ATT, Gini AND worst-rho with these exact lambdas.")
    print("  ATT(all) much better than ATT(common) for the DRL row -> its headline")
    print("     number is carried by the trips it abandoned.")


if __name__ == "__main__":
    main()
