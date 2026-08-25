#!/usr/bin/env python3
"""Beam width 1 must reproduce greedy decoding EXACTLY. Nothing else is checked first.

`RoutingEnv.beam_route` rebuilds the candidate features, the eq.4 cost, the visited
set, the closure mask and the per-vehicle load bookkeeping that `step()` already does.
Any disagreement -- a stale saturation, the wrong mean_rho, a missed closure edge --
produces routes that still look plausible and metrics that are still in the right
range. Nothing raises.

So the first thing beam search has to prove is that at width 1 it is the SAME
algorithm: identical paths, vehicle by vehicle, node by node. Once that holds, a
difference at width > 1 can be attributed to the search rather than to a bug in the
re-implementation.

Both decoders are also run against a torch-free analytic agent, so this test works
without a GPU or a checkpoint; pass --drl to repeat it with a trained one.

Usage:
    cd integration
    python test_beam.py
    python test_beam.py --drl checkpoints/taichung/drl_togo25_f10_800it.pt
    python test_beam.py --close-road 臺灣大道 --close-at 0.10
"""
import argparse
import json
import os
import sys

import numpy as np

import closure as clo
import config as C
import metrics as M
import network as net
import policies as pol


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="taichung", choices=net.GRAPHS)
    ap.add_argument("--drl", default="placeholder",
                    help="checkpoint, or 'placeholder' for the analytic agent (default)")
    ap.add_argument("--vehicles", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--widths", default="1,2,4",
                    help="beam widths to run after the equivalence check")
    ap.add_argument("--close-road", metavar="PREFIX", default=None)
    ap.add_argument("--close-at", type=float, default=0.10)
    cli = ap.parse_args()

    C.N_VEHICLES = cli.vehicles
    g, _ = net.build_graph_for(cli.graph, verbose=False)
    scc = sorted(net.largest_scc(g))
    hubs = sorted(scc, key=lambda n: g.in_degree(n), reverse=True)[:C.N_HOTSPOTS]
    rng = np.random.default_rng(cli.seed)
    o, d = rng.choice(scc, cli.vehicles), rng.choice(hubs, cli.vehicles)
    demand = [(int(a), int(b)) for a, b in zip(o, d) if a != b]

    togo = 0
    meta_path = os.path.splitext(cli.drl)[0] + ".meta.json"
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            togo = int(json.load(f).get("togo_refresh", 0) or 0)

    cl = None
    if cli.close_road:
        cl = clo.Closure(clo.edges_by_road(g, cli.close_road), at=cli.close_at,
                         label=cli.close_road)
        demand, info = clo.select_demand(g, cl, demand, "filter")
        print(f"  closure {cli.close_road} at {cli.close_at:.0%}; "
              f"{info['dropped']} infeasible trips removed")

    max_hops = net.default_max_hops(cli.graph)
    agent = pol.make_drl_agent(cli.drl, g)
    print(f"  {cli.graph} | {len(demand)} vehicles | seed {cli.seed} | "
          f"agent {os.path.basename(cli.drl)} | togo_refresh={togo}")

    def run(beam):
        st = {}
        paths = pol.policy_drl(g, demand, agent, max_hops=max_hops, stats=st,
                               togo_refresh=togo, closure=cl, beam=beam)
        return paths, st

    print("\n=== width-1 equivalence ===")
    greedy, gst = run(0)
    # policy_drl sends beam<=1 down the greedy branch on purpose, so the identity check
    # has to drive the beam decoder directly rather than through beam=1.
    env = pol.RoutingEnv(g, demand, use_penalty=True, max_hops=max_hops,
                         togo_refresh=togo, closure=cl)
    env.reset()
    while not env.done:
        env.commit(*env.beam_route(agent, 1, max_hops))
    beam1 = env.paths
    bst = {"dead_end": env.n_deadend, "max_hops": env.n_maxhops,
           "trivial": env.n_trivial}

    same = sum(1 for a, b in zip(greedy, beam1) if a == b)
    print(f"  identical routes: {same} / {len(demand)}")
    if same != len(demand):
        for i, (a, b) in enumerate(zip(greedy, beam1)):
            if a != b:
                j = next((k for k, (x, y) in enumerate(zip(a or [], b or []))
                          if x != y), min(len(a or []), len(b or [])))
                print(f"\n  first divergence at vehicle {i}, hop {j}:")
                print(f"    greedy {(a or [])[max(0, j - 2):j + 3]}")
                print(f"    beam-1 {(b or [])[max(0, j - 2):j + 3]}")
                break
        raise SystemExit(
            "\nFAIL  width 1 is not the greedy decoder. beam_route rebuilds the "
            "candidate\n      features and the load bookkeeping; one of them disagrees "
            "with step().\n      Check, in this order: the beam's own load delta "
            "(step() adds 1.0 per hop\n      as it goes), mean_rho, the closure mask, "
            "and the visited set.\n      Do NOT report any width>1 number until this "
            "passes.")
    print(f"  failures  greedy dead-end {gst.get('dead_end', 0)} / beam-1 "
          f"{bst.get('dead_end', 0)}")
    print("  PASS -- width 1 is the greedy decoder, so any width>1 difference is the "
          "search.")

    print("\n=== width sweep ===")
    ref = set()
    runs = {}
    for wdt in [int(x) for x in cli.widths.split(",")]:
        paths, st = (greedy, gst) if wdt == 1 else run(wdt)
        runs[wdt] = (paths, st)
        load, _ = M.edge_loads(g, paths)
        ref |= {e for e, v in load.items() if v > 0}
    ref = sorted(ref)
    print(f"  {'width':>6}{'served':>8}{'dead-end':>10}{'max-hops':>10}"
          f"{'ATT':>10}{'worst rho':>11}{'Gini':>9}")
    print("  " + "-" * 64)
    for wdt, (paths, st) in runs.items():
        m = M.evaluate(g, paths, ref)
        print(f"  {wdt:>6}{m['served']:>8}{st.get('dead_end', 0):>10}"
              f"{st.get('max_hops', 0):>10}{m['att']:>10.1f}"
              f"{m['worst_rho']:>11.4f}{m['gini_load']:>9.4f}")
    print("\n  ATT across widths is NOT comparable while served differs -- the trips a "
          "narrow\n  beam abandons are the hard ones. Read dead-end first; ATT only "
          "once served matches.")


if __name__ == "__main__":
    main()
