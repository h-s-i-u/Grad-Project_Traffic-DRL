#!/usr/bin/env python3
"""Sweep the eq.4 penalty weights, to find where the Gini target actually fails.

The proposal (section 4.4) says "alpha, lambda1 and lambda2 will be tuned by grid search
on the validation set". That never happened: alpha and lambda1 have sat at their defaults
throughout, and lambda2 was changed once (0.3 -> 0.8) on METR-LA with the result recorded
as "no effect, inside noise". Meanwhile the report's most visible miss is Gini, whose
current defence is that the shortfall is structural rather than a matter of tuning. That
defence is presently an argument. This makes it a measurement.

WHAT IS SWEPT, AND WHY ONLY THAT
    lambda2 enters two places: the analytic cost of policies 5/6, and RoutingEnv._gcost,
    which is the DRL agent's per-step reward. Policies 1-4 never see it, and policy 5
    runs with use_penalty=False. So changing lambda2 re-prices policy 6 immediately at no
    cost, while policy 7 would need a full retrain per value (~13 h each). Policy 6 is
    also exactly what the report's claim is about: it is the analytic optimiser of eq.4,
    so if eq.4 cannot reach the target even when travel time stops mattering, the
    objective itself cannot.

    alpha is NOT swept, and that is not an omission. The cost is

        alpha * BPR + t_ref * scale * (lambda1 * overflow + lambda2 * spread)

    and scaling (alpha, lambda1, lambda2) by any k > 0 scales the whole cost by k, which
    Dijkstra's argmin is invariant to. Only the ratios matter, so fixing alpha = 1 and
    sweeping lambda loses no generality. Worth stating in the report as the reason a
    three-dimensional grid was not needed.

WHY THIS DOES NOT CALL run_compare.run_once
    That function rebuilds the Gini reference edge set per run, as the union of edges any
    policy touched. Raising lambda2 makes policy 6 detour onto more edges, so the union
    grows, so EVERY policy's Gini shifts -- including the baseline the deltas are measured
    against. The denominator would move with the thing being swept. (實驗記錄 §13.23 ⑦
    recorded the same trap when the reference set moved with the number of policies.)

    Instead: pass one runs every cell and collects the union across ALL of them; pass two
    re-runs and scores against that one fixed set. Same seed gives the same demand and
    the policies are deterministic, so the second pass reproduces the first exactly.
    Consequence to state in the report: these Gini values are NOT directly comparable to
    §4.7's, which used a per-run union over seven policies including the DRL agent. Within
    this sweep they are comparable to each other, which is what the question needs.

HOW TO READ THE RESULT
    Large lambda2 makes policy 6 nearly stop caring about travel time and optimise for an
    even load. Whatever Gini it converges to is the floor this objective can reach on this
    scenario -- measured, not assumed. Three outcomes, all publishable:
      * it plateaus well short of the target  -> the structural claim becomes a number
      * it reaches the target at a large ATT cost -> the claim is wrong, and the honest
        version is a trade-off the objective would never choose
      * it reaches the target cheaply -> lambda2 = 0.8 was simply the wrong pick

    Caveat either way: this bounds the eq.4 FAMILY, not the graph. A true minimum-Gini
    flow assignment is a convex problem and could do better; policy 6 is a greedy
    incremental assignment. The report's claim is about eq.4, so the scope matches.

    cd integration
    python sweep_lambda.py                       # lambda2 sweep, 10 seeds per cell
    python sweep_lambda.py --lambda1 0.5,2.0     # add the lambda1 cross-check
"""
import argparse
import json
import sys

import numpy as np

# The tables use Greek letters and the docstring cites §-numbers, while Windows hands a
# redirected stdout cp1252 -- so piping this to a file would die on the text, not on
# anything real.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

import config as C
import metrics as M
import network as net
import policies as pol
from run_compare import make_demand

# 0 leaves only the saturation term; 0.3 is the proposal's default; 0.8 is what every
# reported number was produced with. The rest walk out towards "ignore travel time".
DEFAULT_L2 = "0,0.3,0.8,1.5,3.0,6.0,12.0"
GINI_TARGET = -30.0                       # proposal section 5
STATIC, HERD = "1 static", "4 hybrid (herding)"
COORD, ORACLE = "5 load-aware", "6 oracle (eq.4)"


def route_all(g, demand):
    """The four policies this sweep needs. 2 and 3 are omitted: they are Dijkstra on a
    single model's forecast and, like 1 and 4, never read lambda."""
    return {STATIC: pol.policy_static(g, demand),
            HERD: pol.policy_prediction_greedy(g, demand),
            COORD: pol.policy_load_aware(g, demand),
            ORACLE: pol.policy_global_penalty(g, demand)}


def structural_note(g, demand, hubs, hops_per_vehicle):
    """What the topology forces, before any policy gets a say.

    Every vehicle bound for a hub must cross one of that hub's in-edges on its last hop,
    and no weighting changes that. Printing it keeps the sweep honest in both directions:
    it is a real floor, and it is a floor on a small share of the traffic.
    """
    in_deg = {h: g.in_degree(h) for h in hubs}
    per_hub = {h: sum(1 for _, d in demand if d == h) for h in hubs}
    n_edges, n_veh = sum(in_deg.values()), sum(per_hub.values())
    cap = float(np.mean([g.edges[e]["cap"] for e in g.edges()]))
    total = n_veh * hops_per_vehicle
    share = n_veh / total
    txt = "\n".join([
        "structural floor (independent of any lambda):",
        f"  {len(hubs)} hubs, in-degrees {sorted(in_deg.values())} -> {n_edges} in-edges "
        f"carry every vehicle's last hop",
        f"  {n_veh} vehicles / {n_edges} edges = {n_veh / n_edges:.1f} per edge even if "
        f"perfectly spread (rho ~ {n_veh / n_edges / cap:.2f})",
        f"  but that is {n_veh:,} of ~{total:,.0f} edge traversals = {100 * share:.1f}% "
        f"of the load",
        f"  -> it pins the floor of worst-rho, which already meets its target. The other "
        f"{100 - 100 * share:.0f}% is what",
        f"     lambda2 can still move, and Gini is a distributional statistic over all "
        f"of it.",
    ])
    return txt, {"hub_in_edges": n_edges, "vehicles_to_hubs": n_veh,
                 "per_edge_if_even": n_veh / n_edges,
                 "rho_if_even": n_veh / n_edges / cap,
                 "forced_share_of_traversals": share}


def paired(scored, policy, key):
    """Mean ± std of the within-seed percentage change vs the herding baseline.

    Paired inside each seed, as 實驗設計 §4.7 requires: every policy saw the same demand,
    so pairing cancels the demand-to-demand variance instead of letting it into the bar.
    """
    d = [100 * (r[policy][key] - r[HERD][key]) / r[HERD][key]
         for r in scored if r[HERD][key]]
    a = np.asarray(d, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", choices=net.GRAPHS, default="taichung")
    ap.add_argument("--vehicles", type=int, default=800)
    ap.add_argument("--scenario", choices=["random", "hotspot"], default=C.SCENARIO)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--repeat", type=int, default=10, metavar="N",
                    help="demand draws per cell; the reporting protocol is 10")
    ap.add_argument("--lambda2", default=DEFAULT_L2,
                    help=f"comma-separated values to sweep (default {DEFAULT_L2})")
    ap.add_argument("--lambda1", default=None, metavar="A,B",
                    help="optional cross-check: repeat the sweep at these lambda1 values, "
                         "to show lambda2's effect does not hinge on lambda1")
    ap.add_argument("--capacity-scale", type=float, default=None)
    ap.add_argument("--out", default="lambda_sweep.json")
    args = ap.parse_args()

    C.SCENARIO, C.N_VEHICLES = args.scenario, args.vehicles
    l2s = [float(x) for x in args.lambda2.split(",")]
    l1s = [float(x) for x in args.lambda1.split(",")] if args.lambda1 else [C.LAMBDA_SAT]
    seeds = [args.seed + i for i in range(args.repeat)]
    cells = [(l1, l2) for l1 in l1s for l2 in l2s]

    g, _ = net.build_graph_for(args.graph, capacity_scale=args.capacity_scale,
                               verbose=False)
    scc = net.largest_scc(g)
    demands = [make_demand(g, scc, np.random.default_rng(s)) for s in seeds]
    probe, hubs = demands[0]

    print(f"\n{'=' * 98}\neq.4 penalty-weight sweep  (proposal §4.4's grid search, "
          f"finally run)\n{'=' * 98}")
    print(f"graph  : {args.graph}  {g.number_of_nodes():,} nodes / "
          f"{g.number_of_edges():,} edges")
    print(f"demand : {len(probe):,} vehicles, {args.scenario}, {len(seeds)} seeds "
          f"{seeds[0]}..{seeds[-1]}")
    print(f"sweep  : lambda2 {l2s}\n         lambda1 {l1s}   alpha fixed at {C.ALPHA} "
          f"(a redundant scale -- see the module docstring)")
    print(f"policy 7 is not run: lambda2 is baked into its reward, so a fair point would "
          f"need a full\n         retrain per value. This bounds policy 6, which is what "
          f"the report's claim is about.\n")
    if args.scenario == "hotspot":
        note, struct = structural_note(g, probe, hubs, 31.4)
        print(note)
    else:
        # The hub in-edge count only means something when the demand is funnelled into
        # those hubs. Under `random` the destinations are spread over the whole SCC, so
        # the same arithmetic printed "5 vehicles / 16 edges" -- true, and completely
        # misleading. A number that is only meaningful under one setting has to be
        # suppressed under the others, not left to be misread.
        struct = None
        print(f"structural floor: not applicable under scenario '{args.scenario}' -- the "
              f"hub in-edge\n  bottleneck only binds when the demand is funnelled into "
              f"those hubs.")

    # --- pass 1: one fixed reference edge set for the whole sweep ---
    print(f"\npass 1/2  collecting the reference edge set over {len(cells)} cells x "
          f"{len(seeds)} seeds ...")
    ref = set()
    for l1, l2 in cells:
        C.LAMBDA_SAT, C.LAMBDA_VAR = l1, l2
        for dem, _ in demands:
            for paths in route_all(g, dem).values():
                load, _ = M.edge_loads(g, paths)
                ref |= {e for e, v in load.items() if v > 0}
    ref = sorted(ref)
    print(f"          {len(ref):,} of {g.number_of_edges():,} edges are touched by some "
          f"policy in some cell;\n          every cell below is scored over exactly this "
          f"set, so the denominator cannot move.")

    # --- pass 2: score every cell against it ---
    print(f"pass 2/2  scoring ...\n")
    rows, invariant = [], None
    for l1, l2 in cells:
        C.LAMBDA_SAT, C.LAMBDA_VAR = l1, l2
        scored = [{k: M.evaluate(g, v, ref) for k, v in route_all(g, dem).items()}
                  for dem, _ in demands]
        # Policies 1 and 4 do not read lambda. If their ATT moves, the sweep is
        # perturbing the demand or the graph rather than the cost, and every delta
        # below would be measured against a denominator that shifted. Checked, not
        # assumed.
        fixed = tuple(round(float(np.mean([r[n]["att"] for r in scored])), 9)
                      for n in (STATIC, HERD))
        if invariant is None:
            invariant = fixed
        elif fixed != invariant:
            raise SystemExit(
                f"error: policies 1/4 moved at lambda1={l1}, lambda2={l2}. They never "
                f"read lambda,\n  so this means the sweep is changing something it "
                f"should not.\n  {invariant} -> {fixed}")
        cell = {"lambda1": l1, "lambda2": l2,
                "baseline_gini_abs": float(np.mean([r[HERD]["gini_load"] for r in scored]))}
        for lbl, name in (("oracle", ORACLE), ("coord", COORD)):
            for key in ("att", "gini_load", "worst_rho"):
                cell[f"{lbl}_{key}"] = list(paired(scored, name, key))
            cell[f"{lbl}_gini_abs"] = float(np.mean([r[name]["gini_load"] for r in scored]))
        rows.append(cell)

    print(f"{'=' * 98}\npolicy 6 (the analytic optimiser of eq.4), paired Δ vs "
          f"'{HERD}', {len(seeds)} seeds\n{'=' * 98}")
    hdr = (f"{'λ1':>5} {'λ2':>6} | {'ATT Δ':>17} | {'Gini Δ':>17} | {'worst-ρ Δ':>17} | "
           f"{'Gini abs':>8}")
    print(hdr + "\n" + "-" * len(hdr))
    for c in rows:
        tag = "  ← reported setting" if (c["lambda1"] == 0.5 and c["lambda2"] == 0.8) else ""
        if c["oracle_gini_load"][0] <= GINI_TARGET:
            tag += "  ✓ TARGET"
        print(f"{c['lambda1']:>5} {c['lambda2']:>6} | "
              f"{c['oracle_att'][0]:+8.1f}±{c['oracle_att'][1]:<5.1f}% | "
              f"{c['oracle_gini_load'][0]:+8.1f}±{c['oracle_gini_load'][1]:<5.1f}% | "
              f"{c['oracle_worst_rho'][0]:+8.1f}±{c['oracle_worst_rho'][1]:<5.1f}% | "
              f"{c['oracle_gini_abs']:8.4f}{tag}")

    best = min(rows, key=lambda c: c["oracle_gini_load"][0])
    rep = next((c for c in rows if c["lambda1"] == 0.5 and c["lambda2"] == 0.8), None)
    print(f"\nbest Gini: {best['oracle_gini_load'][0]:+.1f}% at lambda1={best['lambda1']}, "
          f"lambda2={best['lambda2']}   (target {GINI_TARGET:+.0f}%)")
    if rep:
        print(f"vs the reported (0.5, 0.8): Gini "
              f"{best['oracle_gini_load'][0] - rep['oracle_gini_load'][0]:+.1f} pp "
              f"for ATT {best['oracle_att'][0] - rep['oracle_att'][0]:+.1f} pp")
    if best["oracle_gini_load"][0] > GINI_TARGET:
        print(f"\n=> eq.4 does not reach {GINI_TARGET:.0f}% here even when travel time is "
              f"almost disregarded.\n   The shortfall belongs to the objective and the "
              f"scenario, not to the weight that was\n   picked. ⚠️ This bounds the eq.4 "
              f"family, not the graph: a true minimum-Gini flow\n   assignment is convex "
              f"and could do better than a greedy incremental one.")
    else:
        print(f"\n=> eq.4 DOES reach {GINI_TARGET:.0f}%. The claim that the target is "
              f"structurally out of reach\n   is WRONG and has to be rewritten as a "
              f"trade-off, quoting the ATT cost above.")

    out = {"graph": args.graph, "scenario": args.scenario, "vehicles": len(probe),
           "seeds": seeds, "alpha": C.ALPHA, "baseline": HERD,
           "gini_target_pct": GINI_TARGET, "ref_edges": len(ref),
           "ref_note": "one fixed edge set across all cells; NOT the per-run union "
                       "§4.7 used, so these Gini values are comparable within this "
                       "sweep but not against §4.7",
           "structural": struct, "cells": rows}
    path = C.HERE / args.out
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
