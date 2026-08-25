#!/usr/bin/env python3
"""Train the PPO routing agent (proposal §4.4) and save a checkpoint.

Pipeline:
    build graph (from the vendored adjacency + STGCN/STGAT predictions)
      -> RoutingEnv with eq.(4) reward + arrival/fail shaping + per-episode demand
      -> EGATActorCritic trained by PPOTrainer (clipped surrogate, eq.5)
      -> periodic evaluation on a FIXED held-out hotspot demand
      -> save the checkpoint with the best Gini-weighted score

Design choices (confirmed):
  * demand is RESAMPLED every episode (hotspot scenario) so the agent generalizes
    over OD pairs rather than memorizing one instance.
  * reward shaping is ON: an arrival bonus and a stuck/max-hops penalty (both scaled
    by the mean free-flow edge time) are added on top of the eq.(4) per-step cost.
    The evaluation metrics (ATT/Gini/worst-rho via metrics.py) are computed
    independently, so they remain faithful to the proposal.
  * "best" checkpoint = highest Gini-weighted relative improvement vs the
    prediction-greedy HERDING baseline (0.25*ATT + 0.5*Gini + 0.25*worst-rho),
    discounted if the agent fails to route enough vehicles.

Checkpoints go to checkpoints/<graph>/drl_agent.pt -- one directory per road network.
The weights themselves carry no node-count dependency (unlike STGCN/STGAT, whose
per-node parameters make them transductive), so an agent CAN be loaded onto another
graph; but its score cannot be carried over, because METR-LA runs 1.5 hops in relative
time units and Taichung 41 hops in seconds, each with its own capacity_scale.

The trained checkpoint plugs straight into the benchmark:
    python run_compare.py --graph taichung --drl checkpoints/taichung/drl_agent.pt

Usage:
    python train_drl.py                       # METR-LA -> checkpoints/metr-la/
    python train_drl.py --iters 500 --train-vehicles 200
    # real Taichung road network (seconds, per-edge capacity, longer routes):
    python train_drl.py --graph taichung --train-vehicles 800 --eval-vehicles 800
    python train_drl.py --graph taichung --iters 20 --train-vehicles 300 \
                        --out checkpoints/taichung/smoke.pt      # smoke test
"""
import argparse
import json
import os

import networkx as nx
import numpy as np

import config as C
import metrics as M
import network as net
import policies as pol

try:
    import torch
except ImportError:                       # pragma: no cover
    raise SystemExit("train_drl.py requires PyTorch.  pip install torch")


def hotspot_demand(scc, hubs, n, rng):
    """n vehicles: random origins in the SCC, destinations funneled to the hubs."""
    origins = rng.choice(scc, size=n)
    dests = rng.choice(hubs, size=n)
    return [(int(o), int(d)) for o, d in zip(origins, dests) if o != d]


def save_meta(out, args, g, it, score, served, m):
    """Record, beside the checkpoint, the settings the agent was trained under.

    `togo_refresh` changes what the policy observes, so evaluating with a different
    value silently degrades the agent instead of raising -- exactly the failure mode
    that has bitten this project repeatedly. run_compare.py reads this file so the
    two cannot drift apart. Same sidecar pattern as the prediction .meta.json files.
    """
    meta = {
        "graph": args.graph, "iteration": it, "score": score, "served": served,
        "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
        "capacity_scale": (args.capacity_scale if args.capacity_scale is not None
                           else C.TAICHUNG_CAPACITY_SCALE),
        "togo_refresh": args.togo_refresh,
        "reward_scale": args.reward_scale,
        "arrival_mult": args.arrival_mult, "fail_mult": args.fail_mult,
        "shaping_gamma": args.shaping_gamma,
        "max_hops": args.max_hops, "train_vehicles": args.train_vehicles,
        "eval": {k: m[k] for k in ("att", "gini_load", "worst_rho", "served")},
    }
    with open(os.path.splitext(out)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def build_eval(g, eval_demand):
    """Fixed reference for scoring every checkpoint the same way:
    the analytic policies define a stable edge set (for Gini) and the
    prediction-greedy herding metrics we measure improvement against."""
    base = {
        "static": pol.policy_static(g, eval_demand),
        "herding": pol.policy_prediction_greedy(g, eval_demand),
        "load_aware": pol.policy_load_aware(g, eval_demand),
        "global_penalty": pol.policy_global_penalty(g, eval_demand),
    }
    ref = set()
    for paths in base.values():
        load, _ = M.edge_loads(g, paths)
        ref |= {e for e, v in load.items() if v > 0}
    ref = sorted(ref)
    return ref, M.evaluate(g, base["herding"], ref), M.evaluate(g, base["global_penalty"], ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", choices=net.GRAPHS, default="metr-la",
                    help="road network to train on: 'metr-la' (sensor kernel + STGCN/STGAT "
                         "ensemble) or 'taichung' (real OSM network, times in seconds)")
    ap.add_argument("--capacity-scale", type=float, default=None,
                    help="taichung only: scale the CSV veh/h capacity into the load range "
                         f"the reward was tuned on (default {C.TAICHUNG_CAPACITY_SCALE})")
    ap.add_argument("--iters", type=int, default=200, help="PPO iterations (episodes)")
    ap.add_argument("--train-vehicles", type=int, default=300, help="vehicles per training episode")
    ap.add_argument("--eval-vehicles", type=int, default=C.N_VEHICLES, help="vehicles in the held-out eval")
    ap.add_argument("--eval-every", type=int, default=25, help="evaluate + maybe checkpoint every N iters")
    ap.add_argument("--log-every", type=int, default=10, help="print a training line every N iters")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.03,
                    help="PPO entropy bonus (higher = more exploration; raised 0.01->0.03)")
    ap.add_argument("--max-hops", type=int, default=None,
                    help="give up a trip after this many hops (default: 60 for metr-la, "
                         f"{C.TAICHUNG_MAX_HOPS} for taichung, whose routes run 30-70 hops)")
    ap.add_argument("--arrival-mult", type=float, default=2.0,
                    help="arrival bonus = mult * the reward unit (see --reward-scale)")
    ap.add_argument("--fail-mult", type=float, default=5.0,
                    help="fail penalty = mult * the reward unit (see --reward-scale)")
    ap.add_argument("--reward-scale", default="trip", choices=["trip", "edge"],
                    help="what the multipliers above are relative to. 'trip' (default) "
                         "= mean free-flow shortest-path time over sampled demand, so "
                         "the multipliers mean the same thing on any network. 'edge' "
                         "reproduces the old behaviour and only works when routes are "
                         "a few hops long.")
    ap.add_argument("--togo-refresh", type=int, default=0,
                    help="recompute the agent's distance-to-destination estimate from "
                         "the eq.4 cost (not free-flow) every N vehicles. 0 keeps the "
                         "original free-flow estimate, which is congestion-blind: the "
                         "agent sees rho on the next edge but not beyond it, and its "
                         "routes come out shorter than the oracle's yet 13.6% slower. "
                         "Costs one Dijkstra per destination per refresh (4 under the "
                         "hotspot scenario).")
    ap.add_argument("--shaping-gamma", type=float, default=1.0,
                    help="potential-based shaping with Phi = -time-to-destination "
                         "(Ng et al. 1999); policy-invariant, gives dense per-step "
                         "progress feedback. Pass a negative value to disable.")
    ap.add_argument("--min-served", type=float, default=0.95, help="served-fraction target for full score credit")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
                    help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--out", default=None,
                    help="checkpoint path for the best agent "
                         "(default: checkpoints/<graph>/drl_agent.pt)")
    args = ap.parse_args()

    # argparse has no "optional float", so a negative value is the off switch.
    if args.shaping_gamma is not None and args.shaping_gamma < 0:
        args.shaping_gamma = None

    # Default per graph, so training Taichung cannot overwrite the METR-LA agent.
    if args.out is None:
        args.out = str(C.CKPT_DIR / args.graph / "drl_agent.pt")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    # --- network (same builder as run_compare) ---
    max_hops = args.max_hops or net.default_max_hops(args.graph)
    g, info = net.build_graph_for(args.graph, capacity_scale=args.capacity_scale)
    scc = sorted(net.largest_scc(g))
    hubs = sorted(scc, key=lambda n: g.in_degree(n), reverse=True)[:C.N_HOTSPOTS]
    t_ref = float(np.mean([g.edges[e]["t0"] for e in g.edges()]))

    # Reward scale: one TRIP, not one edge. Each step costs about one edge time, so a
    # trip of H hops accumulates ~-H*t_ref. Scaling the terminal bonus/penalty by
    # t_ref made "walk into a corner and end the episode" the highest-return policy on
    # any network with long routes -- METR-LA averages 1.5 hops and was fine, the
    # Taichung arena averages ~47 and was not. Sampling the actual hotspot demand
    # keeps the estimate faithful to the trips the agent will really be asked to make.
    if args.reward_scale == "trip":
        probe = hotspot_demand(scc, hubs, min(200, len(scc)),
                               np.random.default_rng(args.seed))
        costs = []
        for o, d in probe:
            try:
                costs.append(nx.shortest_path_length(g, o, d, weight="t0"))
            except nx.NetworkXNoPath:
                pass
        r_unit = float(np.mean(costs)) if costs else t_ref
        unit_label = f"mean free-flow TRIP time {r_unit:.4f} ({r_unit / t_ref:.1f} x t_ref)"
    else:
        r_unit = t_ref
        unit_label = f"mean free-flow EDGE time {r_unit:.4f} (legacy scale)"

    print(f"Graph '{args.graph}' {g.number_of_nodes()} nodes / {g.number_of_edges()} edges; "
          f"SCC {len(scc)}; hubs {hubs}; t_ref={t_ref:.4f}")
    print(f"Reward unit: {unit_label}")
    print(f"  arrival bonus {args.arrival_mult:+.2f} | fail penalty "
          f"{-args.fail_mult:+.2f} | per step about "
          f"{-t_ref / r_unit:.4f}  (rewards divided by the unit above)"
          + (f" | potential shaping on (gamma={args.shaping_gamma})"
             if args.shaping_gamma is not None else " | no potential shaping")
          + (f" | to-go from eq.4 cost, refreshed every {args.togo_refresh} vehicles"
             if args.togo_refresh else " | to-go = free-flow (congestion-blind)"))
    if args.graph == "metr-la":
        lo, hi = info["speed_range"]
        print(f"Ensemble speed {lo:.0f}-{hi:.0f} mph "
              f"(STGAT {info['stgat_range'][0]:.0f}-{info['stgat_range'][1]:.0f})")
    else:
        print(f"Edge times in seconds; capacity_scale={info['capacity_scale']}, "
              f"max_hops={max_hops}")
        # Report what is actually on the graph. This branch used to print the
        # "no signal yet" warning unconditionally, contradicting the loader's own
        # "loaded N predicted edge speeds" three lines above.
        obs_edges = sum(1 for _, _, d in g.edges(data=True) if d.get("observed"))
        if obs_edges:
            print(f"  {obs_edges:,} of {g.number_of_edges():,} edges carry a TDX "
                  f"prediction ({obs_edges / g.number_of_edges():.1%}); the rest are "
                  f"scaled to the mean observed slowdown.")
        else:
            print("  ⚠ tpred = t0 everywhere: no predictions loaded. Run "
                  "make_drl_input.py first.")

    # --- fixed held-out evaluation demand + reference baselines ---
    eval_rng = np.random.default_rng(args.seed + 9973)
    eval_demand = hotspot_demand(scc, hubs, args.eval_vehicles, eval_rng)
    eval_ref, herding_m, oracle_m = build_eval(g, eval_demand)
    base_att, base_gini, base_rho = herding_m["att"], herding_m["gini_load"], herding_m["worst_rho"]
    print(f"\nEval demand: {len(eval_demand)} vehicles (fixed). Reference on this demand:")
    print(f"  herding  baseline : ATT {base_att:.4f}  Gini {base_gini:.3f}  worst-rho {base_rho:.3f}")
    print(f"  oracle (eq.4)     : ATT {oracle_m['att']:.4f}  Gini {oracle_m['gini_load']:.3f}  "
          f"worst-rho {oracle_m['worst_rho']:.3f}  <- the agent's target to match/beat\n")

    # --- env (resampled hotspot demand each episode) + agent + PPO ---
    train_rng = np.random.default_rng(args.seed)
    env = pol.RoutingEnv(
        g,
        demand_fn=lambda: hotspot_demand(scc, hubs, args.train_vehicles, train_rng),
        use_penalty=True,
        max_hops=max_hops,
        arrival_bonus=args.arrival_mult * r_unit,
        fail_penalty=args.fail_mult * r_unit,
        shaping_gamma=args.shaping_gamma,
        togo_refresh=args.togo_refresh,
        # Rewards in units of one trip, not seconds. Raw seconds put the value target
        # near -640,000 and the value loss at 4.3e6 against a policy loss of 0.24;
        # clip_grad_norm_ then scaled the actor's gradient to nothing.
        reward_scale=r_unit,
    )
    agent = pol.EGATActorCritic(g).to(args.device)
    trainer = pol.PPOTrainer(env, agent, lr=args.lr, entropy_coef=args.entropy_coef)

    def evaluate_agent():
        agent.eval()
        fail = {}
        paths = pol.policy_drl(g, eval_demand, agent, max_hops=max_hops, stats=fail,
                               togo_refresh=args.togo_refresh)
        m = M.evaluate(g, paths, eval_ref)
        agent.train()
        served = m["served"] / max(1, len(eval_demand))

        def rel(b, x):
            return (b - x) / b if b else 0.0
        # weighted toward Gini (herding suppression), per the scoring choice
        improv = float(0.25 * rel(base_att, m["att"])
                       + 0.50 * rel(base_gini, m["gini_load"])
                       + 0.25 * rel(base_rho, m["worst_rho"]))
        score = improv - 2.0 * max(0.0, args.min_served - served)
        return score, m, served, fail

    # --- training loop ---
    # Selection is LEXICOGRAPHIC: (meets the served threshold, score). served% is an
    # entry requirement, not a term to trade off; among the checkpoints that qualify,
    # score alone decides.
    #
    # The reason is that ATT / Gini / worst-rho are computed over the vehicles that
    # ARRIVED, so scores from different served levels are measured on different
    # subsets of the demand -- and the vehicles a weak policy abandons are the hard
    # ones, which flatters exactly those metrics. `score` already subtracts
    # 2*(min_served - served), but that coefficient is a guess and nowhere near
    # enough: measured here, iter 175 had raw improvement 0.214 at 90% served against
    # iter 400's 0.094 at 98%, a 2.3x inflation that a 0.10 penalty cannot cancel.
    # The contaminated checkpoint won, and run_compare then refused to report it.
    # If a row below the threshold is NOT COMPARABLE for the report, it must not be
    # selectable for the checkpoint either.
    best, best_ok = -float("inf"), False
    agent.train()
    print(f"Training {args.iters} iters on {args.device} (train-vehicles={args.train_vehicles}); "
          f"saving best -> {args.out}")
    print(f"  eligible only at >= {args.min_served:.0%} served; a below-threshold "
          f"checkpoint is kept only if nothing ever qualifies\n")
    for it in range(1, args.iters + 1):
        traj = trainer.collect_episode()
        stats = trainer.update(traj)
        if it % args.eval_every == 0 or it == args.iters:
            score, m, served, fail = evaluate_agent()
            tag = ""
            ok = served >= args.min_served
            if (ok, score) > (best_ok, best):
                best, best_ok = score, ok
                agent.save(args.out)
                save_meta(args.out, args, g, it, score, served, m)
                tag = "  <- saved best" if ok else "  <- saved (below threshold)"
            # Break the failures down: a dead end means the policy trapped itself
            # (every neighbour already visited) and wants a stronger --fail-mult or
            # more training; hitting max_hops means it wandered and wants a larger
            # --max-hops. One served% number cannot tell you which knob to turn.
            miss = (f"  [dead-end {fail.get('dead_end', 0)}"
                    f" / max-hops {fail.get('max_hops', 0)}"
                    f" / unroutable {fail.get('trivial', 0)}]") if served < 0.99 else ""
            print(f"iter {it:4d} | ep_return {sum(t['reward'] for t in traj):8.2f} "
                  f"| ATT {m['att']:.4f}  Gini {m['gini_load']:.3f}  worst-rho {m['worst_rho']:.3f}  "
                  f"served {served * 100:3.0f}% | score {score:+.3f}{tag}{miss}")
        elif it % args.log_every == 0:
            print(f"iter {it:4d} | ep_return {sum(t['reward'] for t in traj):8.2f} "
                  f"| pi_loss {stats.get('policy_loss', 0):+.4f}  v_loss {stats.get('value_loss', 0):.4f}")

    print(f"\nDone. Best score {best:+.3f}. Best agent -> {args.out}")
    if not best_ok:
        print(f"  ⚠ NO checkpoint reached {args.min_served:.0%} served, so the saved one "
              f"is below the threshold.\n"
              f"    run_compare will mark its row NOT COMPARABLE. Train longer, or raise "
              f"--fail-mult\n"
              f"    if the failures are dead-ends / --max-hops if they are hop-limit "
              f"time-outs.")
    bench = f"python run_compare.py --drl {args.out}"
    if args.graph != "metr-la":                 # the benchmark must use the same graph
        bench += f" --graph {args.graph} --vehicles {args.eval_vehicles}"
        if args.capacity_scale is not None:
            bench += f" --capacity-scale {args.capacity_scale}"
    print(f"Benchmark it with:  {bench}")


if __name__ == "__main__":
    main()
