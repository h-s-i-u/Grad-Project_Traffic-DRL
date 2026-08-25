#!/usr/bin/env python3
"""Compare routing strategies on the SAME predicted network and demand.

Reproduces the proposal's M4 benchmark idea (section 五): Dijkstra vs
prediction+static vs our method, scored by ATT, Gini of edge load, and
worst-link saturation — the metrics the proposal uses to quantify the
herding effect (羊群效應) and its mitigation.

    cd integration && python run_compare.py
    python run_compare.py --scenario random --vehicles 200
    python run_compare.py --graph taichung --vehicles 800 --drl checkpoints/taichung/drl_agent.pt
    # reporting protocol: 10 demand draws -> mean ± std with paired deltas
    python run_compare.py --repeat 10 --drl checkpoints/metr-la/drl_agent.pt
    # S3: shut an arterial halfway through the demand (實驗設計 §4.3)
    python run_compare.py --graph taichung --close-list          # what is worth closing
    python run_compare.py --graph taichung --vehicles 800 --repeat 10 \
                          --drl checkpoints/taichung/drl_agent_f10_800it.pt \
                          --close-road 臺灣大道 --close-at 0.5
"""
import argparse
import json
import os

import numpy as np

import closure as clo
import config as C
import metrics as M
import network as net
import policies as pol


def make_demand(g, scc, rng, pool=None):
    """Origin-destination pairs.

    'hotspot' funnels many origins into a few high-in-degree hub nodes (think
    rush-hour into the city centre / an arterial closure pushing everyone toward
    the same detour). That concentration is what triggers the herding effect.

    `pool` (used only by --close-demand resample) restricts both origins and hubs to
    a subset -- the component that survives an S3 closure. Every other closure mode
    leaves this function producing exactly the demand S2 produced, which is what
    lets the two scenarios be compared on the same vehicles.
    """
    scc = sorted(pool if pool is not None else scc)
    hubs = sorted(scc, key=lambda n: g.in_degree(n), reverse=True)[:C.N_HOTSPOTS]
    origins = rng.choice(scc, size=C.N_VEHICLES)
    if C.SCENARIO == "random":
        dests = rng.choice(scc, size=C.N_VEHICLES)
    else:  # hotspot
        dests = rng.choice(hubs, size=C.N_VEHICLES)
    demand = [(int(o), int(d)) for o, d in zip(origins, dests) if o != d]
    return demand, hubs


def _cell(key, value, width=11):
    """One table cell. served_frac is a fraction; everything else is a raw number."""
    if key == "served_frac":
        return f"{100 * value:{width - 1}.1f}%"
    return f"{value:{width}.4f}" if abs(value) < 1000 else f"{value:{width}.1f}"


def fmt(results, baseline_key):
    """Comparison table with % change vs the herding baseline.

    served% comes FIRST deliberately. A policy that abandons most of its vehicles
    scores beautifully on every other column — the trips that finish are the short
    easy ones, and the abandoned ones add no load at all — so ATT, worst ρ and
    saturated% are only comparable between rows that served the same demand. A
    20-iteration agent once posted ATT −74% and worst ρ −94.7% having delivered 11 of
    300 vehicles, and nothing in this table showed it; you had to multiply
    throughput × ATT by hand to find out. 實驗設計 §4.6 lists Served % as a required
    metric — it was computed by metrics.evaluate() all along, just never displayed.
    """
    cols = [
        ("served%", "served_frac", "high"),
        ("ATT", "att", "low"),
        ("TSTT", "tstt", "low"),
        ("worst ρ", "worst_rho", "low"),
        ("saturated%", "frac_saturated", "low"),
        ("Gini(load)", "gini_load", "low"),
        ("throughput*", "throughput_proxy", "high"),
    ]
    name_w = max(len(n) for n in results)
    head = f"{'policy':<{name_w}} | " + " | ".join(f"{c[0]:>11}" for c in cols)
    lines = [head, "-" * len(head)]
    base = results[baseline_key]
    for name, r in results.items():
        cells = [_cell(key, r[key]) for _, key, _ in cols]
        lines.append(f"{name:<{name_w}} | " + " | ".join(cells))

    low = {n for n, r in results.items() if r.get("served_frac", 1.0) < MIN_SERVED}
    # deltas vs baseline for the headline metrics
    lines.append("")
    lines.append(f"Δ vs '{baseline_key}' (negative = improvement):")
    for name, r in results.items():
        if name == baseline_key:
            continue
        d_att = 100 * (r["att"] - base["att"]) / base["att"] if base["att"] else 0
        d_gini = 100 * (r["gini_load"] - base["gini_load"]) / base["gini_load"] if base["gini_load"] else 0
        d_rho = 100 * (r["worst_rho"] - base["worst_rho"]) / base["worst_rho"] if base["worst_rho"] else 0
        flag = "   <- NOT COMPARABLE (see below)" if name in low else ""
        lines.append(f"  {name:<{name_w}} : ATT {d_att:+6.1f}% | "
                     f"Gini {d_gini:+6.1f}% | worst ρ {d_rho:+6.1f}%{flag}")
    if low:
        lines += ["", f"  ⚠ served below {MIN_SERVED:.0%}: {', '.join(sorted(low))}",
                  "    Unserved vehicles contribute no travel time and no edge load, so "
                  "that row's ATT,",
                  "    worst ρ and saturated% are flattered by exactly the trips it "
                  "failed to deliver.",
                  "    Fix the served rate before reading anything else in the row."]
    lines += congestion_note(base)
    return "\n".join(lines)


# The ablation ladder of 實驗設計 §4.5. (4) is the herding case — everyone follows the
# same prediction greedily — so every Δ is measured against it.
BASELINE = "4 hybrid + Dijkstra (HERDING)"

# columns shown in the multi-seed aggregate table. served% leads for the same reason
# it leads in fmt(): every other column is meaningless without it.
AGG_COLS = [("served%", "served_frac"), ("ATT", "att"), ("worst ρ", "worst_rho"),
            ("Gini(load)", "gini_load"), ("throughput*", "throughput_proxy")]

# 實驗設計 §4.6: "Served %：成功抵達比例（低於 95% 需說明）"
MIN_SERVED = 0.95

# The herding baseline's worst-link saturation has to land in a band where there is
# congestion to suppress. 實驗設計 §4.4 calibrates capacity_scale to put it at 2-3.
CONGESTION_BAND = (1.5, 5.0)


def congestion_note(base, name_w=0):
    """Warn when the scenario has no herding to suppress -- or is already gridlocked.

    Below ~1 no link is saturated, so every policy routes almost freely and (1)..(5)
    come out within noise of each other: measured at 300 vehicles on a capacity scale
    calibrated for 800, the baseline sat at worst-rho 0.95 and the deltas were
    meaningless. Above ~5 everything is jammed and the comparison saturates the other
    way. Neither case is a code error, which is why it needs saying out loud.
    """
    rho = base.get("worst_rho")
    if rho is None or CONGESTION_BAND[0] <= rho <= CONGESTION_BAND[1]:
        return []
    if rho < CONGESTION_BAND[0]:
        why = (f"nothing is saturated, so there is no herding to suppress and every "
               f"policy routes near-freely")
    else:
        why = "the network is gridlocked, so every policy is equally bad"
    return ["", f"  ⚠ herding baseline worst-rho {rho:.3f} is outside "
                f"{CONGESTION_BAND[0]}-{CONGESTION_BAND[1]}: {why}.",
            "    Re-run calibrate_taichung.py, or use the vehicle count it was "
            "calibrated for, before reading the deltas."]


def has_edge_attr(g, attr):
    """Is `attr` present on the graph's edges?

    networkx silently treats a missing edge attribute as weight=1, which would turn a
    shortest-path call into hop-count routing without any error — so the single-model
    baselines must be skipped rather than run on a graph that lacks their attribute.
    build_graph writes attributes uniformly, so checking one edge is enough.
    """
    for _, _, d in g.edges(data=True):
        return attr in d
    return False


def fmt_failures(infos, agent_label):
    """Break the agent's unserved trips into their three causes.

    A served% on its own cannot be acted on: a dead end means the policy trapped
    itself (every neighbour already visited), max-hops means it wandered, and a
    trivial trip was never routable. Under S3 the agent meets 83 edges that vanish
    from an action space it trained on, so this is the difference between "the
    closure broke it" and "it was already like that".
    """
    counts = [i["drl_failures"] for i in infos if i.get("drl_failures")]
    if not counts:
        return ""
    keys = ("dead_end", "max_hops", "trivial")

    def line(label, dicts):
        mean = {k: np.mean([c.get(k, 0) for c in dicts]) for k in keys}
        if not any(mean.values()):
            return f"\n  {label}: no failed trips."
        return (f"\n  {label} unserved (mean over {len(dicts)} seed(s)): "
                + ", ".join(f"{k.replace('_', '-')} {mean[k]:.1f}" for k in keys))

    out = line(agent_label, counts)
    for w in sorted({int(k[4:]) for c in counts for k in c if k.startswith("beam")}):
        out += line(f"{agent_label} beam-{w}",
                    [c.get(f"beam{w}", {}) for c in counts])
    return out


def run_once(g, scc, seed, max_hops, agent, agent_label, togo_refresh=0,
             closure=None, close_demand="filter", pool=None, beams=()):
    """One demand sample -> ({policy: metrics}, demand, hubs, info).

    Everything downstream of the RNG lives here, so repeating with different seeds
    gives independent demand draws on the SAME graph and agent.
    """
    rng = np.random.default_rng(seed)
    demand, hubs = make_demand(g, scc, rng, pool=pool)
    n_drawn = len(demand)
    demand, dem_info = clo.select_demand(g, closure, demand, close_demand)

    runs = {"1 static (free-flow)": pol.policy_static(g, demand, closure)}
    # (2)(3) route on a single model's own prediction; only available when the graph
    # carries per-model tpred (METR-LA). Taichung has no per-model speeds yet.
    if has_edge_attr(g, "tpred_stgcn"):
        runs["2 STGCN + Dijkstra"] = pol.policy_prediction_greedy(
            g, demand, "tpred_stgcn", closure)
    if has_edge_attr(g, "tpred_stgat"):
        runs["3 STGAT + Dijkstra"] = pol.policy_prediction_greedy(
            g, demand, "tpred_stgat", closure)
    runs[BASELINE] = pol.policy_prediction_greedy(g, demand, closure=closure)
    runs["5 load-aware (coord.)"] = pol.policy_load_aware(g, demand, closure)
    runs["6 global-penalty (oracle)"] = pol.policy_global_penalty(g, demand, closure)
    drl_stats = {}
    if agent is not None:
        # stats: "not served" has three causes needing three different fixes, and
        # under S3 the agent meets a topology it never trained on -- so a drop in
        # served% has to be readable as dead-end vs wandering, not just a number.
        runs[agent_label] = pol.policy_drl(g, demand, agent, max_hops=max_hops,
                                           togo_refresh=togo_refresh,
                                           closure=closure, stats=drl_stats)
        # Same weights, wider decoding -- one row per width, as Lei et al. 2022 report
        # G and BS side by side. Greedy stays in the table: the pair is the result.
        for w in beams:
            drl_stats[f"beam{w}"] = st = {}
            runs[f"{agent_label} beam-{w}"] = pol.policy_drl(
                g, demand, agent, max_hops=max_hops, togo_refresh=togo_refresh,
                closure=closure, stats=st, beam=w)

    # fixed reference edge set = union of links used by any policy (fair Gini)
    ref = set()
    for paths in runs.values():
        load, _ = M.edge_loads(g, paths)
        ref |= {e for e, v in load.items() if v > 0}
    ref = sorted(ref)

    out = {name: M.evaluate(g, paths, ref) for name, paths in runs.items()}
    # Fraction, not the raw count: demand size varies by a few vehicles between seeds
    # (o == d pairs are dropped), so the aggregate has to average fractions.
    for r in out.values():
        r["served_frac"] = r["served"] / len(demand) if demand else 0.0
    info = {"drawn": n_drawn, "routed": len(demand), "demand": dem_info,
            "drl_failures": dict(drl_stats)}
    return out, demand, hubs, info


def _mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


def aggregate(all_results):
    """{policy: {metric: {'mean': m, 'std': s}}} across seeds."""
    out = {}
    for name in all_results[0]:
        out[name] = {}
        for key in all_results[0][name]:
            m, s = _mean_std([r[name][key] for r in all_results])
            out[name][key] = {"mean": m, "std": s}
    return out


def fmt_aggregate(all_results, baseline_key, seeds):
    """mean ± std across seeds, plus PAIRED deltas vs the herding baseline.

    Deltas are computed WITHIN each seed (all policies saw the same demand) and then
    averaged. This paired comparison is tighter than comparing the means, because it
    cancels the demand-to-demand variance that affects every policy alike.
    """
    names = list(all_results[0])
    name_w = max(len(n) for n in names)
    head = f"{'policy':<{name_w}} | " + " | ".join(f"{lbl:>17}" for lbl, _ in AGG_COLS)
    lines = [f"Aggregate over {len(all_results)} seeds {seeds}:", head, "-" * len(head)]
    low = set()
    for n in names:
        cells = []
        for _, key in AGG_COLS:
            m, s = _mean_std([r[n][key] for r in all_results])
            if key == "served_frac":
                cells.append(f"{100 * m:8.1f}%±{100 * s:<7.1f}")
                if m < MIN_SERVED:
                    low.add(n)
            else:
                cells.append(f"{m:9.4f}±{s:<7.4f}" if abs(m) < 1000
                             else f"{m:9.1f}±{s:<7.1f}")
        lines.append(f"{n:<{name_w}} | " + " | ".join(cells))

    lines += ["", f"Δ vs '{baseline_key}' (paired per seed, mean ± std; negative = improvement):"]
    for n in names:
        if n == baseline_key:
            continue
        parts = []
        for lbl, key in (("ATT", "att"), ("Gini", "gini_load"), ("worst ρ", "worst_rho")):
            d = [100 * (r[n][key] - r[baseline_key][key]) / r[baseline_key][key]
                 if r[baseline_key][key] else 0.0 for r in all_results]
            m, s = _mean_std(d)
            parts.append(f"{lbl} {m:+6.1f}±{s:4.1f}%")
        flag = "   <- NOT COMPARABLE" if n in low else ""
        lines.append(f"  {n:<{name_w}} : " + " | ".join(parts) + flag)
    if low:
        lines += ["", f"  ⚠ served below {MIN_SERVED:.0%}: {', '.join(sorted(low))} — "
                      f"unserved vehicles add no",
                  "    travel time and no load, so those rows' ATT / worst ρ are "
                  "flattered by their own failures."]
    # Averaged over seeds: the scenario is either congested or not, so one check is
    # enough rather than one per seed.
    lines += congestion_note(
        {"worst_rho": _mean_std([r[baseline_key]["worst_rho"] for r in all_results])[0]})
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", choices=net.GRAPHS, default="metr-la",
                    help="road network: 'metr-la' (sensor kernel + STGCN/STGAT ensemble) "
                         "or 'taichung' (real OSM network, times in seconds)")
    ap.add_argument("--capacity-scale", type=float, default=None,
                    help="taichung only: scale the CSV veh/h capacity into the load range "
                         f"the reward was tuned on (default {C.TAICHUNG_CAPACITY_SCALE})")
    ap.add_argument("--max-hops", type=int, default=None,
                    help="hop budget per trip for the DRL rollout (default: 60 for "
                         f"metr-la, {C.TAICHUNG_MAX_HOPS} for taichung)")
    ap.add_argument("--scenario", choices=["random", "hotspot"], default=C.SCENARIO)
    ap.add_argument("--vehicles", type=int, default=C.N_VEHICLES)
    ap.add_argument("--capacity", type=float, default=C.EDGE_CAPACITY)
    ap.add_argument("--seed", type=int, default=C.SEED,
                    help=f"base RNG seed for demand sampling (default {C.SEED})")
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                    help="draw N demands (seed, seed+1, ... seed+N-1) and report "
                         "mean ± std with paired deltas — the reporting protocol for "
                         "results, since a single demand varies too much to conclude from")
    ap.add_argument("--drl", default=None, metavar="placeholder|CKPT.pt",
                    help="add the DRL agent as a 5th policy: 'placeholder' (analytic "
                         "stand-in, no training needed) or a path to a trained checkpoint")
    ap.add_argument("--beam", default="", metavar="W[,W...]",
                    help="score the SAME agent with beam-search decoding as well, one "
                         "extra row per width (e.g. --beam 8). Greedy always stays in "
                         "the table -- the pair is the ablation. Costs about one extra "
                         "rollout per width per seed")
    s3 = ap.add_argument_group(
        "S3 arterial closure (實驗設計 §4.3)",
        "Shut a road partway through the demand, so vehicles dispatched later are all "
        "handed the same new fastest route at once. Omit these and the run is "
        "identical to before.")
    s3.add_argument("--close-road", metavar="PREFIX", default=None,
                    help="close every edge whose road_name starts with PREFIX. Prefix, "
                         "so '臺灣大道' closes the whole corridor (83 edges) and "
                         "'臺灣大道三段' only that segment (20). taichung only.")
    s3.add_argument("--close-edge", metavar="FROM,TO", default=None,
                    help="close one edge, given in original OSM node ids")
    s3.add_argument("--close-busiest", action="store_true",
                    help="close the heaviest road that still leaves the hotspot hubs "
                         "reachable (the single busiest one usually does not)")
    s3.add_argument("--close-at", type=float, default=0.5, metavar="FRAC",
                    help="fraction of the demand dispatched before the closure "
                         "(default 0.5; 0 = closed from the start)")
    s3.add_argument("--close-demand", choices=["filter", "wave", "resample", "none"],
                    default="filter",
                    help="what to do with trips that have no route after the closure "
                         "(default filter: drop them, ~3.8%%, and say so)")
    s3.add_argument("--close-list", action="store_true",
                    help="print the per-road impact table and exit, without running "
                         "the comparison")
    args = ap.parse_args()
    C.SCENARIO, C.N_VEHICLES, C.EDGE_CAPACITY = args.scenario, args.vehicles, args.capacity
    max_hops = args.max_hops or net.default_max_hops(args.graph)
    seeds = [args.seed + i for i in range(max(1, args.repeat))]

    g, info = net.build_graph_for(args.graph, capacity_scale=args.capacity_scale)
    scc = net.largest_scc(g)

    print(f"Graph '{args.graph}': {g.number_of_nodes()} nodes, {g.number_of_edges()} "
          f"directed edges; largest SCC = {len(scc)} nodes")
    if args.graph == "metr-la":
        lo, hi = info["speed_range"]
        print(f"Ensemble speed {lo:.1f}~{hi:.1f} mph "
              f"(STGCN {info['stgcn_range'][0]:.0f}-{info['stgcn_range'][1]:.0f}, "
              f"STGAT {info['stgat_range'][0]:.0f}-{info['stgat_range'][1]:.0f})")
        if info["stgat_out_of_range"]:
            print("  ⚠ STGAT predictions exceed the physical speed range and were clamped "
                  f"to {C.SPEED_MAX} mph (prediction-module issue — see README).")
    else:
        print(f"Edge times in seconds (speed limits); capacity_scale="
              f"{info['capacity_scale']}, max_hops={max_hops}")
        # Report the coverage actually present on the graph rather than assuming there
        # is none: this branch used to print the "no live-speed source" warning
        # unconditionally, contradicting the same run's policy table, which sends you
        # debugging a wiring problem that does not exist.
        n_edges = g.number_of_edges()
        # `observed` is set by the loader. Do not infer it from tpred != t0: with
        # tpred_fallback="network_mean" every edge differs from t0.
        covered = sum(1 for _, _, d in g.edges(data=True) if d.get("observed"))
        if not covered:
            print("  ⚠ No live-speed source: tpred = t0 everywhere, so policy (4) is "
                  "identical to (1) and the single-model baselines (2)(3) are skipped. "
                  "Run make_drl_input.py to write taichung_pred_edges.csv.")
        else:
            fb = info.get("tpred_fallback", "free_flow")
            rest = ("keep their free-flow time" if fb == "free_flow" else
                    "are scaled to the mean observed slowdown")
            print(f"  {covered:,} of {n_edges:,} edges carry a prediction "
                  f"({covered / n_edges:.1%}); the rest {rest} "
                  f"(tpred_fallback={fb}).")
            if fb == "free_flow":
                print("  ⚠ 'free_flow' assumes every unobserved road runs at the speed "
                      "limit while measured roads report about half that, so (2)(3)(4) "
                      "route away from the instrumented arterials. Compare against "
                      "'network_mean' before reading their deltas.")
            if covered / n_edges < 0.05:
                print("  ⚠ Below ~5% coverage the Dijkstra-on-tpred baselines (2)(3)(4) "
                      "converge on (1) — (4) is the denominator of every delta, so the "
                      "herding comparison loses its meaning. Shrink the routing graph "
                      "with TDX_Data/build_arena.py rather than reading these numbers.")

    # --- S3 arterial closure (實驗設計 §4.3) ---------------------------------
    # Built from the FIRST seed's demand and kept fixed across seeds: the incident is
    # a property of the scenario, not of a demand draw. Letting --close-busiest pick a
    # different road per seed would make the 10 runs unpairable.
    closure = close_pool = close_info = None
    if args.close_road or args.close_edge or args.close_busiest or args.close_list:
        probe, hubs0 = make_demand(g, scc, np.random.default_rng(seeds[0]))
        load, total = clo.baseline_load(g, probe, pol.policy_prediction_greedy)
        try:
            if args.close_list:
                print(f"\nPer-road impact, herding baseline on the OPEN network "
                      f"(seed {seeds[0]}, {len(probe)} vehicles):\n")
                print(clo.fmt_survey(clo.survey(g, probe, load, total, hubs0)))
                return
            if args.close_busiest:
                label = clo.busiest_road(g, probe, load, total, hubs0)
                edges = clo.edges_by_road(g, label)
            elif args.close_road:
                label, edges = args.close_road, clo.edges_by_road(g, args.close_road)
            else:
                label, edges = args.close_edge, clo.edges_by_endpoints(g, args.close_edge)
            closure = clo.Closure(edges, at=args.close_at, label=label)
        except ValueError as e:
            raise SystemExit(f"error: {e}")
        close_info = clo.inspect(g, closure, probe, load, total, hubs0)
        print()
        for line in clo.fmt_inspect(close_info):
            print(line)
        if args.close_demand == "resample":
            close_pool = clo.post_closure_pool(g, closure)
            print(f"[S3] origins restricted to the {len(close_pool)} nodes that survive "
                  f"the closure (--close-demand resample); this demand is NOT the "
                  f"demand S2 used")

    # DRL agent slot (proposal §4.4). Off by default; opt in with --drl.
    # Built once and reused across seeds: the agent is fixed, only the demand varies.
    agent = agent_label = None
    togo_refresh = 0
    if args.drl:
        agent = pol.make_drl_agent(args.drl, g)
        # Read the settings the checkpoint was trained under. togo_refresh changes the
        # observation the policy reads, so evaluating with a different value degrades
        # the agent silently rather than raising.
        meta_path = os.path.splitext(args.drl)[0] + ".meta.json"
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            togo_refresh = int(meta.get("togo_refresh", 0) or 0)
            print(f"  agent trained at iter {meta.get('iteration')} "
                  f"(served {meta.get('served', float('nan')):.1%}, "
                  f"togo_refresh={togo_refresh}, "
                  f"capacity_scale={meta.get('capacity_scale')})")
            if meta.get("graph") and meta["graph"] != args.graph:
                print(f"  ⚠ the checkpoint was trained on '{meta['graph']}' but this run "
                      f"is '{args.graph}'; its score does not carry over.")
        elif args.drl not in ("placeholder", "oracle"):
            print(f"  ⚠ no {os.path.basename(meta_path)} beside the checkpoint -- "
                  f"assuming togo_refresh=0. If it was trained with a non-zero value "
                  f"the agent is being evaluated on observations it never saw.")
        agent_label = ("7 drl-agent (placeholder)" if args.drl in ("placeholder", "oracle")
                       # basename only: a full checkpoint path widens every row of the
                       # table by ~30 characters and the filename already identifies it
                       else f"7 drl-agent ({os.path.basename(args.drl)})")
    else:
        print("(DRL slot scaffolded but inactive — add it with "
              "`--drl placeholder` or `--drl path/to/checkpoint.pt`.)")

    beams = []
    if args.beam:
        try:
            beams = sorted({int(w) for w in args.beam.split(",")})
        except ValueError:
            raise SystemExit(f"error: --beam wants widths, got {args.beam!r}")
        if any(w < 2 for w in beams):
            raise SystemExit("error: --beam widths must be >= 2; width 1 IS greedy "
                             "decoding and is already the row above")
        if agent is None:
            raise SystemExit("error: --beam needs --drl; it decodes an agent's policy")
        print(f"  beam-search decoding at width {beams} as extra rows (same weights, "
              f"same observations -- only the decoding differs)")

    print(f"Scenario '{C.SCENARIO}': ~{C.N_VEHICLES} vehicles, "
          f"capacity={C.EDGE_CAPACITY}/edge, seeds={seeds}\n")

    all_results, hubs, n_veh, infos = [], None, None, []
    for s in seeds:
        results, demand, hubs, info = run_once(g, scc, s, max_hops, agent, agent_label,
                                               togo_refresh, closure, args.close_demand,
                                               close_pool, beams)
        all_results.append(results)
        infos.append(info)
        n_veh = len(demand)
        if closure is not None and s == seeds[0]:
            for line in clo.fmt_demand(info["demand"], info["drawn"]):
                print(line)
            print()
        if len(seeds) > 1:
            base, ours = results[BASELINE], results["6 global-penalty (oracle)"]
            d = 100 * (ours["gini_load"] - base["gini_load"]) / base["gini_load"]
            drop = (f" | {info['demand']['dropped']} infeasible"
                    if info["demand"].get("dropped") else "")
            print(f"  seed {s:>4}: {n_veh} vehicles{drop} | oracle Gini Δ {d:+6.1f}%")

    if len(seeds) == 1:
        print(fmt(all_results[0], baseline_key=BASELINE))
    else:
        print()
        print(fmt_aggregate(all_results, BASELINE, seeds))
    print("\n  * throughput proxy = served vehicles / ATT (static-assignment proxy, not SUMO).")
    print(fmt_failures(infos, agent_label))

    out = {
        "graph": args.graph,
        "scenario": "S3" if closure is not None else C.SCENARIO,
        "closure": (None if close_info is None else
                    {**close_info, "demand_mode": args.close_demand,
                     "dropped_per_seed": [i["demand"].get("dropped", 0) for i in infos],
                     "drl_failures": [i["drl_failures"] for i in infos]}),
        "seeds": seeds,
        "n_vehicles": n_veh,
        "capacity": C.EDGE_CAPACITY if args.graph == "metr-la" else "per-edge (CSV)",
        "capacity_scale": info.get("capacity_scale"),
        "max_hops": max_hops,
        "hubs": hubs,
        "stgat_out_of_range": info.get("stgat_out_of_range"),
        "results": all_results[0],                                  # first (or only) seed
        "runs": {str(s): r for s, r in zip(seeds, all_results)},    # per-seed detail
        "aggregate": aggregate(all_results) if len(seeds) > 1 else None,
    }
    # S3 goes to its own file: the two scenarios are run alternately and a shared
    # name would silently overwrite the S2 table that §13.15 is built from.
    path = C.HERE / ("results_s3.json" if closure is not None else "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
