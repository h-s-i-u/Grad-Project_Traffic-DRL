#!/usr/bin/env python3
"""Metrics from a SUMO replay, defined the way the report defines them.

    python sumo_metrics.py sumo_a/s3/w154          # re-aggregate a finished directory

run_sumo_compare.py calls the functions below directly; the command line only re-reads
an existing tree of runs and prints the table again.

WHAT IS MEASURED, AND WHERE IT COMES FROM
    ATT           mean `duration` in tripinfo.xml over the trips that arrived: actual
                  departure to actual arrival, so the time a car spent waiting to be
                  inserted (`departDelay`) is NOT inside it. Reported separately.
    ATT (common)  the same, restricted to the trips that arrived under EVERY policy of
                  the seed. The report's headline ATT is scored the same way (log 13.16,
                  item 7), because a policy that abandons its hardest trips would
                  otherwise post a better mean for a reason that is not routing.
    served        trips that arrived / trips demanded. A trip that was routed but had
                  not arrived when the run ended counts as not served.
    worst-rho     edgeData with freq = the 154-second window that `capacity_scale`
                  implies (實驗設計 §4.4): `entered` vehicles per edge per interval over
                  the edge's capacity, maximum over every edge and interval. This is the
                  same quantity reroute_service.LoadWindow reports live.
    Gini          Gini of total `entered` per edge over the whole run, on the reference
                  set every policy of the seed used (union of edges with load), exactly
                  as metrics.evaluate does on the assignment. Edges only some policies
                  used count as zero for the others.
    frac_saturated share of arena edges whose PEAK interval rho exceeds RHO_THRESHOLD.

    Absolute values are NOT comparable with the BPR-side numbers in the report: the
    assignment model puts every vehicle on the road at once, SUMO spreads them over
    time. What is comparable is the paired delta against the herding baseline, seed by
    seed, which is what the table prints.

SUSTAINED DEMAND (run_sumo_compare.py --periods K)
    A static assignment describes a steady flow -- the same 800 trips every 154 s -- not
    one pulse of 800. With K periods the same routes are replayed K times, vehicle ids
    prefixed p<k>_, departures shifted by k x 154 s. Period 0 starts on an empty network
    and the last period drains into one, so `ATT mid` / `served mid` score the middle
    periods only (1 .. K-2); `ATT` scores every period. worst-rho and Gini are taken over
    the whole run, where the steady-state peak is anyway.
"""
import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

import config as C
import metrics as M
import network as net
from export_sumo import edge_id

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

BASELINE = "4_herding"
POLICY_ORDER = ["1_static", "4_herding", "6_oracle", "7_drl"]
LOAD_WINDOW_S = int(round(C.TAICHUNG_CAPACITY_SCALE * 3600))     # 154
_PERIOD = re.compile(r"^p(\d+)_")


def period_of(vid):
    """0 for a single-pulse replay; k for the k-th period of a sustained one."""
    m = _PERIOD.match(vid)
    return int(m.group(1)) if m else 0


def middle_periods(n_periods):
    """The periods scored as steady state: all of them below three, else 1 .. K-2."""
    return set(range(n_periods)) if n_periods < 3 else set(range(1, n_periods - 1))


# --------------------------------------------------------------------------- #
#  parsing
# --------------------------------------------------------------------------- #
def parse_tripinfo(path):
    """{vehicle id: {depart, arrival, duration, departDelay, timeLoss, routeLength}}.

    Written with --tripinfo-output.write-unfinished, so vehicles still driving when the
    run ended are present with arrival -1; they are kept so that `demanded` and
    `served` can be told apart, and dropped from every time average.
    """
    out = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "tripinfo":
            out[el.get("id")] = {
                "depart": float(el.get("depart")),
                "arrival": float(el.get("arrival", -1)),
                "duration": float(el.get("duration")),
                "departDelay": float(el.get("departDelay", 0)),
                "timeLoss": float(el.get("timeLoss", 0)),
                "routeLength": float(el.get("routeLength", 0)),
            }
            el.clear()
    return out


def parse_edgedata(path):
    """(per-interval [{edge: entered}], totals {edge: entered}) from an edgeData file."""
    intervals, totals = [], {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "interval":
            cur = {}
            for e in el.findall("edge"):
                n = float(e.get("entered", 0))
                if n > 0:
                    cur[e.get("id")] = n
                    totals[e.get("id")] = totals.get(e.get("id"), 0.0) + n
            intervals.append(cur)
            el.clear()
    return intervals, totals


def edge_capacities(g):
    """{edge id: capacity in vehicles per LOAD_WINDOW_S} from the same graph the routes
    were planned on."""
    return {edge_id(g, u, v): g.edges[(u, v)]["cap"] for u, v in g.edges()}


# --------------------------------------------------------------------------- #
#  one seed: every policy on the same demand
# --------------------------------------------------------------------------- #
def run_metrics(trips, intervals, totals, cap, ref_edges, common_ids, demanded,
                n_periods=1):
    arrived = {v: t for v, t in trips.items() if t["arrival"] >= 0}
    dur = np.array([t["duration"] for t in arrived.values()]) if arrived else np.array([0.0])
    common = [arrived[v]["duration"] for v in common_ids if v in arrived]
    mid = middle_periods(n_periods)
    arrived_mid = {v: t for v, t in arrived.items() if period_of(v) in mid}
    common_mid = [arrived[v]["duration"] for v in common_ids
                  if v in arrived and period_of(v) in mid]
    peak = {}
    for cur in intervals:
        for e, n in cur.items():
            if e in cap:
                peak[e] = max(peak.get(e, 0.0), n / cap[e])
    rho_vals = [peak.get(e, 0.0) for e in cap]
    att = float(dur.mean())
    demanded_all = demanded * n_periods
    demanded_mid = demanded * len(mid)
    return {
        "demanded": int(demanded_all),
        "periods": int(n_periods),
        "routed": len(trips),
        "served": len(arrived),
        "served_frac": len(arrived) / demanded_all if demanded_all else 0.0,
        "served_mid": len(arrived_mid) / demanded_mid if demanded_mid else 0.0,
        "att": att,
        "att_common": float(np.mean(common)) if common else 0.0,
        "n_common": len(common),
        "att_mid": float(np.mean([t["duration"] for t in arrived_mid.values()])) if arrived_mid else 0.0,
        "att_common_mid": float(np.mean(common_mid)) if common_mid else 0.0,
        "depart_delay": float(np.mean([t["departDelay"] for t in arrived.values()])) if arrived else 0.0,
        "time_loss": float(np.mean([t["timeLoss"] for t in arrived.values()])) if arrived else 0.0,
        "worst_rho": float(max(rho_vals)) if rho_vals else 0.0,
        "frac_saturated": float(np.mean([r > C.RHO_THRESHOLD for r in rho_vals])) if rho_vals else 0.0,
        "gini_load": M.gini([totals.get(e, 0.0) for e in ref_edges]),
        "throughput_proxy": float(len(arrived) / att) if att > 0 else 0.0,
    }


def seed_results(run_dir, policies, cap, demanded):
    """{policy: metrics} for one seed directory holding tripinfo_<p>.xml and
    edgedata_<p>.xml for every policy. Reference edge set and common trip set are
    built across the policies present, as run_compare does within a seed."""
    parsed = {}
    for p in policies:
        ti = os.path.join(run_dir, f"tripinfo_{p}.xml")
        ed = os.path.join(run_dir, f"edgedata_{p}.xml")
        if not (os.path.isfile(ti) and os.path.isfile(ed)):
            continue
        trips = parse_tripinfo(ti)
        intervals, totals = parse_edgedata(ed)
        parsed[p] = (trips, intervals, totals)
    if not parsed:
        return {}
    ref = sorted(set().union(*(set(t[2]) for t in parsed.values())) & set(cap))
    common = set.intersection(*({v for v, t in p[0].items() if t["arrival"] >= 0}
                                for p in parsed.values()))
    n_periods = 1 + max((period_of(v) for t in parsed.values() for v in t[0]), default=0)
    return {p: run_metrics(trips, intervals, totals, cap, ref, common, demanded, n_periods)
            for p, (trips, intervals, totals) in parsed.items()}


# --------------------------------------------------------------------------- #
#  across seeds
# --------------------------------------------------------------------------- #
def _mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


COLS = (("served", "served_frac"), ("ATT", "att"), ("ATT common", "att_common"),
        ("worst ρ", "worst_rho"), ("Gini", "gini_load"), ("dep. delay", "depart_delay"))
DELTA_COLS = (("ATT", "att"), ("ATT common", "att_common"), ("worst ρ", "worst_rho"),
              ("Gini", "gini_load"))
# Shown only for a sustained replay (periods >= 3), where the middle periods differ.
MID_COLS = (("served mid", "served_mid"), ("ATT mid", "att_mid"),
            ("ATT common mid", "att_common_mid"))
MID_DELTA_COLS = (("ATT mid", "att_mid"), ("ATT common mid", "att_common_mid"))


def aggregate(per_seed, baseline=BASELINE):
    """{policy: {metric: {mean, std}}} plus paired deltas vs the baseline."""
    names = [p for p in POLICY_ORDER if all(p in r for r in per_seed)]
    sustained = per_seed[0][names[0]].get("periods", 1) >= 3 if names else False
    delta_cols = DELTA_COLS + (MID_DELTA_COLS if sustained else ())
    out = {"n_seeds": len(per_seed), "sustained": sustained,
           "policies": {}, "delta_vs_baseline": {}}
    for n in names:
        out["policies"][n] = {}
        for key in per_seed[0][n]:
            m, s = _mean_std([r[n][key] for r in per_seed])
            out["policies"][n][key] = {"mean": m, "std": s}
        if n != baseline and baseline in names:
            out["delta_vs_baseline"][n] = {}
            for _, key in delta_cols:
                d = [100 * (r[n][key] - r[baseline][key]) / r[baseline][key]
                     if r[baseline][key] else 0.0 for r in per_seed]
                m, s = _mean_std(d)
                out["delta_vs_baseline"][n][key] = {"mean": m, "std": s}
    return out


def _cell(key, m, s):
    if key.startswith("served"):
        return f"{100 * m:8.1f}%±{100 * s:<7.1f}"
    if abs(m) >= 1000:
        return f"{m:9.1f}±{s:<7.1f}"
    return f"{m:9.4f}±{s:<7.4f}"


def fmt_table(per_seed, title, baseline=BASELINE):
    agg = aggregate(per_seed, baseline)
    names = list(agg["policies"])
    w = max(len(n) for n in names) if names else 8
    cols = COLS + (MID_COLS if agg["sustained"] else ())
    delta_cols = DELTA_COLS + (MID_DELTA_COLS if agg["sustained"] else ())
    periods = per_seed[0][names[0]].get("periods", 1) if names else 1
    head = f"{'policy':<{w}} | " + " | ".join(f"{lbl:>17}" for lbl, _ in cols)
    note = (f"({agg['n_seeds']} seeds; SUMO replay of the assignment; "
            f"rho = entries per {LOAD_WINDOW_S} s / capacity"
            + (f"; {periods} periods, 'mid' = periods {sorted(middle_periods(periods))}"
               if agg["sustained"] else "") + ")")
    lines = [title, note, head, "-" * len(head)]
    for n in names:
        cells = [_cell(key, agg["policies"][n][key]["mean"], agg["policies"][n][key]["std"])
                 for _, key in cols]
        lines.append(f"{n:<{w}} | " + " | ".join(cells))
    if agg["delta_vs_baseline"]:
        lines += ["", f"Δ vs '{baseline}' (paired per seed, mean ± std; negative = improvement):"]
        for n, d in agg["delta_vs_baseline"].items():
            parts = [f"{lbl} {d[key]['mean']:+6.1f}±{d[key]['std']:4.1f}%" for lbl, key in delta_cols]
            lines.append(f"  {n:<{w}} : " + " | ".join(parts))
    return "\n".join(lines), agg


def collect(tree_dir, policies=POLICY_ORDER, g=None):
    """Every seed*/ under `tree_dir` -> [per-seed {policy: metrics}], in seed order.

    The demanded count comes from the export's summary json in the seed directory
    (`vehicles`, after the S3 feasibility filter)."""
    g = g or net.build_graph_for("taichung", verbose=False)[0]
    cap = edge_capacities(g)
    per_seed, seeds = [], []
    for d in sorted(glob.glob(os.path.join(tree_dir, "seed*")),
                    key=lambda p: int(os.path.basename(p)[4:])):
        summ = glob.glob(os.path.join(d, "*_summary.json"))
        demanded = 0
        if summ:
            with open(summ[0], encoding="utf-8") as f:
                demanded = int(json.load(f).get("vehicles", 0))
        r = seed_results(d, policies, cap, demanded)
        if r:
            per_seed.append(r)
            seeds.append(int(os.path.basename(d)[4:]))
    return per_seed, seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree", help="directory holding seed*/ run directories")
    ap.add_argument("--policies", default=",".join(POLICY_ORDER))
    ap.add_argument("--json", default=None, help="write the aggregate here")
    cli = ap.parse_args()
    per_seed, seeds = collect(cli.tree, cli.policies.split(","))
    if not per_seed:
        raise SystemExit(f"no complete runs under {cli.tree}")
    text, agg = fmt_table(per_seed, f"{cli.tree}  seeds {seeds}")
    print(text)
    if cli.json:
        with open(cli.json, "w", encoding="utf-8") as f:
            json.dump({"seeds": seeds, "aggregate": agg, "per_seed": per_seed}, f,
                      indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
