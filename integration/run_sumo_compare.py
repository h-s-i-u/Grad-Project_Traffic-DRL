#!/usr/bin/env python3
"""Replay validation (mode A): the report's scenarios, decided under BPR, driven in SUMO.

    cd integration
    python export_sumo.py --drl checkpoints/taichung/drl_fusion_togo25.pt   # once, then
    (cd sumo && sh build_net.sh)                                            # the checked net
    python run_sumo_compare.py --drl checkpoints/taichung/drl_fusion_togo25.pt
    python run_sumo_compare.py --drl ... --dry-run          # just list the jobs

WHAT THIS TESTS
    實驗設計 §9.2 ③: on the BPR arena the analytic oracle IS the cost function's own
    optimiser, so policy 7 cannot be argued on quality there. SUMO is a world whose cost
    function is not the one any policy was built on. Routes are assigned exactly as
    run_compare does -- same demand generator, same seeds, same S3 closure, beam-8 for
    policy 7 -- and SUMO replays each policy's assignment and measures. It is a
    transfer test of the assignment, not training in SUMO (that would be mode B).

THE ONE MAPPING THAT IS A CHOICE
    The assignment has no clock; `capacity_scale` implies one (實驗設計 §4.4): capacity is
    vehicles per 154 s, so an 800-vehicle assignment is one 154-second demand period.
    --windows 154 spreads the departures over exactly that; 600 is the exporter's gentler
    default, kept as the sensitivity case. Routes do not depend on the window, so each
    seed is exported once and the other windows rescale the departure times.

SUSTAINED DEMAND (--periods K)
    A static assignment describes a steady flow, not a pulse: the same 800 trips every
    154 s. A single pulse of 800 disperses before it reaches the busy edges, and the
    first replay (log 13.30) measured the busiest edge at rho 0.84 per window against
    the assignment's 3.33. With --periods K the SAME routes are replayed K times --
    vehicle ids prefixed p<k>_, departures shifted by k x W -- so nothing about the
    routing changes and only the world's saturation does. S2 only: the S3 closure is a
    one-off event inside the dispatch, and replaying it means closing the road again
    every period. sumo_metrics scores the middle periods separately.

LAYOUT
    <out>/<scenario>/seed<S>/               export: routes, summary, load json
    <out>/<scenario>/seed<S>/w<W>[p<K>]/    per window (and period count): routes with
                                            departures over W s, tripinfo_<policy>.xml,
                                            edgedata_<policy>.xml
    <out>/<scenario>_w<W>[_p<K>].txt/.json  the table and its numbers
    <out>/results.json                      everything

Resumable: an export with its summary present, and a run with a complete tripinfo, are
skipped. Delete the file to redo it.
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import config as C
import network as net
import sumo_metrics as SM

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# The report's S3 (實驗設計 §4.3): the whole corridor, after 10% of the demand is out.
S3_ROAD, S3_AT = "臺灣大道", 0.10
EXPORT_WINDOW = 154.0            # the window the routes are exported at; others rescale


def sumo_binary(name="sumo"):
    found = shutil.which(name)
    if found:
        return found
    try:
        from sumolib import checkBinary
        return checkBinary(name)
    except Exception as exc:
        raise SystemExit(f"cannot find {name}: {exc}\n  pip install eclipse-sumo traci sumolib")


# --------------------------------------------------------------------------- #
#  export (one per scenario x seed)
# --------------------------------------------------------------------------- #
def export_seed(cli, scen, seed, out_dir):
    """Run export_sumo.py for one seed; skipped if its summary already exists."""
    tag = scen
    summary = os.path.join(out_dir, f"{tag}_summary.json")
    if os.path.isfile(summary):
        return "skip"
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "export_sumo.py"),
           "--out-dir", out_dir, "--tag", tag, "--seed", str(seed),
           "--vehicles", str(cli.vehicles), "--window", str(EXPORT_WINDOW),
           "--policies", cli.policies, "--beam", str(cli.beam), "--shapes", "none"]
    if cli.drl:
        cmd += ["--drl", cli.drl]
    if scen == "s3":
        cmd += ["--close-road", cli.close_road, "--close-at", str(cli.close_at)]
    log = os.path.join(out_dir, "export.log")
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not os.path.isfile(summary):
        raise SystemExit(f"export failed for {scen} seed {seed}; see {log}")
    return "done"


def rescale_departs(src, dst, factor):
    """The same routes with every departure time multiplied by `factor`."""
    tree = ET.parse(src)
    for veh in tree.getroot().iter("vehicle"):
        veh.set("depart", f"{float(veh.get('depart')) * factor:.2f}")
    tree.write(dst, encoding="utf-8", xml_declaration=True)


def build_route_file(src, dst, factor, periods, period_s):
    """Departures rescaled by `factor`, then the whole fleet repeated `periods` times.

    Period k's copy of vehicle v is p<k>_v, departing period_s * k later. Routes are
    copied verbatim: the point of a sustained replay is that only the world changes.
    """
    tree = ET.parse(src)
    root = tree.getroot()
    vehicles = list(root.iter("vehicle"))
    for veh in vehicles:
        veh.set("depart", f"{float(veh.get('depart')) * factor:.2f}")
    if periods > 1:
        for veh in vehicles:
            root.remove(veh)
        for k in range(periods):
            for veh in vehicles:
                copy = ET.fromstring(ET.tostring(veh))
                copy.set("id", f"p{k}_{veh.get('id')}")
                copy.set("depart", f"{float(veh.get('depart')) + k * period_s:.2f}")
                root.append(copy)
    tree.write(dst, encoding="utf-8", xml_declaration=True)


def policy_names(policies):
    lut = {"1": "1_static", "4": "4_herding", "6": "6_oracle", "7": "7_drl"}
    return [lut[p.strip()] for p in policies.split(",")]


# --------------------------------------------------------------------------- #
#  one SUMO run
# --------------------------------------------------------------------------- #
def run_complete(tripinfo):
    if not os.path.isfile(tripinfo):
        return False
    with open(tripinfo, "rb") as f:
        f.seek(max(0, os.path.getsize(tripinfo) - 200))
        return b"</tripinfos>" in f.read()


def sumo_job(binary, net_path, run_dir, rou, pol, seed, end):
    tripinfo = os.path.join(run_dir, f"tripinfo_{pol}.xml")
    if run_complete(tripinfo):
        return pol, "skip", 0.0
    add = os.path.join(run_dir, f"edgedata_{pol}.add.xml")
    with open(add, "w", encoding="utf-8") as f:
        f.write(f'<additional>\n  <edgeData id="w{SM.LOAD_WINDOW_S}" '
                f'file="edgedata_{pol}.xml" freq="{SM.LOAD_WINDOW_S}" '
                f'excludeEmpty="true"/>\n</additional>\n')
    cmd = [binary, "-n", net_path, "-r", os.path.basename(rou),
           "--additional-files", os.path.basename(add),
           "--tripinfo-output", os.path.basename(tripinfo),
           "--tripinfo-output.write-unfinished", "true",
           "--time-to-teleport", "-1", "--end", str(end), "--seed", str(seed),
           "--no-step-log", "true", "--no-warnings", "true",
           "--duration-log.disable", "true",
           "--log", f"sumo_{pol}.log"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=run_dir, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not run_complete(tripinfo):
        return pol, f"FAILED (see {os.path.join(run_dir, f'sumo_{pol}.log')})", time.time() - t0
    return pol, "done", time.time() - t0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drl", default=None, metavar="CKPT.pt", help="needed for policy 7")
    ap.add_argument("--policies", default="1,4,6,7")
    ap.add_argument("--beam", type=int, default=8,
                    help="decoding width for policy 7; the report's quality row is beam-8")
    ap.add_argument("--seeds", type=int, default=10,
                    help="C.SEED, C.SEED+1, ... -- run_compare --repeat does the same")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--vehicles", type=int, default=800,
                    help="capacity_scale was calibrated at 800")
    ap.add_argument("--windows", default="154,600",
                    help="seconds the departures are spread over; the first is the "
                         "one the routes are exported at (154 = the capacity window)")
    ap.add_argument("--scenarios", default="s2,s3")
    ap.add_argument("--periods", type=int, default=1, metavar="K",
                    help="replay the same routes K times, each period one window later "
                         "(module docstring: SUSTAINED DEMAND). S2 only; 5 scores three "
                         "middle periods")
    ap.add_argument("--close-road", default=S3_ROAD)
    ap.add_argument("--close-at", type=float, default=S3_AT)
    ap.add_argument("--net", default=os.path.join(HERE, "sumo", "taichung.net.xml"),
                    help="the network build_net.sh produced (with its turn check)")
    ap.add_argument("--out", default=os.path.join(HERE, "sumo_a"))
    ap.add_argument("--jobs", type=int, default=4, help="SUMO runs in parallel")
    ap.add_argument("--end", type=int, default=7200,
                    help="simulation cap in seconds; trips not arrived by then are unserved")
    ap.add_argument("--dry-run", action="store_true", help="list the work, do nothing")
    cli = ap.parse_args()

    if "7" in cli.policies.split(",") and not cli.drl:
        ap.error("policy 7 needs --drl")
    if not os.path.isfile(cli.net):
        raise SystemExit(f"{cli.net} is missing: run export_sumo.py, then sh build_net.sh "
                         f"in sumo/ -- and read its turn check")
    windows = [float(w) for w in cli.windows.split(",")]
    scenarios = [s.strip() for s in cli.scenarios.split(",")]
    seeds = [cli.seed + i for i in range(cli.seeds)]
    pols = policy_names(cli.policies)
    net_path = os.path.abspath(cli.net)
    K = max(1, cli.periods)
    if K > 1 and any(s != "s2" for s in scenarios):
        raise SystemExit("--periods applies to S2 only: the S3 closure is a one-off event "
                         "inside the dispatch, and replaying it closes the road again every "
                         "period. Pass --scenarios s2.")
    suffix = f"p{K}" if K > 1 else ""
    os.makedirs(cli.out, exist_ok=True)

    print(f"\n{'=' * 88}\nSUMO replay validation -- {', '.join(scenarios)} x windows "
          f"{cli.windows} x {len(seeds)} seeds x {', '.join(pols)}"
          + (f" x {K} periods" if K > 1 else "") + f"\n{'=' * 88}")
    print(f"network : {net_path}\nroutes  : {cli.vehicles} vehicles, beam-{cli.beam} for "
          f"policy 7, S3 = {cli.close_road} at {cli.close_at:.0%}\nout     : {cli.out}")
    n_runs = len(scenarios) * len(seeds) * len(windows) * len(pols)
    print(f"jobs    : {len(scenarios) * len(seeds)} exports, {n_runs} SUMO runs, "
          f"{cli.jobs} in parallel")
    if cli.dry_run:
        for scen in scenarios:
            for seed in seeds:
                print(f"  export {scen} seed {seed} -> {os.path.join(cli.out, scen, f'seed{seed}')}")
                for w in windows:
                    for p in pols:
                        print(f"    sumo  w{w:g}{suffix} {p}")
        return

    # --- exports: sequential, they share the GPU ------------------------------
    t_all = time.time()
    for scen in scenarios:
        for seed in seeds:
            d = os.path.join(cli.out, scen, f"seed{seed}")
            t0 = time.time()
            st = export_seed(cli, scen, seed, d)
            print(f"export {scen} seed {seed}: {st}" +
                  (f" ({time.time() - t0:.0f} s)" if st == "done" else ""), flush=True)
            # every window gets its own route files: the export's routes, departures
            # rescaled to the window, repeated K times for a sustained replay
            for w in windows:
                wd = os.path.join(d, f"w{w:g}{suffix}")
                os.makedirs(wd, exist_ok=True)
                for p in pols:
                    src = os.path.join(d, f"{scen}_{p}.rou.xml")
                    dst = os.path.join(wd, f"{scen}_{p}.rou.xml")
                    if not os.path.isfile(src):
                        raise SystemExit(f"missing {src}: the export did not write policy {p}")
                    if not os.path.isfile(dst):
                        if w == EXPORT_WINDOW and K == 1:
                            shutil.copyfile(src, dst)
                        else:
                            build_route_file(src, dst, w / EXPORT_WINDOW, K, w)

    # --- SUMO runs: parallel -----------------------------------------------------
    binary = sumo_binary("sumo")
    jobs = []
    for scen in scenarios:
        for seed in seeds:
            for w in windows:
                wd = os.path.join(cli.out, scen, f"seed{seed}", f"w{w:g}{suffix}")
                for p in pols:
                    jobs.append((scen, seed, w, p, wd,
                                 os.path.join(wd, f"{scen}_{p}.rou.xml")))
    print(f"\n{len(jobs)} SUMO runs ...", flush=True)
    done = skipped = failed = 0
    with cf.ThreadPoolExecutor(max_workers=cli.jobs) as ex:
        futs = {ex.submit(sumo_job, binary, net_path, wd, rou, p, seed, cli.end):
                (scen, seed, w, p) for scen, seed, w, p, wd, rou in jobs}
        for fut in cf.as_completed(futs):
            scen, seed, w, p = futs[fut]
            pol, st, secs = fut.result()
            if st == "skip":
                skipped += 1
            elif st == "done":
                done += 1
                print(f"  {scen} seed {seed} w{w:g} {p:<10} {secs:5.0f} s", flush=True)
            else:
                failed += 1
                print(f"  {scen} seed {seed} w{w:g} {p:<10} {st}", flush=True)
    print(f"runs: {done} done, {skipped} skipped, {failed} failed "
          f"({(time.time() - t_all) / 60:.1f} min in total)")

    # --- metrics -------------------------------------------------------------------
    g, _ = net.build_graph_for("taichung", verbose=False)
    results = {"vehicles": cli.vehicles, "beam": cli.beam, "seeds": seeds,
               "close_road": cli.close_road, "close_at": cli.close_at,
               "periods": K, "load_window_s": SM.LOAD_WINDOW_S, "tables": {}}
    results_path = os.path.join(cli.out, f"results{('_' + suffix) if suffix else ''}.json")
    for scen in scenarios:
        for w in windows:
            # collect() walks seed*/ directories; the window lives one level below, so
            # build the per-seed list here
            cap = SM.edge_capacities(g)
            per_seed, used = [], []
            for seed in seeds:
                d = os.path.join(cli.out, scen, f"seed{seed}")
                summ = os.path.join(d, f"{scen}_summary.json")
                demanded = 0
                if os.path.isfile(summ):
                    with open(summ, encoding="utf-8") as f:
                        demanded = int(json.load(f).get("vehicles", 0))
                r = SM.seed_results(os.path.join(d, f"w{w:g}{suffix}"), pols, cap, demanded)
                if r and all(p in r for p in pols):
                    per_seed.append(r)
                    used.append(seed)
            key = f"{scen}_w{w:g}" + (f"_{suffix}" if suffix else "")
            if not per_seed:
                print(f"\n{key}: no complete seed")
                continue
            title = (f"{scen.upper()} -- departures over {w:g} s"
                     + (f", {cli.close_road} closed at {cli.close_at:.0%}" if scen == "s3" else "")
                     + (f", the same fleet every {w:g} s for {K} periods" if K > 1 else ""))
            text, agg = SM.fmt_table(per_seed, title)
            print("\n" + text)
            with open(os.path.join(cli.out, f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write(text + "\n")
            results["tables"][key] = {"seeds": used, "aggregate": agg, "per_seed": per_seed}
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {results_path} and one .txt per table")


if __name__ == "__main__":
    main()
