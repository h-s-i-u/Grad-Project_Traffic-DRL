#!/usr/bin/env python3
"""Export the arena and a set of routed vehicles as SUMO input files.

This is the downstream half of the SUMO integration ("mode A" in
paper_work/交接_SUMO_20260831.md): the routing decisions are made here, under the BPR
congestion model, and SUMO replays and renders them. Nothing in the output needs PyTorch,
so whoever runs SUMO never has to install torch_geometric.

WHY THE NETWORK IS EXPORTED TOO, NOT JUST THE ROUTES
    Map/arena_{nodes,edges}_taichung.csv already carry everything netconvert needs --
    coordinates, length, speed limit, lane count, one-way, all with zero missing values.
    Building the .net.xml from them rather than re-extracting OSM means WE choose the edge
    ids, so the "my node/edge id <-> SUMO edge id" mapping table that the June handover
    left open for two months simply does not exist: an edge is `<from_osmid>_<to_osmid>`
    on both sides, and the correspondence is the identity.

COORDINATES
    Node positions are projected to metres with a local equirectangular projection about
    the arena's centroid, written into projection.json so a map overlay can invert it.
    Over a city-sized extent at this latitude the distortion is well under 0.1%, and it
    does not reach travel times anyway: every edge carries an explicit `length` taken from
    length_m, so SUMO uses the measured road length rather than the drawn geometry.

TIME
    The assignment model has no clock. Vehicles are dispatched in order and load
    accumulates; that ordering is the only temporal structure there is, and the S3 closure
    is defined on it (`at` is a fraction of the dispatch sequence, not a wall-clock time).
    So departures are spread linearly over --window seconds, and the closure lands at
    `at * window`. The mapping is a presentation choice, not a measurement -- say so if
    the demo shows a clock.

    cd integration
    python export_sumo.py --drl checkpoints/taichung/drl_fusion_togo25.pt
    python export_sumo.py --drl checkpoints/taichung/drl_fusion_togo25.pt \\
           --close-road 臺灣大道 --close-at 0.10 --vehicles 300 --tag s3
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

import closure as clo
import config as C
import metrics as M
import network as net
import policies as pol
from run_compare import make_demand

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

EARTH_R = 6_371_000.0


def edge_id(g, u, v):
    """`<from_osmid>_<to_osmid>` -- the id on both sides, so no mapping table exists."""
    return f"{g.nodes[u]['osmid']}_{g.nodes[v]['osmid']}"


def project(g):
    """Local equirectangular projection to metres, about the arena centroid."""
    lats = [g.nodes[n]["lat"] for n in g.nodes]
    lons = [g.nodes[n]["lon"] for n in g.nodes]
    lat0, lon0 = float(np.mean(lats)), float(np.mean(lons))
    kx = EARTH_R * math.cos(math.radians(lat0)) * math.pi / 180.0
    ky = EARTH_R * math.pi / 180.0
    xy = {n: ((g.nodes[n]["lon"] - lon0) * kx, (g.nodes[n]["lat"] - lat0) * ky)
          for n in g.nodes}
    return xy, {"lat0": lat0, "lon0": lon0, "x_per_deg_lon": kx, "y_per_deg_lat": ky,
                "inverse": "lon = lon0 + x / x_per_deg_lon;  lat = lat0 + y / y_per_deg_lat"}


def pretty(root, path):
    xml = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml("    ")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


def write_network(g, out_dir):
    xy, proj = project(g)
    nod = ET.Element("nodes")
    for n in g.nodes:
        x, y = xy[n]
        ET.SubElement(nod, "node", id=str(g.nodes[n]["osmid"]),
                      x=f"{x:.2f}", y=f"{y:.2f}", type="priority")
    pretty(nod, os.path.join(out_dir, "taichung.nod.xml"))

    edg = ET.Element("edges")
    for u, v, d in g.edges(data=True):
        # `length` is set explicitly so SUMO uses the measured road length rather than the
        # distance between the drawn node positions; the projection is for looks only.
        # `speed` is m/s, and 88.2% of the underlying limits are imputed at 50/30 km/h
        # per 道路交通安全規則 §93 -- flagged in the CSV as speed_imputed.
        ET.SubElement(edg, "edge", id=edge_id(g, u, v),
                      **{"from": str(g.nodes[u]["osmid"]), "to": str(g.nodes[v]["osmid"])},
                      numLanes=str(int(d.get("lanes", 1) or 1)),
                      speed=f"{d['length'] / max(d['t0'], 1e-6):.2f}",
                      length=f"{d['length']:.2f}")
    pretty(edg, os.path.join(out_dir, "taichung.edg.xml"))

    with open(os.path.join(out_dir, "projection.json"), "w", encoding="utf-8") as f:
        json.dump(proj, f, indent=2)
    return proj


def write_routes(g, paths, out_path, window, label, closure=None, close_time=None):
    """One <vehicle> per served trip, departures spread over `window` seconds."""
    n = len(paths)
    root = ET.Element("routes")
    # An XML comment may not contain a double hyphen, so the prose here uses an
    # em dash. Left as a comment rather than dropped: whoever opens this file in SUMO
    # needs to know the clock is a presentation choice, not a measurement.
    root.append(ET.Comment(
        f" {label} | {sum(1 for p in paths if p)} of {n} vehicles served. "
        f"Departure times are the dispatch ORDER mapped onto {window:g} s; the "
        f"assignment model has no clock. "
        + (f"The closure lands at t = {close_time:.1f} s. " if close_time else "")))
    ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
                  length="5", maxSpeed="33.33")
    served = 0
    for i, p in enumerate(paths):
        if not p or len(p) < 2:
            continue                      # unserved: no route to write, counted below
        served += 1
        veh = ET.SubElement(root, "vehicle", id=f"v{i}", type="car",
                            depart=f"{window * i / max(1, n - 1):.2f}")
        ET.SubElement(veh, "route",
                      edges=" ".join(edge_id(g, a, b) for a, b in zip(p[:-1], p[1:])))
    pretty(root, out_path)
    return served, n - served


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicles", type=int, default=300,
                    help="a booth demo wants fewer than the 800 the report uses: the "
                         "router stays interactive (13.9 ms/vehicle) and SUMO stays "
                         "watchable")
    ap.add_argument("--scenario", choices=["random", "hotspot"], default=C.SCENARIO)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--drl", default=None, metavar="CKPT.pt",
                    help="also export policy 7. Without it only the analytic policies "
                         "are written, and no torch import happens")
    ap.add_argument("--policies", default="4,7",
                    help="which to export: 1 static, 4 herding baseline, 6 oracle, 7 DRL. "
                         "4 and 7 side by side ARE the demo")
    ap.add_argument("--window", type=float, default=600.0, metavar="SEC")
    ap.add_argument("--out-dir", default="sumo")
    ap.add_argument("--tag", default="s2")
    ap.add_argument("--capacity-scale", type=float, default=None)
    ap.add_argument("--close-road", default=None, metavar="PREFIX")
    ap.add_argument("--close-at", type=float, default=0.5, metavar="FRAC")
    cli = ap.parse_args()

    C.SCENARIO, C.N_VEHICLES = cli.scenario, cli.vehicles
    want = {p.strip() for p in cli.policies.split(",")}
    max_hops = net.default_max_hops("taichung")
    g, _ = net.build_graph_for("taichung", capacity_scale=cli.capacity_scale, verbose=False)
    scc = net.largest_scc(g)
    demand, hubs = make_demand(g, scc, np.random.default_rng(cli.seed))

    closure = close_time = None
    if cli.close_road:
        edges = clo.edges_by_road(g, cli.close_road)
        closure = clo.Closure(edges, at=cli.close_at, label=cli.close_road)
        demand, dem_info = clo.select_demand(g, closure, demand, "filter")
        close_time = cli.window * closure.cutoff(len(demand)) / max(1, len(demand) - 1)

    os.makedirs(cli.out_dir, exist_ok=True)
    print(f"\n{'=' * 88}\nSUMO export -- arena + routed vehicles ({cli.tag})\n{'=' * 88}")
    print(f"graph  : {g.number_of_nodes():,} nodes / {g.number_of_edges():,} edges")
    print(f"demand : {len(demand):,} vehicles, {cli.scenario}, seed {cli.seed}, "
          f"departures over {cli.window:g} s")
    if closure:
        print(f"closure: {closure.label}, {len(closure)} edges, at {cli.close_at:.0%} of "
              f"the dispatch order -> t = {close_time:.1f} s ({dem_info})")

    proj = write_network(g, cli.out_dir)
    print(f"\nnetwork -> taichung.nod.xml + taichung.edg.xml + projection.json")
    print(f"  edge ids are <from_osmid>_<to_osmid>, so no id mapping table is needed")
    print(f"  projection: equirectangular about ({proj['lat0']:.5f}, {proj['lon0']:.5f}); "
          f"every edge carries an explicit length, so geometry does not affect travel time")

    routed = {}
    if "1" in want:
        routed["1_static"] = pol.policy_static(g, demand, closure)
    if "4" in want:
        routed["4_herding"] = pol.policy_prediction_greedy(g, demand, closure=closure)
    if "6" in want:
        routed["6_oracle"] = pol.policy_global_penalty(g, demand, closure)
    if "7" in want:
        if not cli.drl:
            raise SystemExit("error: policy 7 requested but --drl was not given")
        meta_path = os.path.splitext(cli.drl)[0] + ".meta.json"
        togo = 0
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                togo = int(json.load(f).get("togo_refresh", 0) or 0)
        agent = pol.make_drl_agent(cli.drl, g)
        # Greedy, not beam-8. The report quotes beam-8 for quality, but greedy is what
        # runs in 13.9 ms/vehicle and keeps a live booth responsive (§13.25 ②). Label the
        # demo accordingly rather than quoting the report's numbers over it.
        routed["7_drl"] = pol.policy_drl(g, demand, agent, max_hops=max_hops,
                                         togo_refresh=togo, closure=closure)

    print(f"\nroutes:")
    ref = set()
    for paths in routed.values():
        load, _ = M.edge_loads(g, paths)
        ref |= {e for e, v in load.items() if v > 0}
    ref = sorted(ref)
    summary = {}
    for name, paths in routed.items():
        out = os.path.join(cli.out_dir, f"{cli.tag}_{name}.rou.xml")
        served, lost = write_routes(g, paths, out, cli.window, f"{cli.tag} {name}",
                                    closure, close_time)
        m = M.evaluate(g, paths, ref)
        summary[name] = {"served": served, "unserved": lost, "att": m["att"],
                         "gini_load": m["gini_load"], "worst_rho": m["worst_rho"]}
        print(f"  {name:<12} served {served:>4}/{len(paths):<4} "
              f"ATT {m['att']:8.1f}  Gini {m['gini_load']:.4f}  "
              f"worst-rho {m['worst_rho']:.3f}  -> {os.path.basename(out)}")

    # The load per edge is what makes the herding effect visible; SUMO can colour by it,
    # and a Leaflet overlay can use it without SUMO at all.
    for name, paths in routed.items():
        load, _ = M.edge_loads(g, paths)
        rows = [{"edge": edge_id(g, u, v), "vehicles": load[(u, v)],
                 "rho": load[(u, v)] / g.edges[(u, v)]["cap"]}
                for u, v in g.edges() if load.get((u, v), 0) > 0]
        with open(os.path.join(cli.out_dir, f"{cli.tag}_{name}.load.json"),
                  "w", encoding="utf-8") as f:
            json.dump(sorted(rows, key=lambda r: -r["rho"]), f, indent=1)

    cmd = ("netconvert --node-files=taichung.nod.xml --edge-files=taichung.edg.xml "
           "--output-file=taichung.net.xml")
    with open(os.path.join(cli.out_dir, "build_net.sh"), "w", encoding="utf-8") as f:
        f.write(f"#!/bin/sh\n# one command; ids come from our CSVs so nothing has to be "
                f"mapped\n{cmd}\n")
    with open(os.path.join(cli.out_dir, f"{cli.tag}.sumocfg"), "w", encoding="utf-8") as f:
        f.write('<configuration>\n  <input>\n'
                '    <net-file value="taichung.net.xml"/>\n'
                f'    <route-files value="{cli.tag}_'
                f'{sorted(routed)[0] if routed else "4_herding"}.rou.xml"/>\n'
                '  </input>\n  <time>\n    <begin value="0"/>\n'
                f'    <end value="{cli.window * 2:.0f}"/>\n  </time>\n</configuration>\n')

    with open(os.path.join(cli.out_dir, f"{cli.tag}_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"tag": cli.tag, "vehicles": len(demand), "scenario": cli.scenario,
                   "seed": cli.seed, "window_s": cli.window,
                   "closure": (None if not closure else
                               {"label": closure.label, "edges": len(closure),
                                "at_fraction": cli.close_at, "at_seconds": close_time}),
                   "decoding": "greedy (not beam-8) -- see the note in the source",
                   "policies": summary, "projection": proj}, f,
                  indent=2, ensure_ascii=False)

    print(f"\nalso wrote: build_net.sh, {cli.tag}.sumocfg, {cli.tag}_summary.json, "
          f"and one *.load.json per policy\n  (edge -> vehicles/rho, which is what makes "
          f"the herding effect visible; usable without SUMO)")
    print(f"\nnext, in {cli.out_dir}/ :\n    sh build_net.sh\n    sumo-gui -c {cli.tag}.sumocfg")


if __name__ == "__main__":
    main()
