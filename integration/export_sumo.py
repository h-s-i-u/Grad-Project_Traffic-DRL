#!/usr/bin/env python3
"""Export the arena and a set of routed vehicles as SUMO input files.

This is the downstream half of the SUMO integration: the routing decisions are made here,
under the BPR congestion model, and SUMO replays and renders them. Nothing in the output
needs PyTorch, so whoever runs SUMO never has to install torch_geometric.

WHY THE NETWORK IS EXPORTED TOO, NOT JUST THE ROUTES
    Map/arena_{nodes,edges}_taichung.csv already carry everything netconvert needs --
    coordinates, length, speed limit, lane count, one-way, all with zero missing values.
    Building the .net.xml from them rather than re-extracting OSM means WE choose the edge
    ids, so the "my node/edge id <-> SUMO edge id" mapping table that stayed open for two
    months simply does not exist: an edge is `<from_osmid>_<to_osmid>` on both sides, and
    the correspondence is the identity.

COORDINATES
    Node positions are projected to metres with a local equirectangular projection about
    the arena's centroid, written into projection.json so a map overlay can invert it.
    Over a city-sized extent at this latitude the distortion is well under 0.1%, and it
    does not reach travel times anyway: every edge carries an explicit `length` taken from
    length_m, so SUMO uses the measured road length rather than the drawn geometry.

SHAPES
    An arena edge is a merged chain and the CSV keeps only its endpoints, so without more
    netconvert draws every road as a straight chord -- 4.3 km of 十甲東路 as one line
    across the city, curves as a few kinks, chords crossing each other. demo/
    build_geometry.py recovered the real polylines (demo/arena_geometry.json, committed);
    --shapes projects them and writes each as the edge's `shape`. This is not cosmetic
    only: netconvert infers the TURNS at a junction from the direction the edges arrive
    in, and two chords of a bending road can meet at 176 degrees where the road itself
    bends by 30 -- netconvert then files the continuation as a U-turn and builds no
    connection (民權路 at node 5521014889, refused live on 7 Sep). Real shapes cut the
    transitions that look like U-turns from 27 to 14 of 3,655.

CONNECTIONS CHECK
    The remaining 14 are settled by measurement, not by guessing netconvert's thresholds:
    `--check-net taichung.net.xml` reads every <connection> the built network has and
    compares it with every turn the graph allows (u -> v -> w with w != u; immediate
    reversals are on no route the router can produce, so they are not required). Missing
    pairs are listed; taichung.fix.con.xml then states, lane by lane, every turn out of
    each incoming edge that has a gap (netconvert reads a listed edge's connections as
    complete, so naming only the gap would drop that edge's other turns), and
    build_net.sh rebuilds with that file and checks again. Every other edge keeps
    netconvert's own lane assignment.

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


DEFAULT_SHAPES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "demo", "arena_geometry.json")


def load_shapes(path):
    """{edge_id: [[lat, lon], ...]} from demo/build_geometry.py, or {} if absent."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_network(g, out_dir, shapes=None):
    xy, proj = project(g)
    kx, ky, lat0, lon0 = (proj["x_per_deg_lon"], proj["y_per_deg_lat"],
                          proj["lat0"], proj["lon0"])
    nod = ET.Element("nodes")
    for n in g.nodes:
        x, y = xy[n]
        ET.SubElement(nod, "node", id=str(g.nodes[n]["osmid"]),
                      x=f"{x:.2f}", y=f"{y:.2f}", type="priority")
    pretty(nod, os.path.join(out_dir, "taichung.nod.xml"))

    edg = ET.Element("edges")
    n_shaped = 0
    for u, v, d in g.edges(data=True):
        # `length` is set explicitly so SUMO uses the measured road length rather than the
        # distance between the drawn node positions; the projection is for looks only.
        # `speed` is m/s, and 88.2% of the underlying limits are imputed at 50/30 km/h
        # per 道路交通安全規則 §93 -- flagged in the CSV as speed_imputed.
        eid = edge_id(g, u, v)
        attrs = {"from": str(g.nodes[u]["osmid"]), "to": str(g.nodes[v]["osmid"]),
                 "numLanes": str(int(d.get("lanes", 1) or 1)),
                 "speed": f"{d['length'] / max(d['t0'], 1e-6):.2f}",
                 "length": f"{d['length']:.2f}"}
        pts = (shapes or {}).get(eid)
        if pts and len(pts) >= 2:
            # The recovered chain runs between exactly these two nodes, so its ends are
            # snapped onto the node positions netconvert will use.
            line = [((lon - lon0) * kx, (lat - lat0) * ky) for lat, lon in pts]
            line[0], line[-1] = xy[u], xy[v]
            attrs["shape"] = " ".join(f"{x:.2f},{y:.2f}" for x, y in line)
            n_shaped += 1
        ET.SubElement(edg, "edge", id=eid, **attrs)
    pretty(edg, os.path.join(out_dir, "taichung.edg.xml"))

    with open(os.path.join(out_dir, "projection.json"), "w", encoding="utf-8") as f:
        json.dump(proj, f, indent=2)
    return proj, n_shaped


def graph_turns(g):
    """{(edge_id A, edge_id B): (u, v, w)} for every u -> v -> w with w != u.

    Immediate reversals (w == u) are left out on purpose: Dijkstra never cycles and the
    agent never revisits a node, so no route the router can produce contains one, and
    requiring them would only add turnarounds netconvert rightly skips.
    """
    return {(edge_id(g, u, v), edge_id(g, v, w)): (u, v, w)
            for v in g.nodes for u in g.predecessors(v) for w in g.successors(v)
            if u != w}


def net_connections(net_path):
    """{(from edge, to edge)} present in a built .net.xml (internal edges skipped)."""
    have = set()
    for _, el in ET.iterparse(net_path, events=("end",)):
        if el.tag == "connection":
            a, b = el.get("from"), el.get("to")
            if a and b and not a.startswith(":"):
                have.add((a, b))
            el.clear()
    return have


def write_fix_connections(g, want, missing, path):
    """Lane-level connections for every turn OUT OF an incoming edge that has a gap.

    Not just the missing pairs: netconvert takes a listed edge's connections as that
    edge's COMPLETE set, so a file naming only the gap drops the turns netconvert had
    built for the same edge by itself (measured: fixing 3 gaps this way opened 3 others,
    'Lane ... is not connected from any incoming edge'). Lane i continues into lane i, or
    the last lane when the next edge is narrower; a wider next edge has its extra lanes
    fed from the last lane so every lane stays reachable.
    """
    from_edges = {a for (a, _), _ in missing}
    con = ET.Element("connections")
    n_pairs = n_lanes = 0
    for (a, b), (u, v, w) in sorted(want.items()):
        if a not in from_edges:
            continue
        na = max(1, int(g.edges[(u, v)].get("lanes", 1) or 1))
        nb = max(1, int(g.edges[(v, w)].get("lanes", 1) or 1))
        pairs = [(i, min(i, nb - 1)) for i in range(na)]
        pairs += [(na - 1, j) for j in range(na, nb)]
        for i, j in pairs:
            ET.SubElement(con, "connection", **{"from": a, "to": b,
                                                "fromLane": str(i), "toLane": str(j)})
        n_pairs += 1
        n_lanes += len(pairs)
    pretty(con, path)
    return len(from_edges), n_pairs, n_lanes


def check_net(g, net_path, out_dir, strict=False):
    """Compare the built network's turns with the graph's; write the fix file if needed.

    Returns the number of missing pairs. With `strict`, a non-zero count is an error --
    that is the second pass of build_net.sh, after the fix file has been applied.
    """
    want = graph_turns(g)
    have = net_connections(net_path)
    missing = sorted((k, want[k]) for k in want if k not in have)
    fix = os.path.join(out_dir, "taichung.fix.con.xml")
    print(f"\n{'=' * 88}\nnetwork check -- {os.path.basename(net_path)}\n{'=' * 88}")
    print(f"turns the graph allows : {len(want):,}")
    print(f"present in the network : {len(want) - len(missing):,}")
    print(f"missing                : {len(missing)}")
    if not missing:
        if os.path.isfile(fix):
            os.remove(fix)             # a stale fix file would trigger a needless rebuild
        print("every turn the router can produce exists in SUMO's network")
        return 0
    for (a, b), (u, v, w) in missing:
        print(f"  {a} -> {b}   ({g.edges[(u, v)].get('road_name') or '-'} -> "
              f"{g.edges[(v, w)].get('road_name') or '-'}, at node {g.nodes[v]['osmid']})")
    if strict:
        raise SystemExit(f"error: {len(missing)} turn(s) still missing after the fix "
                         f"file was applied -- inspect the pairs above in netedit")
    n_from, n_pairs, n_lanes = write_fix_connections(g, want, missing, fix)
    print(f"\nwrote {os.path.basename(fix)}: every turn out of the {n_from} incoming "
          f"edge(s) with a gap -- {n_pairs} edge pairs, {n_lanes} lane pairs "
          f"({len(missing)} of them were missing). build_net.sh rebuilds with it and "
          f"checks again.")
    return len(missing)


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
    ap.add_argument("--beam", type=int, default=0, metavar="W",
                    help="decoding width for policy 7. 0 = greedy, which is what the live "
                         "demo runs (13.9 ms/vehicle); 8 is what the report quotes for "
                         "quality and what the replay validation (run_sumo_compare.py) "
                         "must use to compare against the reported row")
    ap.add_argument("--shapes", default=DEFAULT_SHAPES, metavar="JSON",
                    help="real road shapes from demo/build_geometry.py, written as each "
                         "edge's `shape` (module docstring: SHAPES). Default: the "
                         "committed demo/arena_geometry.json if it exists; 'none' to skip")
    ap.add_argument("--check-net", default=None, metavar="NET.xml",
                    help="compare a built network's turns with the graph's, write "
                         "taichung.fix.con.xml for any that are missing, and exit. "
                         "build_net.sh runs this; nothing else is exported")
    ap.add_argument("--strict", action="store_true",
                    help="with --check-net: fail if any turn is still missing")
    cli = ap.parse_args()

    if cli.check_net:
        g, _ = net.build_graph_for("taichung", capacity_scale=cli.capacity_scale,
                                   verbose=False)
        os.makedirs(cli.out_dir, exist_ok=True)
        sys.exit(1 if check_net(g, cli.check_net, cli.out_dir, cli.strict) and cli.strict
                 else 0)

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

    shapes = {} if cli.shapes == "none" else load_shapes(cli.shapes)
    proj, n_shaped = write_network(g, cli.out_dir, shapes)
    print(f"\nnetwork -> taichung.nod.xml + taichung.edg.xml + projection.json")
    print(f"  edge ids are <from_osmid>_<to_osmid>, so no id mapping table is needed")
    print(f"  projection: equirectangular about ({proj['lat0']:.5f}, {proj['lon0']:.5f}); "
          f"every edge carries an explicit length, so geometry does not affect travel time")
    if n_shaped:
        print(f"  shapes: {n_shaped:,} edges drawn along their recovered road shape "
              f"({os.path.relpath(cli.shapes)}); the rest are straight in reality")
    else:
        print(f"  shapes: none -- every merged edge will draw as a straight chord and "
              f"netconvert will guess turns from those chords. Run demo/build_geometry.py "
              f"or pass --shapes")

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
        # Greedy by default: that is what runs in 13.9 ms/vehicle and keeps a live booth
        # responsive (§13.25 ②). The report quotes beam-8 for quality, so a replay meant
        # to be read against the reported row passes --beam 8; label whichever is used
        # rather than quoting the report's numbers over a greedy run.
        routed["7_drl"] = pol.policy_drl(g, demand, agent, max_hops=max_hops,
                                         togo_refresh=togo, closure=closure,
                                         beam=cli.beam)

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

    here = os.path.dirname(os.path.abspath(__file__))
    out_abs = os.path.abspath(cli.out_dir)
    build = ("netconvert --node-files=taichung.nod.xml --edge-files=taichung.edg.xml "
             "--output-file=taichung.net.xml")
    check = (f'(cd "{here}" && python export_sumo.py --check-net '
             f'"{out_abs}/taichung.net.xml" --out-dir "{out_abs}"')
    with open(os.path.join(cli.out_dir, "build_net.sh"), "w", encoding="utf-8") as f:
        f.write(f"#!/bin/sh\n"
                f"# Two passes. Ids come from our CSVs so nothing has to be mapped; the\n"
                f"# check compares every turn the graph allows with what netconvert built\n"
                f"# and, only if something is missing, rebuilds with a fix file and\n"
                f"# checks again (module docstring: CONNECTIONS CHECK).\n"
                f"set -e\n"
                f"{build}\n"
                f"{check})\n"
                f"if [ -s taichung.fix.con.xml ]; then\n"
                f"    {build.replace('--output-file', '--connection-files=taichung.fix.con.xml --output-file')}\n"
                f"    {check} --strict)\n"
                f"fi\n")
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
                   "decoding": (f"beam-{cli.beam}" if cli.beam > 1 else
                                "greedy (not beam-8) -- see the note in the source"),
                   "beam": cli.beam,
                   "policies": summary, "projection": proj}, f,
                  indent=2, ensure_ascii=False)

    print(f"\nalso wrote: build_net.sh, {cli.tag}.sumocfg, {cli.tag}_summary.json, "
          f"and one *.load.json per policy\n  (edge -> vehicles/rho, which is what makes "
          f"the herding effect visible; usable without SUMO)")
    print(f"\nnext, in {cli.out_dir}/ :\n    sh build_net.sh      # netconvert, then the "
          f"turn check, then a rebuild only if a turn is missing\n"
          f"    sumo-gui -c {cli.tag}.sumocfg")


if __name__ == "__main__":
    main()
