#!/usr/bin/env python3
"""Web front end for the booth demo -- handover doc section 4.3.

    cd demo
    pip install fastapi uvicorn
    python app.py --drl ../integration/checkpoints/taichung/drl_fusion_togo25.pt
    # then open http://127.0.0.1:8000

WHAT YOU GET WITHOUT SUMO
    Two worlds side by side over the same demand: one routed by policy 4 (the herding
    baseline -- every vehicle follows the same forecast) and one by policy 7 (the trained
    agent). Vehicles advance along their routes, edges colour by load / capacity, and a
    visitor shutting a road makes both worlds re-plan under their own policy.

    That is deliberately the whole demo, not a placeholder. The handover doc (5.3) argues
    the most convincing picture -- two load heat maps diverging on the same closure --
    needs no simulator at all, and building it this way means a delay in SUMO cannot take
    the booth down with it. SUMO then upgrades the picture from "coloured edges" to
    "watch the cars".

THE SPLIT WITH controller.py
    Everything here talks to a `Backend`. `FakeBackend` below is one implementation;
    `controller.py` is the other -- same five methods, but state comes from TraCI and
    routes are injected with traci.vehicle.setRoute(). Nothing in app.py or index.html
    changes when it lands.

WHAT IS AND IS NOT MEASURED HERE
    Vehicles move under the same BPR volume-delay function the report uses, so a jammed
    corridor really does slow its traffic and the herding world really does fall behind.
    It is NOT a microscopic simulation: no car-following, no junctions, no signals.
    🔴 Nothing on this page is a result. The reported numbers come from run_compare.py
    and live in the log by section; this is an illustration of them.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "integration"))


import config as C                                                    # noqa: E402
from reroute_service import LOAD_WINDOW_S, LoadWindow, Router         # noqa: E402
# PANES, the Backend contract and the demand generator live in shared.py so that
# controller.py's SumoBackend gets the SAME trips and the SAME common subset.
from shared import (PANES, Backend, Funnel, arena_shapes,             # noqa: E402
                    common_subset, plan_all)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


class Vehicle:
    """One car, walking its route edge by edge."""
    __slots__ = ("vid", "slot", "dest", "route", "i", "left", "done")

    def __init__(self, vid, slot, dest, route):
        self.vid, self.slot, self.dest, self.route = vid, slot, dest, route
        self.i, self.left, self.done = 0, 0.0, False

    @property
    def edge(self):
        return None if self.done else self.route[self.i]

    def remaining(self):
        return self.route[self.i:]


class World:
    """One fleet under one policy, advancing in simulated seconds."""

    def __init__(self, router, key):
        self.r, self.key = router, key
        self.t = 0.0
        self.closed = set()
        self.window = LoadWindow()
        self.load = {}
        self.fleet = []
        self.arrived = self.stranded = self.episodes = 0
        # Two DIFFERENT events, kept apart on purpose. `plan_stats` is the episode's
        # assignment (of 800 demanded trips, how many this policy could route);
        # `last_reroute` is a visitor's closure (of the cars still driving, how many got a
        # new route). Sharing one field made the panel read "798/436" -- a numerator from
        # the dispatch over a denominator from the re-route.
        self.plan_stats = {}
        self.last_reroute = {}
        self._attrs = {eid: (router.g.edges[e]["t0"], router.g.edges[e]["cap"])
                       for e, eid in router.edge_ids.items()}

    # ---------------------------------------------------------------- setup ---
    def adopt(self, demand, offsets, paths, keep):
        """Take an episode's assignment: one batch, this world's policy, common subset.

        🔴 The assignment is a BATCH and has to be. eq. 4 couples vehicles through the
        load they leave behind -- vehicle k routes around what 0..k-1 already filled --
        so a batch is where policies 6 and 7 differ from 4 at all. Replacing arrivals one
        at a time instead (which this first did) hands every vehicle an empty network to
        read, and the panes converge no matter which policy each one names.

        🔴 `keep` is the COMMON subset: the trips EVERY policy could route. Dropping only
        each world's own failures instead gave the two panes different fleet sizes -- 768
        against 641 under a closure, because greedy decoding dead-ends (log 13.23) -- and
        a pane with 17% fewer cars carries less load and posts a better worst-rho for a
        reason that has nothing to do with its policy. The report scores the same way, on
        the trips all policies completed.

        Vehicles are placed PARTWAY along their routes, at the same fraction in both
        worlds, so the map opens already loaded and the panes stay comparable.
        """
        self.fleet = []
        for i in keep:
            p, (_, d), frac = paths[i], demand[i], offsets[i]
            edges = [self.r.edge_ids[e] for e in zip(p[:-1], p[1:])]
            v = Vehicle(f"v{i}", i, self.r.i2o[d], edges)
            v.i = min(int(frac * len(edges)), len(edges) - 1)
            v.left = self._travel(edges[v.i])
            self.fleet.append(v)
        # Per EPISODE, not cumulative: the report's served% is "of this fleet", and a
        # counter that runs past the fleet size cannot be read against it.
        self.arrived = self.stranded = 0
        self.episodes += 1
        self._observe()

    def spent(self):
        """Fraction of the fleet that has finished -- when to dispatch the next episode."""
        if not self.fleet:
            return 1.0
        return sum(1 for v in self.fleet if v.done) / len(self.fleet)

    def _travel(self, eid):
        """BPR time on this edge at the current load -- so a jam really does slow cars."""
        t0, cap = self._attrs[eid]
        return t0 * (1.0 + C.BPR_A * (self.load.get(eid, 0.0) / cap) ** C.BPR_B)

    # ------------------------------------------------------------- stepping ---
    def step(self, dt):
        self.t += dt
        for v in self.fleet:
            if v.done:
                continue
            v.left -= dt
            # `while`, not `if`: a short edge can be crossed several times inside one
            # tick, and dropping those would under-count the load window.
            while v.left <= 0.0 and not v.done:
                if v.i + 1 >= len(v.route):
                    v.done, self.arrived = True, self.arrived + 1
                    break
                v.i += 1
                if v.edge in self.closed:
                    # Committed to this edge when the road shut. Real traffic queues;
                    # here the car stops and stays in the count as stranded rather than
                    # quietly disappearing, because "served" is a reported metric.
                    v.done, self.stranded = True, self.stranded + 1
                    break
                v.left += self._travel(v.edge)
        self._observe()

    def _observe(self):
        self.window.observe(self.t, {v.vid: v.edge for v in self.fleet if not v.done})
        self.load = self.window.counts()

    def closed_edges(self):
        return {e for e in (self.r.parse_edge(x) for x in self.closed) if e}

    # ------------------------------------------------------------ re-routing ---
    def reroute(self):
        """Re-plan every car still driving, under this world's own policy."""
        active = [(v.vid, v.edge, v.dest, v.remaining())
                  for v in self.fleet if not v.done]
        if not active:
            return {"routed": 0, "active": 0, "seconds": 0.0}
        routes = self.r.reroute(active, sorted(self.closed), policy=self.key)
        by_id = {v.vid: v for v in self.fleet}
        for vid, route in routes.items():
            v = by_id[vid]
            # The router returns a route that STARTS on the car's current edge, so the
            # car keeps its position: index 0, and whatever time it had left on that edge.
            v.route, v.i = route, 0
        self.last_reroute = dict(self.r.last_stats)
        return self.last_reroute

    # ---------------------------------------------------------------- report ---
    def snapshot(self):
        st = self.r.network_state(self.load)
        return {
            "t": round(self.t, 1),
            "driving": sum(1 for v in self.fleet if not v.done),
            "episode": self.episodes,
            "fleet": len(self.fleet),
            "plan": self.plan_stats,
            "arrived": self.arrived,
            "stranded": self.stranded,
            "worst_rho": round(st["worst_rho"], 3),
            "gini_load": round(st["gini_load"], 4),
            "frac_saturated": round(st["frac_saturated"], 4),
            "edges": {k: v["rho"] for k, v in st["edges"].items()},
            "last_reroute": self.last_reroute,
        }


class FakeBackend(Backend):
    """Two worlds and a stepping thread, no simulator."""

    def __init__(self, router, vehicles=800, speed=5.0, tick=0.25, seed=C.SEED,
                 refresh=0.85, episodes=0):
        self.r = router
        self.n, self.speed, self.tick, self.seed = vehicles, speed, tick, seed
        # Fraction of the fleet that must have finished before the next
        # episode is dispatched. Lower means fresher traffic and more
        # frequent pauses while policy 7 re-plans (~6 s at 800 vehicles).
        self.refresh = refresh
        # 0 = keep dispatching forever (a booth should never run dry);
        # N = stop after N episodes and hold the final frame, which is the
        # shape the reported experiment has and the only way to read a
        # served% off this page against it.
        self.episodes = episodes
        self.lock = threading.RLock()
        self.busy = ""
        self._stop = threading.Event()
        self.funnel = Funnel(router, seed, vehicles)
        self._detail = arena_shapes()
        self._build()
        threading.Thread(target=self._run, daemon=True).start()

    def _build(self):
        self.episode = -1
        self.dropped = 0
        self.finished = False
        self.closed = set()
        # Road NAMES as well as edge ids: the page highlights the buttons
        # from this, so the state survives a reload.
        self.closed_roads = []
        self.worlds = {p["key"]: World(self.r, p["key"])
                       for p in PANES if p["key"] in self.r.available()}
        self._next_episode()

    def _next_episode(self):
        """Plan one episode for every world, then hand them all the SAME trip subset."""
        self.episode += 1
        demand, offsets = self.funnel.episode(self.episode)

        def progress(k):
            self.busy = f"episode {self.episode + 1}: planning {k}"
        plans, stats = plan_all(self.r, list(self.worlds), demand, self.closed, progress)
        # Every policy has to be able to route a trip for it to count. See World.adopt.
        keep = common_subset(plans)
        self.dropped = len(demand) - len(keep)
        for k, w in self.worlds.items():
            w.plan_stats = stats[k]
            w.last_reroute = {}          # a new episode has had no closure re-route yet
            w.adopt(demand, offsets, plans[k], keep)
        self.busy = ""

    def _run(self):
        while not self._stop.is_set():
            time.sleep(self.tick)
            if self.busy or self.finished:
                continue                      # re-planning, or the run is over
            with self.lock:
                for w in self.worlds.values():
                    w.step(self.tick * self.speed)
                # Both worlds move on together, or they stop being one experiment.
                if all(w.spent() >= self.refresh for w in self.worlds.values()):
                    if self.episodes and self.episode + 1 >= self.episodes:
                        self.finished = True   # --episodes reached; hold the final frame
                    else:
                        self._next_episode()

    # ------------------------------------------------------------------------ #
    def geometry(self):
        return self.r.geometry(self._detail)

    def roads(self):
        return [r for r in self.r.roads(limit=8) if r["safe"]]

    def state(self):
        # 🔴 Do not wait for the lock while a re-plan holds it. close_road() keeps it for
        # as long as policy 7 takes (~6 s at 800 vehicles), and blocking here would freeze
        # the page for those 6 seconds -- precisely when it should be saying "re-planning".
        # An empty `panes` leaves the last frame on screen, which is what the visitor
        # wants to keep looking at anyway.
        if self.busy:
            return {"panes": {}, "closed": sorted(self.closed), "busy": self.busy,
                    "closed_roads": list(self.closed_roads),
                    "window_s": round(LOAD_WINDOW_S)}
        with self.lock:
            return {"panes": {k: w.snapshot() for k, w in self.worlds.items()},
                    "closed": sorted(self.closed), "busy": self.busy,
                    "dropped": self.dropped, "finished": self.finished,
                    "closed_roads": list(self.closed_roads),
                    "window_s": round(LOAD_WINDOW_S)}

    def close_road(self, road):
        edges = self.r.road_edges(road)           # raises ValueError on an unknown name
        out = {}
        if road not in self.closed_roads:
            self.closed_roads.append(road)
        self.busy = f"closing {road}"
        try:
            with self.lock:
                self.closed.update(edges)
                for k, w in self.worlds.items():
                    w.closed.update(edges)
                    self.busy = f"{road}: re-planning {k}"
                    out[k] = w.reroute()
        finally:
            self.busy = ""
        return {"road": road, "edges": len(edges), "panes": out}

    def reset(self):
        self.busy = "resetting"
        try:
            with self.lock:
                self._build()
        finally:
            self.busy = ""
        return {"ok": True}

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------- #
def build_app(backend):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
    except ImportError:
        raise SystemExit("FastAPI is not installed:  pip install fastapi uvicorn")

    app = FastAPI(title="Dynamic Road-Network Optimization -- booth demo")

    @app.get("/")
    def index():
        return FileResponse(HERE / "index.html")

    @app.get("/geometry")
    def geometry():
        return {"edges": backend.geometry(), "panes": PANES, "roads": backend.roads()}

    @app.get("/state")
    def state():
        return backend.state()

    @app.post("/close")
    def close(body: dict):
        road = (body or {}).get("road")
        if not road:
            raise HTTPException(400, 'body must be {"road": "<name>"}')
        try:
            return backend.close_road(road)
        except ValueError as exc:                 # unknown road name
            raise HTTPException(404, str(exc))

    @app.post("/reset")
    def reset():
        return backend.reset()

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drl", default=None, metavar="CKPT.pt",
                    help="without it the right pane is unavailable and only the herding "
                         "baseline is shown")
    ap.add_argument("--vehicles", type=int, default=800,
                    help="800 is what capacity_scale was calibrated for; at 300 no edge "
                         "crosses RHO_THRESHOLD and the eq.4 penalty never fires")
    ap.add_argument("--speed", type=float, default=5.0, metavar="X",
                    help="simulated seconds per real second")
    ap.add_argument("--beam", type=int, default=0, metavar="W",
                    help="beam width for policy 7 (default greedy). Greedy is ~6 s per "
                         "re-plan at 800 vehicles on GPU; beam-8 is 8-10x that")
    ap.add_argument("--episodes", type=int, default=0, metavar="N",
                    help="stop after N 800-vehicle episodes and hold the last frame "
                         "(0 = keep going, which is what a booth wants)")
    ap.add_argument("--refresh", type=float, default=0.85, metavar="FRAC",
                    help="dispatch the next 800-vehicle episode once this fraction of the "
                         "current one has finished")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--backend", default="fake", choices=["fake", "sumo"],
                    help="'fake' needs no simulator. 'sumo' imports "
                         "controller.SumoBackend(router, vehicles=, speed=) -- the same "
                         "five Backend methods over TraCI. Nothing else changes")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    cli = ap.parse_args()

    print(f"\n{'=' * 78}\nbooth demo -- building the router\n{'=' * 78}")
    router = Router(drl=cli.drl, device=cli.device, beam=cli.beam).warmup()
    if "drl" not in router.available():
        print("[demo] no --drl checkpoint, so the right pane will be empty. Pass\n"
              "       --drl ../integration/checkpoints/taichung/drl_fusion_togo25.pt")
    if cli.backend == "sumo":
        # Imported here, not at the top: 'fake' must keep working on a machine with no
        # SUMO and no controller.py, which is the whole point of having two backends.
        try:
            from controller import SumoBackend
        except ImportError as exc:
            raise SystemExit(f"--backend sumo needs demo/controller.py with a SumoBackend "
                             f"class (see the Backend docstring in this file): {exc}")
        backend = SumoBackend(router, vehicles=cli.vehicles, speed=cli.speed)
    else:
        backend = FakeBackend(router, vehicles=cli.vehicles, speed=cli.speed,
                              refresh=cli.refresh, episodes=cli.episodes)
    print(f"[demo] backend={cli.backend}, {cli.vehicles} vehicles per pane, "
          f"{cli.speed:g}x speed, panes: {', '.join(router.available())}")
    # A booth is exactly where the venue wifi fails. Leaflet from a CDN is a single point
    # of failure for the whole page; map tiles are not (without them the basemap is blank
    # but every road still draws, because the polylines come from our own graph).
    if not (HERE / "vendor" / "leaflet.js").is_file():
        print("[demo] WARNING the page loads Leaflet from unpkg, so it needs internet.\n"
              "       For the booth, vendor it once and re-point the two tags in "
              "index.html:\n"
              "         mkdir -p vendor && cd vendor\n"
              "         curl -LO https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\n"
              "         curl -LO https://unpkg.com/leaflet@1.9.4/dist/leaflet.css")
    print(f"[demo] open  http://{cli.host}:{cli.port}\n")

    import uvicorn
    uvicorn.run(build_app(backend), host=cli.host, port=cli.port, log_level="warning")


if __name__ == "__main__":
    main()
