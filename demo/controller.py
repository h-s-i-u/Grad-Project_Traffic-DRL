#!/usr/bin/env python3
"""SumoBackend -- the same five Backend methods as FakeBackend, with the cars in SUMO.

    cd demo
    pip install eclipse-sumo traci sumolib          # binaries + the two Python packages
    (cd ../integration && python export_sumo.py --drl checkpoints/taichung/drl_fusion_togo25.pt
     && cd sumo && sh build_net.sh)                 # once: taichung.net.xml
    python controller.py --selftest --mock          # no SUMO needed: exercises the bookkeeping
    python controller.py --selftest --drl ../integration/checkpoints/taichung/drl_fusion_togo25.pt
    python app.py --backend sumo --drl ../integration/checkpoints/taichung/drl_fusion_togo25.pt

WHAT CHANGES AGAINST FakeBackend, AND WHAT DOES NOT
    Demand, episodes, the common subset and the re-routing call are shared (shared.py,
    reroute_service.py); the page, the endpoints and the JSON shapes are untouched. What
    this file adds is the seam described in the development log, section 8: each pane is
    ONE SUMO INSTANCE, because a single simulation cannot carry two policies -- policy 4
    and policy 7 produce different traffic states. Both instances are stepped in lockstep
    from one thread.

    Load is counted the way rho is defined: edge ENTRIES over the last ~154 s
    (`LoadWindow`), from one road-id snapshot per step, not
    getLastStepVehicleNumber(). SUMO's load goes to the display through
    network_state() and NEVER into reroute(): seeding the router with it drops policy 7
    from 737/800 served to 345/800 while leaving policies 1/4/6 untouched (log 13.28).

    ATT is not on the panel. Offline it is the BPR estimate; live, the only true value is
    what SUMO measured, and `tripinfo` is where a later version should read it from.

THREE THINGS THAT DIFFER FROM THE FAKE STEPPER, ON PURPOSE
    * Vehicles are INSERTED, not placed. SUMO cannot put a car mid-route, so the route is
      cut at the same start fraction both panes use and the car departs at its head. Up
      to 800 insertions at t=0 queue on their first edges; `pending` on the panel is how
      many have not entered yet.
    * A car whose remaining route crosses the closure and that the router could not
      re-route is REMOVED at closure time and counted as stranded. The fake stepper
      strands the same set when they reach the closed edge; SUMO would instead teleport
      them across it after --time-to-teleport, which is neither honest nor visible.
    * Nothing here checks the clock against wall time: `--speed` is simulated seconds
      per real second, as before, and one SUMO step is one simulated second.

THE CLOSURE IS NOT WRITTEN INTO SUMO'S PERMISSIONS
    The obvious move is edge.setDisallowed() on the closed edges, and the first two live
    runs did exactly that. It refused 20-40 re-routes per pane with "No connection
    between edge A and edge B", where B was always a closed edge that the NEW route
    never contained. SUMO validates a replacement route from the beginning of the
    vehicle's history, not from its current edge -- the driven prefix is kept -- so any
    car that had used 臺灣大道 earlier in the episode was refused a new route the moment
    the road was disallowed, wherever it was by then. The closure therefore lives in the
    router (which masks the edges) and in the stranding rule above; nothing is asked of
    SUMO's permissions. What a visitor sees is the same: no new car enters the road, and
    the ones on it drive off. sumo-gui does not paint the road shut; the web page does.
    (Two things this diagnosis replaced, both wrong: that netconvert had dropped turns at
    sharp chord angles -- the routes SUMO named were never the routes we sent -- and that
    the "Vehicle X is not known" errors meant SUMO had lost cars. Those were the stale
    subscriptions of cars WE removed, reported once each on the next step.)

TRACI COST
    One road-id snapshot per step over 800 cars is 800 round trips if done with
    getRoadID(); done with subscriptions it is three calls per step regardless of fleet
    size, which is the difference between 5x speed and a page that cannot keep up.
"""
import argparse
import atexit
import os
import shutil
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "integration"))

import config as C                                                    # noqa: E402
from reroute_service import LOAD_WINDOW_S, LoadWindow, Router         # noqa: E402
from shared import (PANES, Backend, Funnel, arena_shapes,             # noqa: E402
                    common_subset, plan_all)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# The vType export_sumo.py writes, verbatim, so a replayed .rou.xml and a live fleet drive
# the same car.
VTYPE = ('<vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" '
         'maxSpeed="33.33"/>')
END_S = 10 ** 9      # never: a live simulation ends when the client closes it
JUNCTION_MSG = "junction-internal"   # SUMO's reason when a car mid-junction cannot turn


# --------------------------------------------------------------------------- #
#  one simulator instance, everything SUMO-specific
# --------------------------------------------------------------------------- #
class _Sim:
    """One SUMO process behind one TraCI connection.

    `step()` returns (time, {vehicle: road_id}, [arrived]) so the world above never
    touches TraCI itself, and `_MockSim` can stand in with the same five calls.
    """

    def __init__(self, label, cfg, gui=False, step_length=1.0, teleport=-1):
        import traci
        import traci.constants as tc
        self._traci, self._tc = traci, tc
        binary = sumo_binary("sumo-gui" if gui else "sumo")
        # SUMO's own warnings go to a file, not /dev/null: "vehicle X has no valid route"
        # and "teleporting" are exactly the messages that explain a car the bookkeeping
        # has lost track of, and the first real run hid them behind --no-warnings.
        self.log = Path(cfg).parent / f"live_{label}.log"
        cmd = [binary, "-c", str(cfg), "--step-length", str(step_length),
               "--no-step-log", "true", "--no-warnings", "false",
               "--log", str(self.log), "--verbose", "false",
               "--time-to-teleport", str(teleport), "--start", "true",
               "--quit-on-end", "true"]
        traci.start(cmd, label=label)
        self.c = traci.getConnection(label)
        self.label = label
        self._on_edge = {}

    def step(self):
        c, tc = self.c, self._tc
        c.simulationStep()
        # Subscribe each car the moment it enters; from then on its road id arrives in one
        # bulk call per step, and it drops out of the results when it leaves. A car that
        # enters and leaves inside one step is already gone by now -- not an error.
        for vid in c.simulation.getDepartedIDList():
            try:
                c.vehicle.subscribe(vid, [tc.VAR_ROAD_ID])
            except self._traci.TraCIException:
                pass
        res = c.vehicle.getAllSubscriptionResults()
        self._on_edge = {vid: r[tc.VAR_ROAD_ID] for vid, r in res.items()}
        return c.simulation.getTime(), self._on_edge, list(c.simulation.getArrivedIDList())

    def add(self, vid, route_id, edges):
        self.c.route.add(route_id, list(edges))
        self.c.vehicle.add(vid, route_id, typeID="car", depart="now")

    def where(self, vid):
        """(current route edge, remaining route incl. it, departed?). A car not yet
        inserted is at the head of its route, which is what setRoute() needs the first
        edge to be -- but it cannot be inserted onto a closed edge, hence the flag."""
        route = self.c.vehicle.getRoute(vid)
        if not route:
            return None
        i = self.c.vehicle.getRouteIndex(vid)
        departed = i >= 0
        i = max(i, 0)
        return route[i], list(route[i:]), departed

    def set_route(self, vid, edges):
        self.c.vehicle.setRoute(vid, list(edges))

    def remove(self, vid):
        # Drop the subscription first: SUMO otherwise evaluates it once more on the next
        # step, for a car that no longer exists, and prints an error per removed car.
        try:
            self.c.vehicle.unsubscribe(vid)
        except self._traci.TraCIException:
            pass
        self.c.vehicle.remove(vid)

    def close(self):
        try:
            self.c.close()
        except Exception:
            pass


class _MockSim:
    """Stands in for SUMO on a machine without it: every car spends EDGE_STEPS steps on
    each edge, then moves to the next.

    Exercises every piece of bookkeeping above the TraCI line -- episodes, the common
    subset, closure, route injection, stranding, the snapshot -- and nothing below it.
    Not a simulation of anything.
    """
    EDGE_STEPS = 10

    def __init__(self, label, *_, **__):
        self.label, self.t = label, 0.0
        self.veh = {}           # vid -> [route, index, steps left on this edge]

    def step(self):
        self.t += 1.0
        arrived = []
        for vid, st in list(self.veh.items()):
            st[2] -= 1
            if st[2] > 0:
                continue
            st[1] += 1
            st[2] = self.EDGE_STEPS
            if st[1] >= len(st[0]):
                arrived.append(vid)
                del self.veh[vid]
        return self.t, {vid: st[0][st[1]] for vid, st in self.veh.items()}, arrived

    def add(self, vid, route_id, edges):
        self.veh[vid] = [list(edges), 0, self.EDGE_STEPS]

    def where(self, vid):
        st = self.veh.get(vid)
        return None if st is None else (st[0][st[1]], list(st[0][st[1]:]), True)

    def set_route(self, vid, edges):
        st = self.veh[vid]
        if edges[0] != st[0][st[1]]:
            raise ValueError(f"{vid}: route must start on its current edge")
        st[0], st[1] = list(edges), 0

    def remove(self, vid):
        self.veh.pop(vid, None)

    def close(self):
        pass


def sumo_binary(name):
    """Path to `sumo` / `sumo-gui`: PATH first, then SUMO_HOME via sumolib."""
    found = shutil.which(name)
    if found:
        return found
    try:
        from sumolib import checkBinary
        return checkBinary(name)
    except Exception as exc:
        raise SystemExit(
            f"cannot find {name}: {exc}\n"
            f"  pip install eclipse-sumo traci sumolib   (puts the binaries on PATH)\n"
            f"  or install SUMO and set SUMO_HOME")


# --------------------------------------------------------------------------- #
#  one pane
# --------------------------------------------------------------------------- #
class SumoWorld:
    """One fleet under one policy, living in one simulator instance."""

    def __init__(self, router, key, sim):
        self.r, self.key, self.sim = router, key, sim
        self.t = 0.0
        self.closed = set()
        self.window = LoadWindow()
        self.load = {}
        self.on_edge = {}          # last step's {vid: road_id}: the cars actually driving
        self.alive = set()         # inserted or waiting to insert, not yet arrived/stranded
        self.current = set()       # the subset of `alive` that belongs to THIS episode
        self.retry = set()         # re-routed mid-junction; tried again once they exit
        self.dest = {}             # vid -> destination osmid, for the router
        self.fleet = 0
        self.arrived = self.stranded = self.rejected = self.lost = self.episodes = 0
        self._remove_failed = 0
        # Two DIFFERENT events, kept apart on purpose (World in app.py has the story).
        self.plan_stats = {}
        self.last_reroute = {}

    # ---------------------------------------------------------------- setup ---
    def adopt(self, tag, demand, offsets, paths, keep):
        """One episode's assignment: the common subset, routes cut at the start fraction.

        Cars from an earlier episode that are still driving STAY -- removing them would
        make cars vanish on screen -- but they are no longer `current`, so they count in
        `driving` and never in this episode's arrived/served. SUMO ids must be unique for
        the life of the process, so the episode tag is part of every id.
        """
        self.arrived = self.stranded = self.rejected = self.lost = 0
        self.fleet = len(keep)
        self.current = set()
        for i in keep:
            p, (_, d), frac = paths[i], demand[i], offsets[i]
            edges = [self.r.edge_ids[e] for e in zip(p[:-1], p[1:])]
            start = min(int(frac * len(edges)), len(edges) - 1)
            vid = f"{tag}_v{i}"
            try:
                self.sim.add(vid, f"{tag}_r{i}", edges[start:])
            except Exception as exc:
                # An edge netconvert did not keep. Counted, not hidden: if one pane
                # rejects a car the other accepts, the panes stop being one experiment.
                self.rejected += 1
                if self.rejected == 1:
                    print(f"[sumo:{self.key}] SUMO rejected {vid}: {exc}")
                continue
            self.alive.add(vid)
            self.current.add(vid)
            self.dest[vid] = self.r.i2o[d]
        self.episodes += 1

    def spent(self):
        """Fraction of this episode's fleet that has finished."""
        if not self.fleet:
            return 1.0
        return (self.arrived + self.stranded + self.rejected + self.lost) / self.fleet

    # ------------------------------------------------------------- stepping ---
    def step(self):
        self.t, self.on_edge, arrived = self.sim.step()
        self.window.observe(self.t, self.on_edge)
        self.load = self.window.counts()
        for vid in arrived:
            if vid in self.current:
                self.arrived += 1
            self._forget(vid)
        if self.retry:
            # A car inside a junction cannot be handed a route that leaves it by another
            # exit; once it is back on a normal edge (road id not ':...'), route it from
            # wherever the junction put it.
            ready = [v for v in self.retry
                     if v in self.alive and not self.on_edge.get(v, ":").startswith(":")]
            if ready:
                self._reroute(ready)

    def _forget(self, vid):
        self.alive.discard(vid)
        self.current.discard(vid)
        self.retry.discard(vid)
        self.dest.pop(vid, None)

    def _strand(self, vid):
        try:
            self.sim.remove(vid)
        except Exception as exc:
            # SUMO does not know a car we still list as alive: it dropped the car on its
            # own, and its log says why. Counted, and the first reason is printed.
            self._remove_failed += 1
            if self._remove_failed == 1:
                print(f"[sumo:{self.key}] remove failed for {vid}: {exc}")
        if vid in self.current:
            self.stranded += 1
        self._forget(vid)

    # ------------------------------------------------------------ re-routing ---
    def close(self, edges):
        """The closure is a fact about the ROUTER's graph only -- module docstring."""
        self.closed.update(edges)

    def reroute(self):
        """Re-plan every car still on the road, under this world's own policy."""
        self.last_reroute = self._reroute(self.alive)
        return self.last_reroute

    def _reroute(self, vids):
        active, remaining, stranded = [], {}, 0
        self._remove_failed = 0
        gone = 0
        for vid in sorted(vids):
            if vid not in self.alive:
                continue
            try:
                w = self.sim.where(vid)
            except Exception as exc:
                # Listed as alive, unknown to SUMO. It left without appearing in
                # getArrivedIDList() -- removed for an invalid route, teleported out, or
                # similar; the SUMO log has the reason. Dropped here, counted as `lost`.
                gone += 1
                if gone == 1:
                    print(f"[sumo:{self.key}] SUMO no longer knows {vid}: {exc}")
                if vid in self.current:
                    self.lost += 1
                self._forget(vid)
                continue
            if w is None:
                continue
            cur, rem, departed = w
            if not departed and cur in self.closed:
                # Waiting to enter on a road that just shut: SUMO would hold it in the
                # insertion queue forever, invisibly. Same fate as a car with no route.
                self._strand(vid)
                stranded += 1
                continue
            active.append((vid, cur, self.dest[vid], rem))
            remaining[vid] = rem
        if not active:
            return {"routed": 0, "active": 0, "seconds": 0.0,
                    "stranded_now": stranded, "gone": gone}
        routes = self.r.reroute(active, sorted(self.closed), policy=self.key)
        applied = failed = deferred = 0
        first_failure = ""
        for vid, route in routes.items():
            try:
                self.sim.set_route(vid, route)     # begins on the car's current edge
                applied += 1
                self.retry.discard(vid)
            except Exception as exc:
                if JUNCTION_MSG in str(exc):
                    deferred += 1               # step() tries again once it has exited
                    self.retry.add(vid)
                    continue
                # A route the router produced is legal on our graph. A refusal here is
                # SUMO disagreeing with us about the network or about the car, and the
                # answer is to find out why -- never to hand the car to SUMO's own
                # router, which would quietly replace policy 7's decision with a
                # Dijkstra route and dilute the contrast this page exists to show.
                failed += 1
                if failed == 1:
                    first_failure = str(exc)
                    print(f"[sumo:{self.key}] setRoute failed for {vid}: {exc}")
        # Heading into the closure with no legal route: stranded now, not teleported later.
        for vid, rem in remaining.items():
            if vid in routes:
                continue                        # applied, or deferred with a route waiting
            if not (self.closed & set(rem)):
                self.retry.discard(vid)         # old route stays clear of the closure
                continue
            self._strand(vid)
            stranded += 1
        return dict(self.r.last_stats, applied=applied, set_route_failed=failed,
                    deferred=deferred, first_failure=first_failure,
                    stranded_now=stranded, gone=gone, remove_failed=self._remove_failed)

    # ---------------------------------------------------------------- report ---
    def snapshot(self):
        st = self.r.network_state(self.load)
        driving = len(self.on_edge)
        return {
            "t": round(self.t, 1),
            "driving": driving,
            "pending": max(len(self.alive) - driving, 0),
            "episode": self.episodes,
            "fleet": self.fleet,
            "plan": self.plan_stats,
            "arrived": self.arrived,
            "stranded": self.stranded,
            "rejected": self.rejected,
            "lost": self.lost,
            "worst_rho": round(st["worst_rho"], 3),
            "gini_load": round(st["gini_load"], 4),
            "frac_saturated": round(st["frac_saturated"], 4),
            "edges": {k: v["rho"] for k, v in st["edges"].items()},
            "last_reroute": self.last_reroute,
        }


# --------------------------------------------------------------------------- #
#  the backend
# --------------------------------------------------------------------------- #
class SumoBackend(Backend):
    """Two SUMO instances and a stepping thread. app.py constructs it with
    SumoBackend(router, vehicles=, speed=); everything else comes from the environment:

        DEMO_SUMO_DIR   where export_sumo.py wrote taichung.net.xml (default ../integration/sumo)
        DEMO_SUMO_GUI   1 to open sumo-gui windows instead of headless sumo
        DEMO_SUMO_STEP  simulation step length in seconds (default 1)
    """

    def __init__(self, router, vehicles=800, speed=5.0, tick=0.25, seed=C.SEED,
                 refresh=0.85, episodes=0, sumo_dir=None, gui=None, mock=False,
                 panes=None, verbose=True):
        self.r = router
        self.n, self.speed, self.tick, self.seed = vehicles, speed, tick, seed
        self.refresh, self.episodes = refresh, episodes
        self.mock = mock
        self.gui = (os.environ.get("DEMO_SUMO_GUI", "0") == "1") if gui is None else gui
        self.step_length = float(os.environ.get("DEMO_SUMO_STEP", "1.0"))
        self.sumo_dir = Path(sumo_dir or os.environ.get("DEMO_SUMO_DIR")
                             or HERE.parent / "integration" / "sumo")
        self.panes = [p for p in (panes or PANES) if p["key"] in router.available()]
        self.lock = threading.RLock()
        self.busy = ""
        self._stop = threading.Event()
        self._acc = 0.0
        self.sims = {}
        self.worlds = {}
        self.funnel = Funnel(router, seed, vehicles)
        self._detail = arena_shapes(verbose)
        self.cfg = None if mock else self._write_config()
        self._build()
        # uvicorn exits without telling the backend; without this the SUMO processes
        # outlive the page that was driving them.
        atexit.register(self.stop)
        threading.Thread(target=self._run, daemon=True).start()

    # ---------------------------------------------------------------- files ---
    def _write_config(self):
        net = self.sumo_dir / "taichung.net.xml"
        if not net.is_file():
            raise SystemExit(
                f"{net} is missing. Build it once:\n"
                f"    cd integration\n"
                f"    python export_sumo.py --drl checkpoints/taichung/drl_fusion_togo25.pt\n"
                f"    cd sumo && sh build_net.sh\n"
                f"  (do NOT re-extract the network from OSM: the edge ids must stay "
                f"<from_osmid>_<to_osmid>, which is what makes this file need no id map)")
        rou = self.sumo_dir / "live.rou.xml"
        cfg = self.sumo_dir / "live.sumocfg"
        with open(rou, "w", encoding="utf-8") as f:
            f.write(f"<routes>\n    {VTYPE}\n</routes>\n")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('<configuration>\n  <input>\n'
                    '    <net-file value="taichung.net.xml"/>\n'
                    '    <route-files value="live.rou.xml"/>\n'
                    '  </input>\n  <time>\n    <begin value="0"/>\n'
                    f'    <end value="{END_S}"/>\n  </time>\n</configuration>\n')
        return cfg

    # ---------------------------------------------------------------- setup ---
    def _build(self):
        for s in self.sims.values():
            s.close()
        self.sims, self.worlds = {}, {}
        for p in self.panes:
            k = p["key"]
            sim = (_MockSim(k) if self.mock else
                   _Sim(k, self.cfg, gui=self.gui, step_length=self.step_length))
            self.sims[k] = sim
            self.worlds[k] = SumoWorld(self.r, k, sim)
        self.episode = -1
        self.dropped = 0
        self.finished = False
        self.closed = set()
        self.closed_roads = []
        self._next_episode()

    def _next_episode(self):
        self.episode += 1
        demand, offsets = self.funnel.episode(self.episode)

        def progress(k):
            self.busy = f"episode {self.episode + 1}: planning {k}"
        plans, stats = plan_all(self.r, list(self.worlds), demand, self.closed, progress)
        keep = common_subset(plans)
        self.dropped = len(demand) - len(keep)
        for k, w in self.worlds.items():
            w.plan_stats = stats[k]
            w.last_reroute = {}
            w.adopt(f"e{self.episode}", demand, offsets, plans[k], keep)
        self.busy = ""

    def _run(self):
        while not self._stop.is_set():
            time.sleep(self.tick)
            if self.busy or self.finished:
                continue
            with self.lock:
                # Whole steps only; the remainder carries over so 5x speed at 1 s steps
                # and a 0.25 s tick still averages five steps a second.
                self._acc += self.tick * self.speed
                steps = int(self._acc / self.step_length)
                self._acc -= steps * self.step_length
                for _ in range(steps):
                    # Alternate one step per world so the two stay in lockstep.
                    for w in self.worlds.values():
                        w.step()
                if steps and all(w.spent() >= self.refresh for w in self.worlds.values()):
                    if self.episodes and self.episode + 1 >= self.episodes:
                        self.finished = True
                    else:
                        self._next_episode()

    # ------------------------------------------------------------------------ #
    def geometry(self):
        return self.r.geometry(self._detail)

    def roads(self):
        return [r for r in self.r.roads(limit=8) if r["safe"]]

    def state(self):
        # Never wait on the lock while a re-plan holds it -- FakeBackend.state() explains.
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
                    w.close(edges)
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
        for s in self.sims.values():
            s.close()


# --------------------------------------------------------------------------- #
def selftest(cli):
    """Bring two panes up, drive them, shut a road, check the panes stayed one experiment."""
    router = Router(drl=cli.drl, device=cli.device, beam=cli.beam).warmup()
    panes = PANES if "drl" in router.available() else [
        {"key": "herding", "policy": 4}, {"key": "oracle", "policy": 6}]
    if "drl" not in router.available():
        print("[selftest] no --drl: exercising herding vs oracle instead")
    be = SumoBackend(router, vehicles=cli.vehicles, speed=cli.speed, mock=cli.mock,
                     panes=panes, episodes=0)
    fails = []

    def check(cond, msg):
        print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
        if not cond:
            fails.append(msg)

    def settle(what, limit=120):
        t0 = time.time()
        while be.busy and time.time() - t0 < limit:
            time.sleep(0.2)
        s = be.state()
        check(not be.busy, f"{what}: not busy")
        return s

    s = settle("episode 0")
    keys = list(be.worlds)
    fleets = {k: s["panes"][k]["fleet"] for k in keys}
    print(f"  fleets {fleets}, dropped {s['dropped']}, "
          f"plan {[s['panes'][k]['plan'].get('planned') for k in keys]}")
    check(len(set(fleets.values())) == 1, "both panes drive the same fleet size")
    check(all(v > 0 for v in fleets.values()), "fleet is non-empty")

    print(f"[selftest] driving {cli.drive:g} simulated seconds")
    time.sleep(cli.drive / cli.speed + 1.0)
    s = settle("after driving")
    for k in keys:
        d = s["panes"][k]
        print(f"  {k:<8} ep {d['episode']} t {d['t']:>7} driving {d['driving']:>4} "
              f"pending {d['pending']:>4} arrived {d['arrived']:>4} "
              f"rejected {d['rejected']:>3} worst-rho {d['worst_rho']:.3f}  "
              f"edges loaded {len(d['edges'])}")
        check(d["driving"] + d["arrived"] > 0, f"{k}: cars are moving")
        check(d["rejected"] == 0, f"{k}: SUMO accepted every route at insertion")
        if d["episode"] == 1:
            # Only exact before the first roll-over: from episode 2 on, `driving` also
            # carries the previous episode's stragglers, which `fleet` does not.
            check(d["driving"] + d["pending"] + d["arrived"] + d["stranded"]
                  + d["lost"] + d["rejected"] == d["fleet"],
                  f"{k}: driving + pending + arrived + stranded + lost + rejected == "
                  f"fleet (the subscription flow lost no car)")

    road = cli.close
    print(f"[selftest] closing {road}")
    out = be.close_road(road)
    s = settle("after closure")
    for k in keys:
        rr = out["panes"][k]
        d = s["panes"][k]
        print(f"  {k:<8} active {rr.get('active', 0):>4} routed {rr.get('routed', 0):>4} "
              f"applied {rr.get('applied', 0):>4} deferred {rr.get('deferred', 0):>2} "
              f"setRoute-failed {rr.get('set_route_failed', 0):>3} "
              f"stranded {d['stranded']:>3} gone {rr.get('gone', 0):>3} "
              f"remove-failed {rr.get('remove_failed', 0):>3} "
              f"no_path {rr.get('no_path', 0):>3} {rr.get('seconds', 0):.1f} s")
        if rr.get("first_failure"):
            print(f"           first setRoute failure: {rr['first_failure']}")
        check(rr.get("active", 0) > 0, f"{k}: closure saw active cars")
        check(rr.get("applied", 0) > 0, f"{k}: at least one new route was injected")
        check(rr.get("set_route_failed", 0) == 0,
              f"{k}: SUMO accepted every injected route (deferred mid-junction cars "
              f"are retried by step() and do not count)")
        check(rr.get("gone", 0) == 0 and rr.get("remove_failed", 0) == 0,
              f"{k}: every car we list as alive is known to SUMO "
              f"(otherwise read {be.sims[k].log if hasattr(be.sims[k], 'log') else 'the SUMO log'})")
    check(set(s["closed"]) == set(be.r.road_edges(road)), "closed edges reported")

    time.sleep(2.0 / max(cli.speed, 1) + 0.5)
    s = settle("after more driving")
    check(all(k in s["panes"] for k in keys), "state still has every pane")
    for k in keys:
        d = s["panes"][k]
        w = be.worlds[k]
        print(f"  {k:<8} t {d['t']:>7} driving {d['driving']:>4} arrived {d['arrived']:>4} "
              f"stranded {d['stranded']:>3} lost {d['lost']:>3} still to retry {len(w.retry)}")
        check(not w.retry, f"{k}: every mid-junction car has since been re-routed")

    be.reset()
    s = settle("after reset")
    check(not s["closed"] and not s["closed_roads"], "reset re-opened the road")
    check(all(s["panes"][k]["episode"] == 1 for k in keys), "reset restarted at episode 1")
    be.stop()
    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} problem(s)"
          + (" -- " + "; ".join(fails) if fails else ""))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mock", action="store_true",
                    help="selftest without SUMO: cars advance one edge per step")
    ap.add_argument("--drl", default=None, metavar="CKPT.pt")
    ap.add_argument("--vehicles", type=int, default=800)
    ap.add_argument("--speed", type=float, default=20.0,
                    help="simulated seconds per real second during the selftest. Two "
                         "SUMO instances with 800 cars manage roughly 30 steps/s, so "
                         "asking for more only makes the drive shorter than requested")
    ap.add_argument("--drive", type=float, default=300.0, metavar="SEC",
                    help="simulated seconds to drive before closing the road")
    ap.add_argument("--close", default="臺灣大道", metavar="ROAD")
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    cli = ap.parse_args()
    if not cli.selftest:
        ap.error("this module is imported by app.py --backend sumo; "
                 "run it directly only with --selftest")
    sys.exit(selftest(cli))


if __name__ == "__main__":
    main()
