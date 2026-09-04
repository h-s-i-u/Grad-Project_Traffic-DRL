#!/usr/bin/env python3
"""What both backends must share, or the two panes stop being one experiment.

`FakeBackend` (app.py) and `SumoBackend` (controller.py) differ in where the cars live.
Everything that decides WHICH cars, WHICH trips and WHICH subset is here, so that a pane
driven by SUMO and a pane driven by the fake stepper would still be assigned the same 800
trips in the same order and keep the same common subset. The first version of the demo
kept this inside FakeBackend as a private method, which left the SUMO side no way to get
at it (development log, section 8).
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "integration"))

import config as C                                                    # noqa: E402
import network as net                                                 # noqa: E402

# Left pane vs right pane. 4 and 7 side by side ARE the demo (handover 5.3); 6 is a
# dashed theoretical bound and does not belong on screen as a competitor (5.4).
#
# Keys and policy numbers only -- no display strings. Which panes EXIST is a backend fact
# (there is no policy 7 without a checkpoint); what they are CALLED is not, and a React
# front end would only have to strip them back out again.
PANES = [
    {"key": "herding", "policy": 4},
    {"key": "drl", "policy": 7},
]


class Backend:
    """What index.html needs. Both backends implement exactly these five methods."""

    def geometry(self):
        """{edge_id: [[lat, lon], [lat, lon]]} -- sent once, at page load."""
        raise NotImplementedError

    def roads(self):
        """[{road, edges, km, corridor, safe}] -- the closable-road buttons."""
        raise NotImplementedError

    def state(self):
        """{"panes": {key: snapshot}, "closed": [...], "busy": str}."""
        raise NotImplementedError

    def close_road(self, road):
        """Shut a road in every world and re-plan each under its own policy."""
        raise NotImplementedError

    def reset(self):
        """Re-open everything and start a fresh fleet."""
        raise NotImplementedError


def arena_shapes(verbose=True):
    """Real road shapes from build_geometry.py, or {} -- absence only changes looks.

    Long merged edges draw as straight chords without it; nothing that routes or scores
    reads coordinates, so this is never an error.
    """
    path = HERE / "arena_geometry.json"
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            detail = json.load(f)
        if verbose:
            print(f"[demo] arena_geometry.json: real shapes for {len(detail):,} edges")
        return detail
    if verbose:
        print("[demo] no arena_geometry.json -- long merged edges will draw as straight "
              "lines. Run `python build_geometry.py` once to fix that.")
    return {}


class Funnel:
    """The S2 demand: origins anywhere in the SCC, destinations on the busiest hubs.

    The same funnel run_compare.make_demand uses. That concentration is what produces the
    herding effect, so a demo on uniform demand would have nothing to show.

    Deterministic per episode number: both worlds must get the SAME trips and the SAME
    starting fractions, or the panes stop comparing. The RNG is seeded from a string so
    episode k is the same on every machine and after every reset.
    """

    def __init__(self, router, seed=C.SEED, n=800):
        self.seed, self.n = seed, n
        self.scc = sorted(net.largest_scc(router.g))
        self.hubs = sorted(self.scc, key=lambda x: router.g.in_degree(x),
                           reverse=True)[:C.N_HOTSPOTS]

    def episode(self, k):
        """([(origin_index, dest_index)] * n, [start fraction in 0..0.8] * n)."""
        rng = random.Random(f"{self.seed}-episode-{k}")
        demand = [(rng.choice(self.scc), rng.choice(self.hubs)) for _ in range(self.n)]
        return demand, [rng.random() * 0.8 for _ in range(self.n)]


def plan_all(router, keys, demand, closed=(), progress=None):
    """Assign one episode under every policy: ({key: [node path or None]}, {key: stats}).

    One BATCH per policy, which is where policies 6 and 7 differ from 4 at all: eq. 4
    couples vehicles through the load they leave behind, and a vehicle routed on its own
    reads an empty network whatever policy is named (World.adopt in app.py has the story).
    """
    plans, stats = {}, {}
    for k in keys:
        if progress:
            progress(k)
        plans[k] = router.plan(demand, policy=k, closed=sorted(closed))
        stats[k] = dict(router.last_stats,
                        planned=sum(1 for p in plans[k] if p and len(p) >= 3))
    return plans, stats


def common_subset(plans):
    """Indices of the trips EVERY policy could route -- the only trips either pane drives.

    Keeping each world's own successes gave the panes different fleet sizes (768 against
    641 under a closure, because greedy decoding dead-ends), and a pane with 17% fewer
    cars carries less load and posts a better worst-rho for a reason that has nothing to
    do with its policy. The report scores the same way, on the trips all policies
    completed.
    """
    n = len(next(iter(plans.values())))
    return [i for i in range(n) if all(p[i] and len(p[i]) >= 3 for p in plans.values())]
