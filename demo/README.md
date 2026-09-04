# demo/

A web application that renders two routing policies side by side on the Taichung arena
and lets a road be closed at runtime.

```bash
pip install fastapi uvicorn
cd demo
python app.py --drl ../integration/checkpoints/taichung/drl_fusion_togo25.pt
# http://127.0.0.1:8000
```

Runs without SUMO. Without `--drl` the second pane is omitted.

## Files

| File | Contents |
|---|---|
| `app.py` | FastAPI application and `FakeBackend` — two fleets advancing on a background thread with no simulator |
| `controller.py` | `SumoBackend` — the same five methods, with the cars in two SUMO instances driven over TraCI. `--backend sumo` imports it. Has a `--selftest` that runs with or without SUMO |
| `shared.py` | What both backends must agree on: the `Backend` contract, `PANES`, the episode demand generator, the per-policy batch assignment and the common subset |
| `index.html` | One static page: two synchronised Leaflet maps, the road buttons, and eight figures per pane. No build step |
| `build_geometry.py` | Recovers the real road shape of every arena edge → `arena_geometry.json` |
| `arena_geometry.json` | Committed, because the script's inputs are release artifacts rather than repository files. See below |

## `app.py`

Four endpoints, all plain JSON:

```
GET  /geometry  -> {"edges": {edge_id: [[lat,lon], ...]}, "panes": [...], "roads": [...]}
GET  /state     -> {"panes": {key: snapshot}, "closed": [edge_id],
                    "closed_roads": [name], "dropped": int, "finished": bool,
                    "busy": str, "window_s": int}
POST /close     {"road": "<name or prefix>"}   close it in every world and re-plan each
POST /reset     re-open everything, rebuild the fleets
```

`/geometry` is fetched once; `/state` is polled at 1 Hz. A snapshot carries `t`, `fleet`,
`driving`, `arrived`, `stranded`, `episode`, `worst_rho`, `gini_load`, `frac_saturated`,
`plan`, `last_reroute`, and `edges` as `{edge_id: rho}` for every edge carrying load.

`plan` and `last_reroute` are kept apart deliberately. `plan` is the episode's assignment
— of the 800 trips demanded, how many *this* policy could route. `last_reroute` is a
closure — of the cars still driving, how many got a new route. Sharing one field once
printed `798/436`: a numerator from the dispatch over a denominator from the re-route.

### `Backend`

Five methods — `geometry`, `roads`, `state`, `close_road`, `reset` — defined in
`shared.py`. `FakeBackend` is one implementation; `SumoBackend` in `controller.py` is the
other, reached with `--backend sumo` and constructed as `SumoBackend(router, vehicles=,
speed=)`. Neither `app.py` nor `index.html` knows which one is running.

Demand generation lives in `shared.py` too, on purpose: episode *k* is the same 800 trips
and the same start fractions whichever backend asks for it, and the common subset is
computed by the same function. A pane driven by SUMO and a pane driven by the fake
stepper are therefore still one experiment.

### `FakeBackend`

Two `World`s over identical demand, one per policy, advancing in simulated seconds on a
background thread.

**Episodic, not continuous.** An episode is 800 vehicles assigned in one batch, each world
under its own policy. When `--refresh` (default 0.85) of the fleet has finished, both
worlds dispatch the next episode together.

The batch is the point. eq. 4 couples vehicles through the load they leave behind —
vehicle *k* routes around what 0..*k*−1 already filled — so an assignment is the only
place policies 6 and 7 differ from policy 4 at all. An earlier version replaced arrivals
one at a time; that hands every vehicle an empty network to read, and the two panes
converge whatever policy each one names.

Both panes drive the **same** trips: those every policy could route. Keeping each
world's own successes gave them different fleet sizes — 768 against 641 under a closure,
because greedy decoding dead-ends — and a pane with 17% fewer cars carries less load and
posts a better worst-rho for a reason that has nothing to do with its policy. The
excluded count is shown in the header.

Other behaviour worth knowing:

- Vehicles traverse edges at the **BPR** time `t0 · (1 + 0.15·(load/cap)⁴)`, the same
  volume-delay function the offline evaluation uses, so a loaded corridor slows its own
  traffic.
- The fleet starts **partway along** its routes, at the same fraction in both worlds, so
  the first frame is already loaded and the panes stay comparable.
- A vehicle already committed to an edge when that edge closes stops there and is counted
  as `stranded` rather than removed, because served-fraction is a reported metric.
- `arrived` and `stranded` reset each episode. A counter that runs past the fleet size
  cannot be read against a reported served-fraction.
- It is **not** a microscopic traffic simulation: no car-following, junctions, or signals.
  Nothing this page displays is a result; the reported figures come from
  `integration/run_compare.py`.

### Flags

| Flag | Default | Note |
|---|---|---|
| `--vehicles` | 800 | `TAICHUNG_CAPACITY_SCALE = 0.0429` was calibrated at 800. At 300 no edge crosses `RHO_THRESHOLD = 0.85`, eq. 4's saturation term is identically zero, and the two panes converge |
| `--speed` | 5 | Simulated seconds per real second |
| `--refresh` | 0.85 | Fraction of the fleet that must finish before the next episode. Lower means fresher traffic and more frequent pauses while policy 7 re-plans |
| `--episodes` | 0 | Stop after N episodes and hold the final frame. 0 keeps going, which is what a running demo wants; 1 is how to read a served-fraction off the page |
| `--beam` | 0 (greedy) | Decoding width for policy 7. Greedy plans 800 vehicles in ~6 s on GPU; beam-8 is 8–10× that and recovers most of the dead-ends |
| `--device` | auto | `cuda` when available; CPU is ~1.6× slower per vehicle |
| `--backend` | fake | `sumo` imports `controller.SumoBackend` |

## `index.html`

Leaflet with the canvas renderer — 1,690 polylines per pane restyled once a second is past
what the SVG renderer keeps smooth, and there are two panes. Only edges whose saturation
actually changed are restyled.

The colour ramp breaks at 0.85 because that is `RHO_THRESHOLD`, where eq. 4's saturation
term begins to fire. **Closed roads are magenta dashes** — a colour nowhere on the ramp,
which runs blue → green → amber → orange → red, because red already means worst-case
congestion and "shut" is a different fact. A closed edge keeps that style at zero load,
or the road just closed would fade to the same grey as every road nobody is using.

Display strings live in the `LABEL` table in this file, not in the API.

## `build_geometry.py`

An arena edge is a **merged chain**: simplification collapses runs of pass-through nodes,
twice, and only the two endpoints reach `arena_edges_taichung.csv`. Anything drawing the
network from that file draws every road as a straight chord. Over all 1,690 edges, real
length / chord is 1.000 at the median — but the exceptions are the long ones, and one of
them is 4,279.5 m of road rendered as a 3,391 m straight line across the city.

Recovery is **not** by shortest path: the chain that was merged is not generally the
shortest route between its endpoints, so a plain Dijkstra returns a different road and the
length check throws it away (that approach recovered 72%, failing on essentially every
edge over 1 km). The constraint that works is exact — a node is merged away precisely when
it is a pass-through, so the interior of a merged edge consists of nodes present in the
parent graph and absent from the child:

```
interior of an arena edge       ⊂  simplified nodes − arena nodes
interior of a simplified edge   ⊂  Map_fined nodes  − simplified nodes
```

Two stages, one per merge, each a Dijkstra forbidden to route *through* any node of the
child graph, and each result checked against the edge's own `length_m`. 1,690 of 1,690 are
recovered; the median length residual is 0.0000%.

```bash
python build_geometry.py --dfs-cap 5000000 --loose
```

`--loose` accepts a best-effort shape for edges no chain of the right length was found for
— real roads between the right endpoints, but possibly part of the wrong street. Every use
is listed with its length ratio. Two edges (one road, both directions) need it.

**`arena_geometry.json` is committed.** The script reads `Map/simplified_*` and
`Map/Map_fined/`, neither of which is in the repository, so a fresh clone cannot
regenerate it.

## `controller.py`

```bash
pip install eclipse-sumo traci sumolib            # binaries on PATH plus the two packages
cd ../integration
python export_sumo.py --drl checkpoints/taichung/drl_fusion_togo25.pt   # writes sumo/
cd sumo && sh build_net.sh && cd ../../demo                            # taichung.net.xml, once
python controller.py --selftest --mock            # no SUMO: the bookkeeping only
python controller.py --selftest --drl ../integration/checkpoints/taichung/drl_fusion_togo25.pt
python app.py --backend sumo --drl ../integration/checkpoints/taichung/drl_fusion_togo25.pt
```

**One SUMO instance per pane.** A single simulation cannot carry two policies, because
policy 4 and policy 7 produce different traffic states; `SumoBackend` starts one `sumo`
process per pane and steps them in lockstep from one thread. Each step is one simulated
second; `--speed` is how many of them happen per real second.

Everything SUMO-specific is inside the `_Sim` class (start, step, add, where, set_route,
remove). `_MockSim` implements the same calls with cars that advance one edge every ten
steps, which is what `--selftest --mock` runs against — it exercises episodes, the
common subset, closure, route injection, stranding, the mid-junction retry and the
snapshot, and nothing below the TraCI line.

| Variable | Default | Meaning |
|---|---|---|
| `DEMO_SUMO_DIR` | `../integration/sumo` | where `taichung.net.xml` is. The backend writes `live.rou.xml` (the vType only) and `live.sumocfg` next to it |
| `DEMO_SUMO_GUI` | `0` | `1` opens two `sumo-gui` windows instead of headless `sumo` |
| `DEMO_SUMO_STEP` | `1.0` | simulation step length in seconds |

Four things it does differently from `FakeBackend`, on purpose:

1. **Cars are inserted, not placed.** SUMO cannot put a car mid-route, so the route is
   cut at the same start fraction both panes use and the car departs at its head. Up to
   800 insertions at t = 0 queue on their first edges; `pending` in the snapshot is how
   many have not entered yet.
2. **A car heading into the closure with no legal route is removed at closure time** and
   counted as `stranded`. The fake stepper strands the same set when they reach the
   closed edge; SUMO would instead teleport them across it after `--time-to-teleport`,
   which is neither honest nor visible. Teleporting is switched off (`-1`).
3. **Cars from the previous episode keep driving** when the next one is dispatched.
   Removing them would make cars vanish on screen. They count in `driving`, never in the
   new episode's `arrived`, `stranded` or `fleet`.
4. **The closure is not written into SUMO's permissions.** `edge.setDisallowed()` on the
   closed edges is the obvious move, and the first two live runs did it. It refused
   20–40 re-routes per pane with `No connection between edge A and edge B`, where B was
   always a closed edge that the *new* route never contained: SUMO validates a
   replacement route from the beginning of the car's driven history, which it keeps in
   front of the new edges, so any car that had used 臺灣大道 earlier in the episode was
   refused a new route once the road was disallowed — wherever it was by then. The
   closure lives in the router, which masks the edges, and in rule 2. What a visitor
   sees is the same: no new car enters the road and the ones on it drive off. `sumo-gui`
   does not paint the road shut; the web page does.

Two snapshot fields exist only here: `pending` (above) and `rejected` — routes SUMO
refused at insertion, which happens only if `netconvert` dropped an edge. It should be 0;
the self-test checks that it is, because a car rejected in one pane and accepted in the
other silently un-pairs the experiment.

Load is counted the way ρ is defined — edge *entries* over the last ~154 s — from one
road-id snapshot per step obtained through TraCI subscriptions (three calls per step
regardless of fleet size, against 800 `getRoadID()` round trips). ATT is not on the
panel: offline it is the BPR estimate, live the only true value is what SUMO measured,
and `tripinfo` is where a later version should read it from.

**Two live runs so far (4–5 Sep 2026), 734 cars per pane.** Verified: every exported
edge id survives `netconvert` (`rejected 0`), the subscription flow loses no car
(driving + arrived = fleet), `setRoute()` is accepted on cars in motion. The refusals
were the permission problem above — misread the first time as `netconvert` dropping
turns at sharp chord angles, which led to an explicit-connections change in
`export_sumo.py` that has been reverted: an offline probe showed the router's routes
never contain a closed edge, so the edges SUMO named could only have come from the
driven history. The `Vehicle 'X' is not known` errors on those runs were the stale
subscriptions of cars the backend itself had just removed, reported once each on the
next step; `remove()` now unsubscribes first. One car per closure was refused for being
mid-junction (`Vehicle is on junction-internal edge leading elsewhere`); such cars are
`deferred` and re-routed by `step()` once they are back on a normal edge. A third run
on the reverted network is what closes this section.

### The routing contract

Routing goes through `integration/reroute_service.py`:

```python
routes = router.reroute(active, closed, policy="drl")
#   active = [(veh_id, current_edge_id, dest_osmid, remaining_edge_ids), ...]
#   closed = [edge_id, ...]
#   -> {veh_id: [edge_id, ...]}

paths  = router.plan(demand, policy="drl", closed=closed)
#   demand = [(origin_index, dest_index), ...]  -> [node path or None]
#   for assigning a whole fleet at once, which is what eq. 4 needs
```

Every route `reroute()` returns begins with the vehicle's current edge, which `setRoute()`
requires. A vehicle absent from the result was not re-routed and must be left alone.
`python ../integration/reroute_service.py` exercises all of this against fabricated state
with no simulator present.

Three things that fail quietly rather than loudly:

1. **Do not pass SUMO's load to `reroute()`.** It belongs in `network_state(load)`, which
   feeds the display. Seeding the router with it drops policy 7 from 737/800 served to
   345/800 while leaving policies 1/4/6 untouched.
2. **`load` must count edge *entries over the last ~154 s*,** not
   `traci.edge.getLastStepVehicleNumber()`. `cap` is veh/h × 0.0429, i.e. vehicles per
   0.0429 h, so ρ is a flow ratio over that window. `reroute_service.LoadWindow` maintains
   it from a per-step `{veh_id: traci.vehicle.getRoadID(veh_id)}` snapshot.
3. **Edge ids are `<from_osmid>_<to_osmid>` on both sides.** They match because
   `integration/export_sumo.py` writes the `.nod`/`.edg.xml` and the routes from the same
   graph. Re-extracting from OSM would have `netconvert` invent its own ids and reintroduce
   a mapping table.

## Offline use

The page loads Leaflet from a CDN, so it needs network access. To remove that:

```bash
mkdir -p vendor && cd vendor
curl -LO https://unpkg.com/leaflet@1.9.4/dist/leaflet.js
curl -LO https://unpkg.com/leaflet@1.9.4/dist/leaflet.css
```

then re-point the two tags at the top of `index.html`. `app.py` checks for these at startup
and prints the commands if they are missing. Map *tiles* also come from the network; losing
them leaves the basemap blank while every road still draws, since the polylines come from
the project's own graph.

## Replacing the front end

The API is framework-agnostic JSON over HTTP. Swapping in a React front end changes:

| | |
|---|---|
| Unchanged | `reroute_service.py`, any `Backend`, all four endpoints, every JSON shape |
| Two lines | `GET /` in `app.py`: `FileResponse` → `StaticFiles` over the build output |
| Possibly three lines | CORS middleware, if the dev server runs on its own port |
| Rewritten | `index.html` only |
