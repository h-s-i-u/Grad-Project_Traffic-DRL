# `integration/` — prediction → decision dataflow

This folder wires the traffic-prediction models to the routing decision layer. It builds
a congestion-weighted road graph from the forecasts and compares routing policies,
measuring whether congestion-aware + global-penalty routing suppresses the **herding
effect (羊群效應)** — the core claim of the project.

Two road networks run through the same code path:

| `--graph` | network | edge times | speed source |
|---|---|---|---|
| `metr-la` (default) | 207-sensor Gaussian kernel | relative units | `stg{cn,at}_pred.npy` |
| `taichung` | **the 840-node routing arena** carved out of the real OSM network | **seconds** | `taichung_pred_edges.csv` |

The decision stage is pure NumPy + NetworkX + SciPy (no GPU); only the DRL agent needs
PyTorch + `torch_geometric`, and even that runs on CPU at inference.

---

## How to run

```bash
cd integration

# --- METR-LA ---
python run_compare.py                                   # seven-policy ablation
python run_compare.py --repeat 10 --drl checkpoints/metr-la/drl_agent.pt

# --- Taichung ---
python taichung_loader.py                 # load check + graph stats
python calibrate_taichung.py              # capacity sweep (re-run if vehicle count changes)
python make_drl_input.py --source fusion  # forecasts -> per-edge speeds

# S2, the rush-hour funnel (main scenario)
python run_compare.py --graph taichung --vehicles 800 --repeat 10 \
       --drl checkpoints/taichung/drl_fusion_togo25.pt --beam 8

# S3, an arterial closes partway through the demand
python run_compare.py --graph taichung --vehicles 800 --close-list   # what is worth closing
python run_compare.py --graph taichung --vehicles 800 --repeat 10 \
       --drl checkpoints/taichung/drl_fusion_togo25.pt \
       --close-road 臺灣大道 --close-at 0.10 --beam 8

# eq. 4 weight sweep for policy 6 (the proposal's grid search; log §13.27)
python sweep_lambda.py                                    # lambda2, 10 seeds per cell
python sweep_lambda.py --lambda1 0.5,2.0 --lambda2 0.8,6.0 --out lambda_cross.json
python sweep_lambda.py --scenario random --lambda2 0.8,12.0 --out lambda_random.json

# --- training and diagnosis ---
python train_drl.py --graph taichung --iters 800 --train-vehicles 800 \
                    --eval-vehicles 800 --fail-mult 10.0 --togo-refresh 25 \
                    --out checkpoints/taichung/drl_fusion_togo25.pt
python diagnose_agent.py --drl <ckpt-a> --drl <ckpt-b>   # several agents, ONE run
python test_beam.py --drl <ckpt> --vehicles 800 --widths 1,2,4,8

# --- inference latency (proposal section 5: < 50 ms per decision) ---
# Run BOTH devices: the encoder is throughput-bound and the decoder is launch-bound,
# so which device wins depends on how many times you decode per vehicle.
python bench_latency.py --graph taichung --vehicles 800 \
       --drl <ckpt> --beam 8 --verify --scale
python bench_latency.py --graph taichung --vehicles 800 \
       --drl <ckpt> --beam 8 --device cpu --scale

# --- prediction-side baselines (no checkpoint, no GPU) ---
python ha_baseline.py
python search_ensemble_weight.py --n-pred 12 --split test
```

---

## Files

| file | role |
|---|---|
| `config.py` | paths + every hyper-parameter (single source of truth) |
| `network.py` | `build_graph_for(name)` — one entry point for both networks |
| `pipeline.py` | METR-LA only, written in June and not re-run since: reruns both models' inference into `stg{cn,at}_pred.npy`, then `run_compare.py` (`--skip-infer` reuses the files) |
| `taichung_loader.py` | Taichung CSVs → routing DiGraph; reads predicted edge speeds and `road_name` |
| `policies.py` | the seven policies + `RoutingEnv` / `EGATActorCritic` / `PPOTrainer` / beam decoding |
| `closure.py` | the S3 arterial closure: what to close, when, and what to do with the trips it strands |
| `metrics.py` | ATT, TSTT, Gini(edge load), worst-link ρ, throughput |
| `run_compare.py` | run every policy on shared demand; multi-seed statistics; S3 flags; `--beam` |
| `sweep_lambda.py` | eq. 4 weight sweep for policy 6, scored against one fixed Gini reference edge set (two passes, so the denominator cannot move with λ₂); writes `lambda_{sweep,cross,random}.json` |
| `train_drl.py` | train the PPO + Residual E-GAT agent, save the best checkpoint |
| `make_drl_input.py` | forecasts → per-edge speeds, with a same-instant guard |
| `calibrate_taichung.py` | find the (vehicles, capacity_scale) where congestion actually appears |
| `diagnose_agent.py` | split the agent's ATT gap into detour / congestion / attrition |
| `test_beam.py` | **beam width 1 must equal greedy** — run before trusting any beam number |
| `bench_latency.py` | inference latency: per-hop / per-vehicle / per-fleet, decomposed by component, with the Dijkstra baselines in the same table. Installs its timers on the *instances*, so `policies.py` is untouched; `--verify` proves the timing loop still reproduces `policy_drl` |
| `ha_baseline.py` | Historical-Average floor + anomaly-bucket breakdown (proposal §4.6) |
| `search_ensemble_weight.py` | fixed-weight sweep + error correlation |
| `taichung_pred_edges.csv` | per-edge speeds — what the router consumes (`.meta.json` records its source) |
| `data/adj_mx_dijsk.pkl` | vendored METR-LA adjacency |
| `reroute_service.py` | the resident router the live demo calls: `reroute()` for cars in motion, `plan()` for a whole fleet, `network_state()` for the panel; `policies.py` untouched. Routes from an empty network unless `seed_load=` is passed (see the starting-load note below) |
| `export_sumo.py` | arena + routed vehicles as SUMO inputs. Edge ids are `<from_osmid>_<to_osmid>` on both sides; real road shapes from `demo/arena_geometry.json`; `--check-net` compares every turn the graph allows with the built `.net.xml` (`build_net.sh` is two-pass) |
| `run_sumo_compare.py` | **replay validation (mode A)**: the report's S2/S3 assignments, beam-8, 10 seeds, replayed headless in SUMO and scored |
| `sumo_metrics.py` | ATT / served / worst-ρ / Gini from `tripinfo` and `edgeData`, defined as the report defines them; paired deltas vs the herding baseline |

### SUMO replay validation (mode A)

```bash
python export_sumo.py --drl checkpoints/taichung/drl_fusion_togo25.pt     # network + shapes, once
(cd sumo && sh build_net.sh)                                              # netconvert + turn check
python run_sumo_compare.py --drl checkpoints/taichung/drl_fusion_togo25.pt
python run_sumo_compare.py --drl checkpoints/taichung/drl_fusion_togo25.pt \
       --scenarios s2 --windows 154 --periods 5                          # sustained flow
```

Assignments are made exactly as `run_compare.py` makes them (same demand generator,
seeds 42–51, S3 = 臺灣大道 closed at 10% of the dispatch, beam-8 for policy 7); SUMO
replays each policy's routes and measures. Policies 1, 4, 6 and 7 are replayed. Every
junction is priority-controlled with no signals, teleporting is off (a stuck car stays
stuck), and no vehicle reroutes: each drives the route it was assigned. Departures are
spread over 154 s — the demand period `capacity_scale` implies (實驗設計 §4.4) — with
600 s as the sensitivity case; routes are exported once per seed and the second window
rescales the departure times. Output: one table per scenario × window in `sumo_a/`, plus
`results.json` (`results_p5.json` for the sustained run). Absolute values
are not comparable with the BPR-side numbers (the assignment puts every vehicle on the
road at once); the paired deltas against the herding baseline are. Resumable: finished
exports and runs are skipped.

`--periods K` replays the same routes K times, each period one window later (S2 only):
a static assignment describes a sustained flow, and one pulse of 800 vehicles disperses
before it reaches the busy edges. `sumo_metrics.py` then scores the middle periods
separately (`ATT mid`, `served mid`).

---

## Data flow

```
                     METR-LA                          Taichung
           stgcn_pred.npy ─┐            fusion/  ──►  dump_fusion_test_p3.npz
           stgat_pred.npy ─┴─ ensemble  STGCN ──┐
                            0.2/0.8     STGAT ──┴──►  taichung_pred_{stgcn,stgat}.npy
                               │                              │
                               │                     make_drl_input.py
                               │              (same-instant guard, clamp, section → edge)
           adj_mx_dijsk.pkl    │           Map/arena_*_taichung.csv       │
           d ∝ √(−ln adj) ─────┤           length_m / speed limit ────────┤
                               ▼                                          ▼
                      network.build_graph_for(name)  →  directed road graph
                         edges carry: length · t0 · tpred · tpred_stgcn ·
                                      tpred_stgat · cap · road_name
                                           │
                 OD demand — "hotspot" funnels many cars at a few hub nodes,
                          which is what creates the herding pressure
                                           │
                          (optional) closure.py masks an arterial
                                    partway through the demand
                                           │
 ┌──────┬────────┬────────┬─────────┬────────────┬──────────────┬───────────────────┐
1 static 2 STGCN  3 STGAT  4 hybrid  5 load-aware 6 global-      7 DRL
 (free-  +Dijkstra +Dijkstra +Dijkstra  (coord.)     penalty       (PPO + E-GAT,
  flow)                     (HERDING)                (oracle)      greedy | beam-8)
 └──────┴────────┴────────┴─────────┴────────────┴──────────────┴───────────────────┘
                                           ▼
     every policy is re-scored under the SAME realized BPR congestion model:
     ATT · TSTT · Gini(edge load) · worst-link ρ · throughput · served%
```

---

## The seven policies

A deliberate ablation ladder. Each step adds one capability, except 4 → 5, which drops
the forecast and adds coordination at once; that is harmless here because 1–4 turn out
indistinguishable (see Results):

| # | policy | prediction? | coordination? | eq. 4? | maps to |
|---|---|:--:|:--:|:--:|---|
| 1 | `static` (free-flow Dijkstra) | ✗ | ✗ | ✗ | proposal baseline ① |
| 2 | `STGCN + Dijkstra` | ✓ STGCN only | ✗ | ✗ | proposal baseline ② |
| 3 | `STGAT + Dijkstra` | ✓ STGAT only | ✗ | ✗ | proposal baseline ③ |
| 4 | `hybrid + Dijkstra` | ✓ gated fusion | ✗ | ✗ | **the herding case — Δ is measured against it** |
| 5 | `load-aware` | ✗ | ✓ | ✗ | ablation: isolates coordination |
| 6 | `global-penalty` | ✗ | ✓ | ✓ | **oracle / upper bound** |
| 7 | `drl-agent` | ✓ observation only | ✓ | ✓ | **the method** (proposal baseline ④) |

Policies 5 and 6 assign the fleet in 30 batches and price edges from `t0` through BPR,
never from the forecast. Policy 7 sees the forecast only as an input feature; its reward
and its distance-to-go are built from `t0` as well.

Two mechanisms make this work:

- **BPR volume-delay** `t(load) = t₀(1 + 0.15·(load/cap)⁴)` — piling vehicles onto one
  link inflates its time. Without it the herding effect does not exist numerically.
  It stays the model every policy is trained and scored in; SUMO only replays the
  resulting routes (see *Replayed in SUMO*).
- **eq. 4 generalized cost** — `α·t_e(load) + κ·[λ₁·max(0, ρ−ρ_th)² + λ₂·max(0, ρ−ρ̄)]`,
  the proposal's global penalty rewritten as an edge weight, with κ = 12·t_ref (the mean
  free-flow edge time), α = 1, λ₁ = 0.5, λ₂ = 0.8, ρ_th = 0.85. Policy 5 uses only the
  first term, policy 6 the whole cost, and policy 7 is rewarded with its negative per hop.

> **(6) is the oracle, not the method.** It routes every vehicle with a full Dijkstra
> over the exact global load, pricing edges with the arena's own cost function, so it is
> the direct optimiser of what policy 7 learns to approximate. It is not slow: one request
> takes 0.31–0.33 ms, 43–68x faster than the agent's greedy decoding (log §13.25 ④).
> Treat it as the **upper bound for the eq. 4 objective**, not for every metric (see the
> sustained SUMO replay below).

Policy 7 is reported **twice**, under greedy and under beam-8 decoding. Same weights,
same observations; only the search differs, which is why Lei et al. (2022) report the
pair rather than treating beam as a separate method. Greedy stays in the table because
the pair *is* the ablation.

---

## Three guards, and why each exists

Every one of these was added after a failure that produced plausible numbers and no
error message.

**1. `served%` leads the table, and anything under 95% is flagged NOT COMPARABLE.**
A policy that abandons vehicles is scored only on the ones that arrived, and the ones it
abandons are the hard ones. Measured under S3: greedy decoding posted **ATT −63.0%
against the oracle's −61.2%** — it appeared to beat the theoretical upper bound, purely
because it gave up on 17% of the demand. The inflation scales with the attrition rate
(4.6% abandoned → 2.4pp; 17.2% → 7.2pp).

**2. `.meta.json` sidecars travel with every artefact whose setting changes an input.**
`train_drl.py` writes one beside each checkpoint (`togo_refresh`, `reward_scale`, the
multipliers, `capacity_scale`); `make_drl_input.py` writes one beside
`taichung_pred_edges.csv` recording which forecast produced it. Two CSVs from different
sources are indistinguishable on disk, and which one an agent trained against decides
whether its checkpoint is still valid. `run_compare.py` and `diagnose_agent.py` read
these back: `togo_refresh` is applied automatically, a graph-name mismatch is warned
about, and the rest is printed for a manual check.

**3. `congestion_note()` warns when the herding baseline's worst-link ρ leaves 1.5–5.0.**
Below that nothing is saturated and every policy routes near-freely, so the deltas are
noise; above it everything is jammed. Neither is a code error, which is why it needs
saying out loud.

And one test that is not a guard but a precondition: **`test_beam.py` checks that beam
width 1 reproduces greedy decoding route for route.** The beam decoder rebuilds the
candidate features, the eq. 4 cost, the visited set, the closure mask and the per-hop
load bookkeeping. Two of those were wrong on the first attempt and neither raised —
one of them left the environment 81 edges lighter, which flipped a decision whose two
options were 0.52 apart.

---

## Reporting protocol

A single demand draw varies too much to conclude from, so `--repeat N` draws N demands
and reports **mean ± std** with **paired** deltas (computed within each seed, then
averaged — this cancels the demand-to-demand variance that hits every policy alike).
The ± covers demand sampling only: the agent was trained once (seed 42), so training
variance is not in it.

```bash
python run_compare.py --graph taichung --vehicles 800 --repeat 10 \
       --drl checkpoints/taichung/drl_fusion_togo25.pt --beam 8
```

Gini is consistently the tightest metric — ±0.6% across demand draws against ±8% for
ATT, on both networks — so its small deltas (−10.1% for policy 7 against −3.3% for
coordination alone) are still far outside the noise.

**Gini is computed over the union of edges any policy used**, so adding a policy row
shifts every row's Gini slightly. ATT and worst-link ρ are unaffected. Compare Gini only
within one run.

---

## Results

**METR-LA, hotspot, 300 vehicles, one seed** — Δ vs the herding baseline: ATT
**−33.7%**, worst-link ρ **−93.0%**, Gini **−35.0%**, above all three of the proposal's
targets on that single draw. That is the *similarity graph*, not a road network: 76
neighbours per node against Taichung's 2.01, which is why its numbers are so much larger
and why they say nothing about the arena below. (Measured in June, when `config.py`
mixed the METR-LA ensemble 0.7/0.3; it now ships 0.2/0.8 (log §13.7), so a rerun hands
baseline 4, the denominator, a different forecast and policy 7 a different input.)

**Taichung, 800 vehicles, 10 demand draws, beam-8 decoding.** These are BPR-model
measurements; the last column is what survived being replayed in SUMO (`sumo_metrics.py`,
`run_sumo_compare.py`):

| Δ vs herding baseline | S2 (rush-hour funnel) | S3 (arterial closure) | target | in SUMO |
|---|---:|---:|---|---|
| ATT | **−36.0±7.9%** | **−55.8±5.6%** | ↓20–30% | **withdrawn** — see below |
| worst-link ρ | **−24.6±2.0%** | **−27.5±2.9%** | ↓20% — met | reproduced below saturation |
| Gini(edge load) | −10.1±0.6% | −12.9±0.8% | ↓30% — missed | an identity of the routes |
| served% | 99.9±0.2 | 96.0±1.8 | ≥95% — met | **81% → 97% under sustained flow** |

Gini falls short — and so does the oracle (−18.1% / −17.4%), so the gap is not a
training problem. Nor is it a tuning one: raising λ₂ fifteen-fold (0.8 → 12) takes the
oracle only to −19.1%, with returns that shrink geometrically and extrapolate to about
−22% (`sweep_lambda.py`, log §13.27; scored on a fixed 1,501-edge reference set, so not
digit-for-digit comparable with the table). The limit is topology. The same objective
reaches −50.7% on METR-LA, where the agent reached −35.0%, but that graph gives each node
76 neighbours and a 1.5-hop trip against the arena's 2.01 exits and 28.7 hops, and
spreading load needs parallel roads. On the arena, switching to random demand moves the
oracle by 3 pp; switching graphs moves it by 32 pp. (An earlier explanation, four hubs
whose last hops no policy can avoid, does not hold: their 16 entry edges carry 3.2% of
all edge traversals.)

**Policies 1–4 are indistinguishable.** Identical worst-link ρ on every S2 seed (under
S3, eight of ten), and under S3 the ATT spread between them (1.3–1.7%) is smaller than
its own seed-to-seed variance (±2.1%). Whichever forecast you follow, greedily chasing the
predicted-fastest road herds just as badly — the improvement comes from *coordination*
and the *global penalty*, not from prediction quality. That is the project's central
claim, isolated about as cleanly as it can be.

### Replayed in SUMO

`run_sumo_compare.py` drives the same assignments, vehicle by vehicle, in a microscopic
simulator — a world whose cost function is not the one any policy was built on. Ten
seeds, paired against the herding baseline:

| | single pulse of 800 | sustained: the same fleet every 154 s, 5 periods |
|---|---:|---:|
| policy 7, served | 99.9% | **97.3%** (herding 81.3%, oracle 92.7%) |
| policy 7, served Δ | — | **+16.0 ± 13.3 pp** (oracle +11.4 ± 11.7) |
| policy 7, ATT Δ (common trips) | +3.7 ± 1.0% | +0.7 ± 7.3% (oracle **−10.2 ± 5.7%**) |
| policy 7, worst-ρ Δ | −15.1 ± 10.9% | +1.6 ± 1.5% |

**The spread transfers; the travel-time saving does not.** A single pulse of 800 vehicles
never congests the microscopic world — the busiest edge sits at ρ 0.84 per window against
the assignment's 3.33, because departures spread out before they reach it — so a detour
taken to spread load is pure cost and free-flow shortest paths are the fastest of all.
Under sustained flow congestion does appear, but as *gridlock rather than delay*: nearly
a fifth of the herding baseline's trips never arrive within two hours, while policy 7
delivers 97% of them. Travel time is not comparable across the two worlds; served
fraction is, and it is the result that carries over. worst-ρ discriminates only below
saturation (above it every policy is capped near 1.05 by physical throughput), and Gini
comes out identical at both departure windows because an edge's total entries are fixed
by the routes that cross it — an identity, not evidence. (It differs slightly from the
BPR table only because the reference set holds the four replayed policies and SUMO's
`entered` count leaves out each vehicle's departure edge.)

Policy 7's 97.3% against the oracle's 92.7% is seed variance, not a win: paired per
seed the gap is +4.6 ± 9.5 pp (5 seeds higher, 3 tied at 100%, 2 lower). Delivery
hinges on whether a seed gridlocks, which happened in 8 of 10 seeds for herding, 6 for
the oracle and 1 for policy 7 (seed 45, 73.9% delivered); why the oracle gridlocks more
often is not established (log §13.31 ⑧). The oracle optimises BPR's cost, not SUMO's
delivery, and the two keep the network moving about equally well.

---

## Notes and limitations

- **METR-LA travel times are relative** (the kernel's σ is unknown). Compare ratios,
  not absolute seconds. Taichung times are real seconds.
- **The router runs on the arena, not the whole city.** On the full 27,022-edge network
  only 601 edges (2.2%) carry a forecast, so `tpred == t0` almost everywhere and
  baselines 2–4 collapse onto 1 — which would hollow out every delta, since 4 is the
  denominator. `TDX_Data/build_arena.py` carves out 840 nodes / 1,690 edges at 24.6%
  coverage. Node ids stay the original OSM ids, so routes still render on the full map.
- **Prediction has no measurable effect on routing here**, and this is a data limit
  rather than a bug. Coverage *by length* — what a shortest-path search actually weighs
  — is 14.1%, so unobserved edges all take the same fallback multiplier and a uniform
  scaling cannot reorder paths. Measured four independent ways: raising coverage by
  edge count 1.6-fold (15.2% → 24.6%) changed nothing; 86% of routes are identical under `t0` and `tpred`;
  swapping in a genuinely different forecast (gated fusion, correlation 0.985 with the
  fixed-weight mix) left baselines 2–4 on top of each other; and retraining the agent on
  that forecast moved every headline by under one standard deviation.
- **Capacity needs calibrating per vehicle count.** The CSV holds veh/h (1,360–8,158);
  `capacity_scale = 0.0429` puts the herding baseline at worst-link ρ ≈ 3.3 (3.33 ± 0.27
  over 10 seeds) for 800 vehicles and does not carry over to a different demand size. It
  also fixes the unit: `cap` is vehicles per 0.0429 h ≈ 154 s, so ρ is a flow ratio over
  a 154-second window, not an occupancy, and the SUMO replay and the live demo measure ρ
  over the same window.
- **Speed limits are 89.7% imputed** in the OSM export (39,202 of 43,711 segments, 89.5%
  by length), all at 50 km/h, each with a `speed_imputed` flag. 《道路交通安全規則》§93
  also has a 30 km/h tier for service and track roads, but pyrosm's driving network
  leaves those out, so it never applies (log §15 #12). Since diversion depends on the
  arterial-vs-side-street speed ratio, this is worth a sensitivity check.
- **`tpred_fallback` is a genuine dilemma.** `free_flow` assumes every unobserved road
  runs at the limit while measured roads report about half that, so policies 2–4 route
  *away* from the instrumented arterials and finish 18% slower than free-flow;
  `network_mean` (the default: unobserved edges are scaled by the mean observed slowdown)
  removes that bias but removes the signal with it. `free_flow` is kept so the two can be
  run as a sensitivity pair; this project did not run it.
- **Multiple agents must be diagnosed in one `diagnose_agent.py` run.** Each agent
  abandons a different set of trips, so separate runs compute ATT over different common
  subsets — measured, two runs differed by 8 trips and static's ATT(common) moved 3.7%
  between them, which is larger than the effect being looked for.
- **`--repeat` is the reporting mode.** Single-seed numbers are for quick checks only.
- **Quality and latency come from two different decodings, and both must be stated
  together.** Beam-8 is the row to quote for ATT / Gini / worst-ρ, because greedy's
  17.2% attrition under S3 inflates its ATT delta past the oracle's. But greedy is the
  row that clears the proposal's 50 ms budget; beam-8 runs the decoder once per live
  beam per hop — 218–261 calls per vehicle against about 23 — and lands at 44–104 ms.
  Quoting whichever is favourable in each table would be indefensible.
- **Speed is not the agent's advantage.** Policy 6 answers a request in 0.31–0.33 ms
  (re-pricing amortised; plain Dijkstra for policies 1 and 4 takes 0.27–0.31 ms), 43–68x
  faster than the agent's greedy decoding, and it is better on all three quality metrics.
  Getting under 50 ms needs no learning. Policy 6 is fast *and* better here because the
  arena's cost function is its own — BPR plus eq.4 is what it optimises directly — so the
  agent's case rests on settings where that cost is not analytic. Testing that means
  training inside SUMO, which this project did not do: the replay above only re-times
  routes chosen under BPR.
- **Policy 7 is sensitive to the load already on the network when it decides.** Every
  offline result starts from an empty network. Re-planning the same 800 cars from a
  non-empty starting load drops greedy decoding from 737 to 345 delivered and beam-8 from
  774 to 632, while policies 1, 4 and 6 do not move (log §13.28). So `reroute_service.py`
  routes from an empty network; the simulator's load goes to the display only, and
  `seed_load=` is the explicit opt-in.
- **A failed trip's partial load counts for decisions, not for metrics.** Edges a
  policy-7 vehicle drives before it dead-ends stay loaded for the cars after it, but only
  complete routes are scored. Under S3 about 4% of beam-8's trips end that way, which
  slightly flatters policy 7's worst-ρ and Gini. Kept on purpose: beam search copies the
  behaviour so greedy and beam stay comparable (實驗設計 §8.3).
- **Which device is faster depends on the decoding width, and on scale.** The encoder is
  8.9x faster on GPU, the ≤3-candidate decoder 2.9x slower (pure kernel-launch
  overhead); the crossover is ~68–74 decoder calls per vehicle, so greedy prefers GPU
  and beam-8 prefers CPU by 2x. The GPU also has the worse tail (max 45.09 vs 30.66 ms),
  which for a deadline matters more than its better mean. On the full 9,904-node network
  the GPU's lead widens to 22x and it stops being optional — but there the binding cost
  is `_compute_enc_ctx`, a per-node Python loop that is 57% of the per-vehicle fixed
  cost against the model's 36%.

See `../paper_work/實驗設計.md` for the full experimental design and
`../paper_work/實驗記錄_DRL決策模組.md` for the development history.
