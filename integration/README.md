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

# --- training and diagnosis ---
python train_drl.py --graph taichung --iters 800 --train-vehicles 800 \
                    --eval-vehicles 800 --fail-mult 10.0 --togo-refresh 25 \
                    --out checkpoints/taichung/drl_fusion_togo25.pt
python diagnose_agent.py --drl <ckpt-a> --drl <ckpt-b>   # several agents, ONE run
python test_beam.py --drl <ckpt> --vehicles 800 --widths 1,2,4,8

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
| `taichung_loader.py` | Taichung CSVs → routing DiGraph; reads predicted edge speeds and `road_name` |
| `policies.py` | the seven policies + `RoutingEnv` / `EGATActorCritic` / `PPOTrainer` / beam decoding |
| `closure.py` | the S3 arterial closure: what to close, when, and what to do with the trips it strands |
| `metrics.py` | ATT, TSTT, Gini(edge load), worst-link ρ, throughput |
| `run_compare.py` | run every policy on shared demand; multi-seed statistics; S3 flags; `--beam` |
| `train_drl.py` | train the PPO + Residual E-GAT agent, save the best checkpoint |
| `make_drl_input.py` | forecasts → per-edge speeds, with a same-instant guard |
| `calibrate_taichung.py` | find the (vehicles, capacity_scale) where congestion actually appears |
| `diagnose_agent.py` | split the agent's ATT gap into detour / congestion / attrition |
| `test_beam.py` | **beam width 1 must equal greedy** — run before trusting any beam number |
| `ha_baseline.py` | Historical-Average floor + anomaly-bucket breakdown (proposal §4.6) |
| `search_ensemble_weight.py` | fixed-weight sweep + error correlation |
| `taichung_pred_edges.csv` | per-edge speeds — what the router consumes (`.meta.json` records its source) |
| `data/adj_mx_dijsk.pkl` | vendored METR-LA adjacency |

---

## Data flow

```
                     METR-LA                          Taichung
           stgcn_pred.npy ─┐            fusion/  ──►  dump_fusion_test_p3.npz
           stgat_pred.npy ─┴─ ensemble  STGCN ──┐
                            0.7/0.3     STGAT ──┴──►  taichung_pred_{stgcn,stgat}.npy
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

A deliberate ablation ladder — each row adds exactly one capability:

| # | policy | prediction? | coordination? | eq. 4? | maps to |
|---|---|:--:|:--:|:--:|---|
| 1 | `static` (free-flow Dijkstra) | ✗ | ✗ | ✗ | proposal baseline ① |
| 2 | `STGCN + Dijkstra` | ✓ STGCN only | ✗ | ✗ | proposal baseline ② |
| 3 | `STGAT + Dijkstra` | ✓ STGAT only | ✗ | ✗ | proposal baseline ③ |
| 4 | `hybrid + Dijkstra` | ✓ gated fusion | ✗ | ✗ | **the herding case — Δ is measured against it** |
| 5 | `load-aware` | ✓ | ✓ | ✗ | ablation: isolates coordination |
| 6 | `global-penalty` | ✓ | ✓ | ✓ | **oracle / upper bound** |
| 7 | `drl-agent` | ✓ | ✓ | ✓ | **the method** (proposal baseline ④) |

Two mechanisms make this work:

- **BPR volume-delay** `t(load) = t₀(1 + 0.15·(load/cap)⁴)` — piling vehicles onto one
  link inflates its time. Without it the herding effect does not exist numerically.
  It is a static stand-in for SUMO; the policy and metric code will not change when
  SUMO replaces it.
- **eq. 4 generalized cost** — `α·t_e(load) + λ₁·max(0, ρ−ρ_th)² + λ₂·max(0, ρ−ρ̄)`,
  the proposal's global penalty rewritten as an edge weight.

> **(6) is the oracle, not the method.** It routes every vehicle with a full Dijkstra
> over exact global load — capabilities the DRL agent deliberately lacks in exchange for
> sub-50 ms reactive decisions. Treat it as the **upper bound for the eq. 4 objective**.

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
these back and apply them.

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

```bash
python run_compare.py --graph taichung --vehicles 800 --repeat 10 \
       --drl checkpoints/taichung/drl_fusion_togo25.pt --beam 8
```

Gini is consistently the tightest metric — ±0.6% across demand draws against ±8% for
ATT, on both networks. The headline claim rests on it.

⚠️ **Gini is computed over the union of edges any policy used**, so adding a policy row
shifts every row's Gini slightly. ATT and worst-link ρ are unaffected. Compare Gini only
within one run.

---

## Results

**METR-LA, hotspot, 300 vehicles** — Δ vs the herding baseline: ATT **−33.7%**,
worst-link ρ **−93.0%**, Gini **−35.0%**, clearing all three of the proposal's targets.

**Taichung, 800 vehicles, 10 demand draws, beam-8 decoding:**

| Δ vs herding baseline | S2 (rush-hour funnel) | S3 (arterial closure) | target |
|---|---:|---:|---|
| ATT | **−36.0±7.9%** | **−55.8±5.6%** | ↓20–30% ✅ |
| worst-link ρ | **−24.6±2.0%** | **−27.5±2.9%** | ↓20% ✅ |
| Gini(edge load) | −10.1±0.6% | −12.9±0.8% | ↓30% ❌ |
| served% | 99.9±0.2 | 96.0±1.8 | ≥95% ✅ |

Gini falls short — and so does the oracle (−18.1% / −17.4%), which places that gap in
the scenario rather than in training. On METR-LA the ceiling was −50.7% and the agent
reached −35.0%; here the ceiling itself is −18.1%, because the starting load is already
more even (0.71 against METR-LA's 0.87) and the hotspot demand funnels into four hubs
whose last few hops no policy can avoid.

**Policies 1–4 are indistinguishable.** Identical worst-link ρ down to the standard
deviation, and the ATT spread between them (1.3–1.7%) is smaller than its own
seed-to-seed variance (±2.1%). Whichever forecast you follow, greedily chasing the
predicted-fastest road herds just as badly — the improvement comes from *coordination*
and the *global penalty*, not from prediction quality. That is the project's central
claim, isolated about as cleanly as it can be.

---

## Notes and limitations

- **METR-LA travel times are relative** (the kernel's σ is unknown). Compare ratios,
  not absolute seconds. Taichung times are real seconds.
- **The router runs on the arena, not the whole city.** On the full 27,022-edge network
  only 601 edges (2.2%) carry a forecast, so `tpred == t0` almost everywhere and
  baselines 2–4 collapse onto 1 — which would hollow out every delta, since 4 is the
  denominator. `TDX_Data/build_arena.py` carves out 840 nodes / 1,690 edges at 24.6%
  coverage. Node ids stay the original OSM ids, so routes still render on the full map.
- 🔴 **Prediction has no measurable effect on routing here**, and this is a data limit
  rather than a bug. Coverage *by length* — what a shortest-path search actually weighs
  — is 14.1%, so unobserved edges all take the same fallback multiplier and a uniform
  scaling cannot reorder paths. Measured four independent ways: doubling coverage by
  edge count changed nothing; 86% of routes are identical under `t0` and `tpred`;
  swapping in a genuinely different forecast (gated fusion, correlation 0.985 with the
  fixed-weight mix) left baselines 2–4 on top of each other; and retraining the agent on
  that forecast moved every headline by under one standard deviation.
- **Capacity needs calibrating per vehicle count.** The CSV holds veh/h (1,360–8,158);
  `capacity_scale = 0.0429` puts the herding baseline at worst-link ρ ≈ 3.3 for 800
  vehicles and does not carry over to a different demand size.
- **Speed limits are 88.2% imputed** in the OSM export, filled at 50/30 km/h per
  《道路交通安全規則》§93 with a `speed_imputed` flag. Since diversion depends on the
  arterial-vs-side-street speed ratio, this is worth a sensitivity check.
- **`tpred_fallback` is a genuine dilemma.** `free_flow` assumes every unobserved road
  runs at the limit while measured roads report about half that, so policies 2–4 route
  *away* from the instrumented arterials and finish 18% slower than free-flow;
  `network_mean` removes that bias but removes the signal with it. Both belong in the
  report as a sensitivity pair.
- **Multiple agents must be diagnosed in one `diagnose_agent.py` run.** Each agent
  abandons a different set of trips, so separate runs compute ATT over different common
  subsets — measured, two runs differed by 8 trips and static's ATT(common) moved 3.7%
  between them, which is larger than the effect being looked for.
- **`--repeat` is the reporting mode.** Single-seed numbers are for quick checks only.

See `../paper_work/實驗設計.md` for the full experimental design and
`../paper_work/實驗記錄_DRL決策模組.md` for the development history.
