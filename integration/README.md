# `integration/` — prediction → decision dataflow

This folder wires the traffic-prediction models (STGCN + STGAT) to the routing decision
layer. It builds a congestion-weighted road graph from the predictions and compares
routing policies, measuring whether congestion-aware + global-penalty routing suppresses
the **herding effect (羊群效應)** — the core claim of the project.

Two road networks run through the same code path:

| `--graph` | network | edge times | speed source |
|---|---|---|---|
| `metr-la` (default) | 207-sensor Gaussian kernel | relative units | `stg{cn,at}_pred.npy` |
| `taichung` | real OSM network, 7,489 nodes / 20,390 edges | **seconds** | `taichung_pred_edges.csv` |

The decision stage is pure NumPy + NetworkX + SciPy (no GPU, ~1 s); only the DRL agent
needs PyTorch + `torch_geometric`.

---

## How to run

```bash
cd integration

# METR-LA
python run_compare.py                                   # seven-policy ablation
python run_compare.py --drl checkpoints/metr-la/drl_agent.pt        # add the trained agent
python run_compare.py --repeat 10 --drl checkpoints/metr-la/drl_agent.pt   # mean ± std
python train_drl.py --iters 200 --train-vehicles 300 --entropy-coef 0.03

# Taichung (real road network)
python taichung_loader.py                               # load check + graph stats
python calibrate_taichung.py                            # capacity sweep + comparison
python make_drl_input.py                                # ensemble predictions -> edge speeds
python run_compare.py --graph taichung --vehicles 800
python train_drl.py --graph taichung --iters 20 --train-vehicles 800
```

---

## Files

| file | role |
|---|---|
| `config.py` | paths + every hyper-parameter (single source of truth) |
| `network.py` | `build_graph_for(name)` — one entry point for both networks |
| `taichung_loader.py` | Taichung OSM CSVs → routing DiGraph; reads predicted edge speeds |
| `policies.py` | the seven routing policies + `RoutingEnv` / `EGATActorCritic` / `PPOTrainer` |
| `metrics.py` | ATT, TSTT, Gini(edge load), worst-link ρ, throughput |
| `run_compare.py` | run every policy on shared demand; multi-seed statistics |
| `train_drl.py` | train the PPO + Residual E-GAT agent, save the best checkpoint |
| `make_drl_input.py` | ensemble the two models' section predictions → per-edge speeds |
| `calibrate_taichung.py` | find the (vehicles, capacity_scale) where congestion actually appears |
| `pipeline.py` | orchestrator: (inference) → run_compare |
| `stg{cn,at}_pred.npy` | METR-LA model outputs, written by `../ST*/run_infer.py` |
| `taichung_pred_{stgcn,stgat}.npy` | Taichung section-level predictions (raw, unclamped) |
| `taichung_pred_edges.csv` | per-edge speeds — what the router consumes |
| `data/adj_mx_dijsk.pkl` | vendored METR-LA adjacency |

---

## Data flow

```
                        METR-LA                              Taichung
              stgcn_pred.npy ─┐                  taichung_pred_stgcn.npy ─┐
              stgat_pred.npy ─┴─ ensemble        taichung_pred_stgat.npy ─┴─ make_drl_input.py
                               0.7/0.3                        (clamp, ensemble,
                                  │                            section → OSM edges)
              adj_mx_dijsk.pkl    │                  Map/graph_*_taichung.csv   │
              d ∝ √(−ln adj) ─────┤                  length_m / speed limit ────┤
                                  ▼                                             ▼
                         network.build_graph_for(name)  →  directed road graph
                            edges carry: length · t0 · tpred · tpred_stgcn ·
                                         tpred_stgat · cap
                                              │
                    OD demand — "hotspot" funnels many cars at a few hub nodes,
                             which is what creates the herding pressure
                                              │
    ┌──────┬────────┬────────┬─────────┬────────────┬──────────────┬──────────┐
  1 static 2 STGCN  3 STGAT  4 hybrid  5 load-aware 6 global-      7 DRL
   (free-  +Dijkstra +Dijkstra +Dijkstra  (coord.)     penalty       (PPO +
    flow)                     (HERDING)                (oracle)      E-GAT)
    └──────┴────────┴────────┴─────────┴────────────┴──────────────┴──────────┘
                                              ▼
        every policy is re-scored under the SAME realized BPR congestion model:
        ATT · TSTT · Gini(edge load) · worst-link ρ · throughput
```

---

## The seven policies

A deliberate ablation ladder — each row adds exactly one capability:

| # | policy | prediction? | coordination? | eq. 4? | maps to |
|---|---|:--:|:--:|:--:|---|
| 1 | `static` (free-flow Dijkstra) | ✗ | ✗ | ✗ | proposal baseline ① |
| 2 | `STGCN + Dijkstra` | ✓ STGCN only | ✗ | ✗ | proposal baseline ② |
| 3 | `STGAT + Dijkstra` | ✓ STGAT only | ✗ | ✗ | proposal baseline ③ |
| 4 | `hybrid + Dijkstra` | ✓ ensemble | ✗ | ✗ | **the herding case — Δ is measured against it** |
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
> <50 ms reactive decisions. Treat it as the **upper bound for the eq. 4 objective**, and
> note it is *not* an upper bound on every metric: the agent's worst-link ρ beats it.

---

## Reporting protocol

A single demand draw varies too much to conclude from, so `--repeat N` draws N demands
and reports **mean ± std** with **paired** deltas (computed within each seed, then
averaged — this cancels the demand-to-demand variance that hits every policy alike).

```bash
python run_compare.py --repeat 10 --drl checkpoints/metr-la/drl_agent.pt
```

METR-LA, 5 seeds — note how much tighter Gini is than ATT:

| metric | Δ vs herding baseline | std |
|---|---|---|
| **Gini(load)** | **−61.8%** | **±0.6%** |
| worst-link ρ | −88.0% | ±2.0% |
| ATT | −23.8% | ±8.2% |

The headline claim rests on the most stable metric; ATT swings because some demand
draws are simply harder than others.

---

## Results (METR-LA, hotspot, 300 vehicles)

| policy | ATT | worst-link ρ | Gini(load) |
|---|---:|---:|---:|
| 1 static | 0.1118 | 4.500 | 0.910 |
| 4 hybrid + Dijkstra (**herding baseline**) | 0.0349 | 2.389 | 0.869 |
| 5 load-aware | 0.0226 | 0.389 | 0.546 |
| 6 global-penalty (oracle) | 0.0235 | 0.222 | 0.363 |
| **7 DRL agent (PPO + E-GAT)** | **0.0232** | **0.167** | **0.574** |

Against the herding baseline the DRL agent reaches **ATT −33.7%, worst-link ρ −93.0%,
Gini −35.0%** — clearing the proposal's targets (Gini ↓30%+, worst-link ↓20%+,
ATT ↓20–30%), with worst-link ρ better than the oracle's.

**Policies 2, 3 and 4 land on nearly identical Gini (0.868 / 0.870 / 0.869).** Whichever
model's forecast you follow, greedily chasing the predicted-fastest road herds just as
badly — the improvement comes from *coordination* and the *global penalty*, not from
prediction quality. That is the project's central claim, isolated.

---

## Notes and limitations

- **METR-LA travel times are relative** (the kernel's σ is unknown). Compare ratios,
  not absolute seconds. Taichung times are real seconds.
- **Taichung capacity needs calibrating.** The CSV holds veh/h (1,360–8,158); at a few
  hundred vehicles nothing congests, so `calibrate_taichung.py` finds the
  `capacity_scale` that puts the herding baseline at a target worst-link ρ.
- **93.7% of Taichung edges have no speed limit** in the OSM export; they fall back to
  50 km/h (`config.TAICHUNG_DEFAULT_SPEED_KMH`). Since diversion depends on the
  arterial-vs-side-street speed ratio, this is worth a sensitivity check.
- **Prediction coverage is partial**: 524 of 20,346 Taichung edges (2.6%) currently get
  a predicted speed; the rest keep their free-flow time. Pruning the network to
  200–300 nodes will raise this sharply — but only if the pruning keeps those corridors.
- **`--repeat` is the reporting mode.** Single-seed numbers are for quick checks only.
- Adding the DRL policy shifts every policy's Gini slightly, because Gini is computed
  over the union of edges any policy used and the extra policy enlarges that set. The
  ordering and the conclusions do not change.

See `../paper_work/實驗設計.md` for the full experimental design and
`../paper_work/實驗記錄_DRL決策模組.md` for the development history.
