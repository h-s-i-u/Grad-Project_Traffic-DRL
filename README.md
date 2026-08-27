# 基於機器學習與圖論的動態路網優化
**Dynamic Road-Network Optimization with STGCN / STGAT + Deep Reinforcement Learning**

A Tunghai University (東海大學) graduate project that combines spatio-temporal traffic
**prediction** (STGCN + STGAT) with a **deep-RL routing agent** to suppress the
*herding effect* (羊群效應) — the second-wave congestion that happens when a navigation
system funnels every vehicle that hears about a jam onto the same "fastest" detour.

The core idea, following the project proposal: predict near-future road speeds, turn
them into a congestion-weighted road graph, and route many vehicles with a **global
penalty** that trades a sliver of individual travel time for a far more even network
load — the Wardrop System-Optimum vs User-Equilibrium trade-off (Wardrop, 1952).

---

## Overview

The system has three modules (proposal §4):

1. **Prediction** — STGCN (spectral graph conv over the fixed road topology) + STGAT
   (graph attention for dynamic spatial dependencies), joined by the proposal's
   **gated fusion** (eq. 3) into one model that emits 15/30/60-minute speed forecasts.
2. **Decision** — a PPO agent (Residual E-GAT actor-critic) routes vehicles
   neighbour-by-neighbour. Its reward is the proposal's **eq. (4) global penalty**:
   `R = −α·Δtravel_time − λ₁·Σ max(0, ρ−ρ_th)² − λ₂·Var(ρ)`, which punishes
   over-saturating any link and rewards spreading load evenly.
3. **Visualization** — SUMO + web dashboard (planned; see Roadmap).

The fusion model lives in [`fusion/`](fusion/); everything that connects prediction →
decision lives in [`integration/`](integration/), which doubles as a standalone
evaluation harness for the routing policies.

### Two datasets, one code path

| | METR-LA | Taichung |
|---|---|---|
| role | literature-comparable benchmark | the field scenario (proposal §4.2) |
| sensors / sections | 207 | **202** (all 244 TDX sections map; 42 dropped for >50% missing) |
| timesteps | 34,272 | **50,283** |
| road network | Gaussian-kernel sensor graph | **real OSM: 39,920 nodes → 9,904 after lossless simplification** |
| edge times | relative units | **seconds** |
| missing / imputed | 7.13% | **24.6%** — evaluation must use a mask |

Both run through `python run_compare.py --graph {metr-la,taichung}`.

---

## Architecture / data flow

```
 TDX live speeds ─► build_simplified_network.py ─► build_network.py ─► build_speed.py
 OSM road network         (merge pass-through nodes, honour one-ways)        │
                                                                ┌───────────┴───┐
                                                            STGCN            STGAT
                                                                └───────┬───────┘
                                                          gated fusion (proposal eq. 3)
                                                          15 / 30 / 60 min, one forward
                                                                          │
                                    section speeds ──map──► per-edge speeds│
                                                                          ▼
 adjacency (Gaussian kernel, road distance) ────────► congestion-weighted road graph
                                                        (per-edge travel time + capacity)
                                                                   │
                              many vehicles (origin → destination demand)
                                                                   │
   ┌────────┬─────────┬─────────┬──────────┬────────────┬───────────────┬─────────┐
 1 static  2 STGCN   3 STGAT   4 hybrid   5 load-aware 6 global-penalty 7 DRL
 (Dijkstra) +Dijkstra +Dijkstra (HERDING)  (coordination)  (eq. 4 oracle)  (PPO+E-GAT,
                                                                        greedy | beam-8)
   └────────┴─────────┴─────────┴──────────┴────────────┴───────────────┴─────────┘
                                                                   ▼
        evaluate under one shared BPR congestion model:
        ATT · Gini(edge load) · worst-link saturation ρ · throughput
```

---

## Repository structure

| Path | What it is |
|---|---|
| `STGAT/` | STGAT prediction model — cloned from [xyk0058/STGAT](https://github.com/xyk0058/STGAT) |
| `STGCN/` | STGCN prediction model — cloned from [hazdzz/STGCN](https://github.com/hazdzz/STGCN) |
| `DRL-and-graph-neural-network-for-routing-problems/` | Residual E-GAT / PPO routing base — cloned from [Lei-Kun/DRL-…-routing-problems](https://github.com/Lei-Kun/DRL-and-graph-neural-network-for-routing-problems) |
| **`integration/`** | **The core of this project**: prediction → decision pipeline, routing policies, the PPO/E-GAT agent, and the evaluation harness. See its [README](integration/README.md). |
| **`fusion/`** | **Gated Fusion (proposal §4.3)**: the dual-path model with the eq. 3 gate, plus the unified dataloader both backbones share. See its [README](fusion/README.md). |
| **`TDX_Data/`** | Taichung traffic-data pipeline: fetch TDX → build network → build speed matrix → convert for both models. See its [README](TDX_Data/README.md). |
| `Map/` | Taichung OSM road network (`Map_fined/` = the raw export from teammate A) + the simplified routing graph, speed matrix, mask and adjacency |
| `CWA/` | Central Weather Administration rainfall — evaluated, then **deliberately excluded**; see below |
| `paper_work/` | Experimental design, development log, architecture diagram, written reports |
| `*.pdf` | The three method papers + the project proposal — see Credits |
| `requirements.txt` | Python dependencies (pip freeze of the `traffic_rl` env) |

> `STGAT/`, `STGCN/` and `DRL-...` are upstream repositories with their own history; our
> work concentrates in `integration/` and `TDX_Data/`, plus a small number of documented
> fixes inside the model repos (see `paper_work/實驗記錄_DRL決策模組.md`).

---

## Setup

Python 3.10, CUDA 12.1. A GPU is needed to (re)train the models and the DRL agent; the
routing comparison itself runs on CPU in ~1 s.

```bash
conda create -n traffic_rl python=3.10
conda activate traffic_rl
pip install -r requirements.txt        # torch 2.3.1+cu121, torch_geometric, networkx, scipy, ...
```

---

## Usage

### 1. Build the Taichung dataset (once)

```bash
cd TDX_Data
cp .env.example .env                   # fill in your TDX Client Id / Secret
python fetch_tdx_section_metadata.py   # 244 section coordinates
python fetch_tdx_section_live.py       # ~11.9 GB of speed history
python build_simplified_network.py     # one-way aware, merges pass-through nodes (39,920 -> 9,904)
python build_network.py                # map-matching + road distances (seconds to re-run)
python build_speed.py                  # speed matrix + missing-value mask
python convert_to_stgcn_dataset.py     # -> STGCN/data/taichung/
python convert_to_stgat_dataset.py     # -> STGAT/data/taichung/
```

### 2. Train and evaluate the prediction models

```bash
# STGCN — single-step output, so one model per horizon
cd STGCN
python main.py --dataset taichung --n_pred 3 --epochs 100
mv STGCN_taichung.pt STGCN_taichung_p3.pt      # main.py always writes STGCN_<dataset>.pt
python evaluate_masked.py --dataset taichung --checkpoint STGCN_taichung_p3.pt

# STGAT — emits 12 steps, so one model covers 15/30/60 min
cd STGAT
python train.py --cuda --data data/taichung/ \
    --adj_filename data/taichung/adj_mx_dijsk.pkl \
    --num_of_vertices 202 --params_dir experiment_taichung \
    --lr 3e-4 --epoch 500 --early_stop_maxtry 40
python evaluate_masked_taichung.py
```

> **Always report the masked numbers.** 24.6% of the Taichung cells are imputed and are
> nearly free for a model to "predict", so the unmasked score is flattered and is not
> comparable with METR-LA or the published baselines. Both evaluators print the masked
> and unmasked columns side by side, plus a persistence baseline and a PASS/FAIL verdict.

### 3. Feed the decision layer

```bash
cd STGAT && python run_infer_taichung.py --n-pred 3      # prints its target row R
cd ../STGCN && python run_infer_taichung.py --n-pred 3 --target-row R     --checkpoint STGCN_taichung_p3.pt                   # same instant, or the ensemble refuses
cd integration && python make_drl_input.py        # ensemble -> per-edge speeds
```

### 4. Routing comparison and the DRL agent

```bash
cd integration
python run_compare.py                                   # METR-LA, seven policies
python run_compare.py --repeat 10 --drl checkpoints/metr-la/drl_agent.pt   # mean ± std
python run_compare.py --graph taichung --vehicles 800   # real road network
python train_drl.py --graph taichung --iters 200 --train-vehicles 800
```

**S3, the arterial-closure scenario.** Without these flags the run is unchanged.

```bash
python run_compare.py --graph taichung --vehicles 800 --close-list   # what is worth closing
python run_compare.py --graph taichung --vehicles 800 --repeat 10 \
       --drl checkpoints/taichung/drl_agent_f10_800it.pt \
       --close-road 臺灣大道 --close-at 0.10       # results go to results_s3.json
```

`--close-road` matches a road-name **prefix**, so `臺灣大道` closes the whole corridor
and `臺灣大道三段` closes one segment. `--close-at` is the fraction of the dispatch
sequence that goes out before the incident. See [`integration/closure.py`](integration/closure.py).

### 5. Gated Fusion (proposal §4.3)

```bash
cd fusion
python verify.py --device cpu                      # required: see below
python train.py --freeze none --stgcn-tod                 --epochs 60 --batch 32 --patience 12                 --out checkpoints/fusion_c.pt
python evaluate.py --checkpoint checkpoints/fusion_c.pt --split test --dump-all
```

`verify.py` pushes the two already-trained backbones through the unified dataloader and
checks they reproduce their recorded MAE. That dataloader rebuilds the windowing, both
normalisations, the tensor layouts and the mask from scratch; get any one wrong and
training still runs, the loss still falls, and the result is simply not comparable with
anything else here. **Do not train until it passes.**

### 6. Prediction baselines

```bash
cd integration
python ha_baseline.py            # HA + persistence floors, all three horizons
python ha_baseline.py --n-pred 12
```

Needs no checkpoint and no GPU. It verifies its own window alignment by recomputing
persistence and comparing against the number `STGCN/evaluate_masked.py` reports; a
mismatch aborts rather than printing an incomparable score. Dump both models with
`run_infer_taichung.py --dump-all --device cpu` to add their columns to the
anomaly-bucket breakdown.

See [`integration/README.md`](integration/README.md) for the policy/metric details.

---

## Results

### Prediction (Taichung, masked — real observations only)

202 TDX sections, 50,283 timesteps, 73.8% real observations. STGCN needs one model per
horizon; STGAT emits 12 steps from one.

| model | 15 min | 30 min | 60 min | 12-step avg |
|---|---:|---:|---:|---:|
| **Fusion** (proposal §4.3, one model) | **3.3786** | **3.4799** | **3.5579** | — |
| STGAT | 3.3802 | 3.5127 | 3.6276 | 3.4698 |
| STGCN | 3.5560 | 3.7535 | 3.9549 | — |
| persistence | 4.2852 | 4.6712 | 5.1233 | 4.7328 |
| HA (historical average) | 4.0486 | 4.0489 | 4.0496 | — |
| **STGAT vs persistence** | **−21.1%** | **−24.8%** | **−29.2%** | **−26.7%** |
| **STGAT vs HA** | **−16.5%** | **−13.2%** | **−10.4%** | — |

Among the two single models STGAT leads at every horizon and the gap **widens** with it
(−4.9% → −8.3%), matching the STGAT paper's claim about long-range forecasting. The same
pattern held on the earlier 175-section dataset (−4.3% → −8.6%), so it reproduces across
section sets — but the two are not fed the same inputs (STGAT gets a time-of-day
channel, STGCN does not), so it cannot be attributed to architecture alone. That
asymmetry turns out to matter more than it looks; see *Gated Fusion* below.

Fusion is the proposal's dual-path model and emits all three horizons from one forward.
It is level with STGAT at 15 min and −1.9% at 60.

**Report both floors, not just persistence.** The two baselines fail in opposite ways —
persistence knows the current state and nothing about structure, HA knows the weekly
pattern and nothing about today — so beating only one is weak evidence. Beating both, at
every horizon, is what shows the model combines current observation with learned
structure. It also keeps the long-horizon claim honest: against persistence the margin
*grows* (−21% → −29%), against HA it *shrinks* (−16.5% → −10.4%), because HA does not
degrade with horizon at all. Quoting only the first credits the model for persistence
getting worse.

Three properties of the data that fell out of the HA comparison:

- **HA beats persistence at every horizon, including 15 min** — unusual for traffic
  data, and it is the imputation: persistence reads `v[t − n_pred]`, which is itself an
  ffill copy about a quarter of the time. HA is built from real observations only. (Its
  own unmasked MAE is 5.31 against 4.05 masked, for the mirror-image reason: HA cannot
  reproduce an ffill value, persistence gets it for free.)
- **Anomalies mean-revert within an hour.** Bucketing by how far the *input window* ran
  from its weekly norm (not by `|truth − HA|`, which would be circular), HA's error in
  the most unusual quartile *improves* with horizon (8.50 → 7.92) while persistence's
  degrades (8.38 → 9.81). At 60 min the anomaly was an hour ago and has mostly decayed.
- **Rush hour is the most *regular* period, not the hardest.** HA scores 3.51–3.60 on
  weekday peaks against 4.05 overall; persistence's worst cell in the whole table is
  weekday peak at 60 min (5.56). Peak also carries 21.0% of weekday cells against the
  16.7% a uniform day would give — ETag pairs more often when there is more traffic.

⚠️ **Open question.** HA scores **3.5054** on weekday peaks at 60 min, below STGAT's
overall 3.6276. Those are different cell subsets and cannot be compared directly, but it
has to be checked: if the model does not beat HA during peak hours, its headline margin
comes from off-peak and weekends — and peak is the only time routing matters. Answering
it needs one CPU inference pass per model (`--dump-all --device cpu`).

Two independent evaluators agree on the persistence baseline to within 0.1%
(4.2872/4.6744/5.1281 vs 4.2852/4.6712/5.1233), which is the strongest cross-check we
have on the evaluation path itself.

**Report skill relative to persistence, not raw MAE.** Widening the map from 175 to 202
sections lowered MAE by ~2.2%, but it lowered persistence by ~2.4% as well: the
model/persistence ratio is unchanged to within 1%. The new sections made the dataset
slightly easier; they did not make the models better.

**Always score with the evaluators, never the training log.** `STGAT/train.py` prints
"Evaluate best model" every epoch but scores the *current* net without reloading the
checkpoint — the discrepancy was 5.8% (3.6535 vs 3.4534).

### Ensembling: measured, and it barely helps

Sweeping the fixed weight on the validation set (1,137,678 real observations):

| STGCN : STGAT | MAE | RMSE |
|---|---:|---:|
| 1.00 : 0.00 | 3.3585 | 6.2449 |
| **0.20 : 0.80** | **3.1855** ← best MAE | 6.2637 |
| **0.65 : 0.35** | 3.2458 | **6.1853** ← best RMSE |
| 0.00 : 1.00 | 3.1983 | 6.3532 |

The best mix beats STGAT alone by **0.40%**, and the two models' errors correlate at
**0.934** — they are wrong in the same places. More telling, the MAE-optimal and
RMSE-optimal weights sit at **opposite ends**: STGCN makes fewer large errors, STGAT is
better on average, and one global constant has to choose. That is the empirical case
for the proposal's **Gated Fusion** (eq. 3), which gates per node and per timestep.

**This replicates.** The same sweep on the earlier 175-section dataset — different
sections, different adjacency, separately trained models — gave the same optimum
(0.20/0.80), the same margin (0.43%) and an error correlation of 0.934 to three
decimals. And on the **test** split at 60 min: optimum 0.15 against validation's 0.20
(0.07% apart, so the weight is not overfitted), margin 0.29%, correlation 0.908. It is
not an artefact of one dataset or one split.

### Gated Fusion: what it fixed was not what we expected

[`fusion/`](fusion/README.md) implements the proposal's eq. 3 as written — the gate sits
between the two backbones and the output head, both paths train under one loss, and the
new head emits 15/30/60 min from a single forward. Getting both repositories into one
process needed alias imports (they both ship a package called `model`) and one unified
dataloader that feeds each path its own normalisation; `fusion/verify.py` checks that
the two pretrained backbones reproduce their recorded MAE through it before anything is
trained.

Result on test: **level with STGAT at 15 min, −0.9% at 30, −1.9% at 60.** The gate is
alive — mean opening 0.83 (which is where the fixed-weight search landed, learned rather
than told), spread 0.035 across sections and 0.052 across time — but it is not where the
gain comes from.

Bucketing by how far the input window ran from its weekly norm shows the actual source:

| anomaly bucket | HA | STGCN | STGAT | fusion |
|---|---:|---:|---:|---:|
| Q1 most routine | 1.6674 | 1.7568 | 1.6015 | **1.5328** |
| Q2 | 2.4533 | 2.5181 | 2.3105 | **2.2446** |
| Q3 | 3.6379 | 3.6363 | 3.3363 | **3.2644** |
| Q4 most unusual | 7.9150 | 7.2665 | **6.6047** | 6.6268 |

The improvement is concentrated at the **routine** end and vanishes at the anomalous one
— the opposite of what a better gate would produce, and exactly what giving STGCN a
time-of-day channel would produce. Which is the finding: the proposal assigns the STGCN
path the *regular* component ("尖峰時段、星期週期"), but the shipped implementation feeds
it speed alone, so it was losing to a plain historical average on the most routine cells
and at weekday peak. **Half the proposal's dual-path premise did not hold until we gave
that path an input the proposal itself asks for.** The other half — STGAT owning the
anomalous end — held all along, and fusion cannot beat it there.

### Decision (Taichung arena, 800 vehicles, 10 demand draws)

Paired deltas against the herding baseline — computed within each seed, then averaged,
so the demand-to-demand variance that hits every policy alike cancels out. Policy 7 is
listed twice: the same weights under greedy decoding and under beam search. **Quote the
beam row** — see the warning below the table for why.

Baseline 4 routes on the fusion forecast, and the agent was trained on the same input,
so the whole table is fed by the model the architecture section describes.

**S2 — rush-hour funnel (the main scenario)**

| policy | served% | ATT | worst-link ρ | Gini(load) |
|---|---:|---:|---:|---:|
| 1 static (free-flow) | 100% | 848.3±161.6 | **3.3302±0.2703** | 0.7180 |
| 2 STGCN + Dijkstra | 100% | 861.2±164.1 | **3.3302±0.2703** | 0.7121 |
| 3 STGAT + Dijkstra | 100% | 866.5±164.6 | **3.3302±0.2703** | 0.7123 |
| 4 hybrid + Dijkstra (**herding baseline**) | 100% | 860.4±166.9 | **3.3302±0.2703** | 0.7128 |
| 5 load-aware (coordination only) | 100% | 476.4±9.8 | 2.1099±0.0446 | 0.6892 |
| 6 global-penalty (eq. 4 oracle) | 100% | 468.2±11.7 | **1.9282±0.0837** | **0.5838** |
| 7 DRL agent — greedy | 95.4%±1.8 | 515.4±15.6 | 2.3858±0.0999 | 0.6248 |
| **7 DRL agent — beam-8** | **99.9%±0.2** | 539.3±33.8 | 2.5075±0.1781 | 0.6407 |

| Δ vs herding baseline | ATT | Gini | worst-link ρ |
|---|---:|---:|---:|
| 5 load-aware | −42.9±9.8% | −3.3±0.5% | −36.2±5.8% |
| 6 oracle | −44.0±9.0% | −18.1±0.7% | −41.9±2.7% |
| 7 DRL — greedy | −38.4±9.7% | −12.3±0.8% | −28.1±3.6% |
| **7 DRL — beam-8** | **−36.0±7.9%** | **−10.1±0.6%** | **−24.6±2.0%** |

**S3 — an arterial closes 10% of the way through the demand**

| Δ vs herding baseline | ATT | Gini | worst-link ρ | served% |
|---|---:|---:|---:|---:|
| 5 load-aware | −60.4±4.1% | −7.7±0.5% | −35.7±4.5% | 100% |
| 6 oracle | −61.2±4.1% | −17.4±0.8% | −43.3±2.2% | 100% |
| 7 DRL — greedy | −63.0±4.5% ⚠️ | −13.6±1.0% ⚠️ | −36.1±3.4% ⚠️ | **82.8%±2.8** |
| **7 DRL — beam-8** | **−55.8±5.6%** | **−12.9±0.8%** | **−27.5±2.9%** | **96.0%±1.8** |

Against the proposal's targets, using the beam rows:

| metric | target | S2 | S3 |
|---|---|---:|---:|
| ATT (burst load) | ↓20–30% | **−36.0%** ✅ | **−55.8%** ✅ |
| worst-link ρ | ↓20% | **−24.6%** ✅ | **−27.5%** ✅ |
| Gini(edge load) | ↓30% | −10.1% ❌ | −12.9% ❌ |

#### Feeding the decision layer from fusion changed nothing measurable

Wiring gated fusion into the routing graph and retraining the agent on it moves every
headline by less than one standard deviation:

| | ensemble-fed agent | fusion-fed agent | gap |
|---|---:|---:|---:|
| S2 ATT Δ | −35.7±7.4% | −36.0±7.9% | 0.04σ |
| S2 Gini Δ | −9.9±0.8% | −10.1±0.6% | 0.25σ |
| S2 worst-ρ Δ | −24.7±2.1% | −24.6±2.0% | 0.05σ |
| S3 ATT Δ | −51.9±6.2% | −55.8±5.6% | 0.7σ |
| S3 Gini Δ | −13.1±1.0% | −12.9±0.8% | 0.2σ |
| S3 worst-ρ Δ | −26.8±5.0% | −27.5±2.9% | 0.14σ |

The two forecasts are genuinely different — correlation 0.985 at 15 min, 3.0% median
relative difference, against 0.998 and 0.97% for the pair (hybrid, STGAT) the routing
already treats as interchangeable. And baseline 4 did move: 866.6 → 860.4 s in S2 but
860.4 → 1432.7 in S3, i.e. *better* under one scenario and *worse* under the other by
about 1% each. Policies 1–3 are unchanged to four decimals, confirming that only the
row which consumes the hybrid forecast was touched.

This is the fourth independent check on the same conclusion, and the strongest: the
earlier three varied coverage or the forecast source, this one also **retrained the
agent on the new observations**.

⚠️ It compares two agents from single training runs, so "fusion makes no difference" and
"PPO's run-to-run variance swamps the difference" cannot be separated here. The
structural argument — 86% of routes identical, 14.1% coverage by length — is what makes
the first far more likely.

#### 🔴 A policy that gives up on the hard trips can "beat" the upper bound

Look at the S3 greedy row: **ATT −63.0%, better than the oracle's −61.2%.** It is not
better. It abandoned 17.2% of the demand, and the trips it abandoned are the hard ones,
so its average is taken over an easier set. Beam search recovers them and the number
falls to −55.8%, behind the oracle exactly as it should be.

The inflation scales with the attrition rate:

```
S2   4.6% abandoned  ->  ATT Δ inflated by  2.4pp
S3  17.2% abandoned  ->  ATT Δ inflated by  7.2pp   <- enough to pass the oracle
```

This is why `run_compare.py` puts served% in the first column and flags any row below
95% as NOT COMPARABLE. Without that guard the table would have supported the sentence
"our system outperforms the theoretical upper bound", which cannot be true.

#### What beam search does and does not fix

Beam-8 is the **same weights and the same observations** — only the decoding differs, so
every edge is still chosen by the policy (Lei et al. 2022 report greedy and beam side by
side for the same reason). `test_beam.py` checks that width 1 reproduces greedy route
for route before any wider number is read.

It fixes **completeness**: dead ends drop 36.8 → 0.5 in S2 and 132.4 → 29.6 in S3. It
does **not** fix route quality — widening the beam fourfold on the undisturbed network
moves ATT by 0.8%, and the 10-seed S2 headline gets *worse* (−38.4% → −36.0%) because
the recovered trips are expensive ones. The 11.5% gap to the oracle is model error, not
search error.

#### Reading the rest of the table

**Gini falls short of the 30% target, and so does the oracle** (−18.1% in S2, −17.4% in
S3). On METR-LA the ceiling was −50.7% and the agent reached −35.0%; here the ceiling
itself is −18.1%, because the starting distribution is already more even (0.71 against
METR-LA's 0.87) and the hotspot demand funnels into four hubs whose last few hops no
policy can avoid. That gap is a property of the arena and the demand, not of training.

**Policies 1–4 are indistinguishable** — identical worst-link ρ down to the standard
deviation, and the ATT spread between them (1.3–1.7%) is smaller than its own
seed-to-seed variance (±2.1%). At 24.6% prediction coverage, following a forecast does
not change the route (see below), so the improvement comes entirely from *coordination*
and the *global penalty*. That is the project's central claim, isolated about as
cleanly as it can be.

That ±2.1% is itself a correction. Under the fixed-weight ensemble the same spread read
±0.2%, which looked like remarkable stability; it was an artefact of the 0.20/0.80
weight making baseline 4's routes nearly identical to policy 3's. Fusion separates them,
and the honest statement becomes "the differences are smaller than the noise" rather
than "the differences are tiny".

**The agent beats pure coordination on Gini and only on Gini** (S2: −10.1% vs −3.3%;
S3: −12.9% vs −7.7%) while giving up ATT and worst-link ρ. Policy 5 has congestion
feedback but no eq. 4 penalty, so it is good at avoiding the worst link and poor at
evening out the distribution; the agent trades travel time for spread. That is eq. 4
working as designed — and under the incident it captures 74% of the oracle's Gini
improvement, against 56% on the undisturbed network.

⚠️ **Gini is computed over the union of edges any policy used**, so adding a row shifts
every row's Gini slightly (ATT and worst-link ρ are unaffected). Compare Gini only
within one run.

### Herding also destroys predictability

| policy | ATT std over 10 demands | vs baseline |
|---|---:|---:|
| 1 static / 4 herding | ±161.6 / ±166.9 | — |
| 5 load-aware | **±9.8** | **16× less** |
| 6 oracle | ±11.7 | 14× less |
| 7 DRL agent (beam-8) | ±33.8 | 5× less |

BPR's fourth-power term makes "everyone on the same link" extremely sensitive to which
demand you draw: the uncoordinated policies swing ±19% in mean travel time on the same
network with the same vehicle count. Coordination does not just lower ATT, it makes ATT
*estimable* — and for a navigation service, "we cannot tell you how long today will
take" is its own failure mode.

---

### Why the routing graph is not the whole city

On the full 20,347-edge network only 601 edges (2.2%) carry a prediction, so
`tpred == t0` almost everywhere and the Dijkstra-on-prediction baselines collapse onto
the static one. Baseline (4) is the denominator of every delta in this README, so that
collapse would hollow out the whole comparison.

`TDX_Data/build_arena.py` therefore carves a routing arena out of the network (TDX
corridors + the chains that connect them + the demo corridor + the lane-3 backbone),
then merges the pass-through nodes the subgraph cut creates: **840 nodes / 1,690 edges
at 24.6% coverage, and paths shortened from 46.8 to 29.2 hops**. Node ids stay the
original OSM ids, so routes still render on the full map, and the full export is never
modified.

**Raising coverage did not help, and that is a result rather than a disappointment.**
Re-simplification lifted coverage from 15.2% to 24.6% by edge count — and changed
nothing: policies 1–4 stayed identical to four decimals across all 10 seeds. The reason
is that coverage *by length* is what a shortest-path search actually weighs, and merging
edges does not remove road, so it stayed at 14.1% either way. Unobserved edges all get
the same fallback multiplier, and a uniform scaling cannot reorder paths; the observed
edges perturb path cost by only about 2%, less than the gap between competing routes.
Measured directly: **86% of routes are identical under `t0` and under `tpred`**.

With TDX's 202 sections covering ~600 edges in a network that needs ≥1,690 to stay
strongly connected and offer alternatives, prediction cannot dominate route choice here.
This is a data-availability limit, and it is disclosed rather than worked around.

### S3: closing an arterial mid-run

The proposal opens with an incident on 臺灣大道 pushing every navigation app onto the
same detour. S2's herding comes from demand; S3's comes from the network changing under
the vehicles, which is a different mechanism — so the closure lands *after* a fraction of
the demand has been dispatched. Vehicles already on their way keep their routes, and the
load they left on the closed road still counts.

The closure is **masked, never removed**: the actor's `edge_index`/`edge_static` are
state-dict buffers sized from the graph, so a graph with 83 fewer edges cannot load a
checkpoint at all. It is applied at three places — the fixed-weight Dijkstra of policies
1–4, the incremental assignment of 5–6, and both the action mask *and* the to-go estimate
of the agent. That last one is the trap: skip it and the agent's only sense of "how far
is left" points down a road that no longer exists, and nothing raises.

The 10-seed result is in the decision table above: the incident nearly doubles the
herding baseline's travel time (860 → 1433 s), and the coordinated policies' advantage
grows with it — the oracle goes from −44.0% to −61.2%, the agent from −36.0% to −55.8%.
**Gini does not follow**: the oracle's Gini improvement actually shrinks slightly
(−18.1% → −17.4%), so the incident scenario is not the lever that reaches the 30%
target. The agent does close more of the gap, though, capturing 74% of the oracle's Gini
improvement against 56% on the undisturbed network.

Two things the arena measurements settled beforehand, neither of which was the expected
answer:

**Load share does not predict disruption.** Same 766 trips, closed from the start:

| closed | edges | share of baseline load | ATT | ΔATT |
|---|---:|---:|---:|---:|
| (open) | 0 | — | 723.3 | — |
| 臺灣大道 (whole) | 83 | 8.81% | 1479.8 | **+104.6%** |
| 臺灣大道二段 | 36 | **6.32%** | 757.3 | **+4.7%** |
| 臺灣大道三段 | 20 | **2.24%** | 1467.7 | **+102.9%** |

The busiest segment has good parallels and barely matters; a quieter one is a throat and
accounts for nearly the whole effect. The same trap catches the data-driven option: the
single busiest road in the arena (松竹路, 20.5% of load) severs all four hotspot hubs and
leaves 269 of 799 trips unroutable, so `--close-busiest` has to guard on hub reachability
rather than take the maximum.

**Closing mid-run can make load distribution *more* even.** Sweeping when the incident
lands (`--close-at`, a fraction of the dispatch sequence):

| `--close-at` | ΔATT | ΔGini |
|---|---:|---:|
| 0.00 | +104.6% | +5.9% |
| 0.10 | +90.5% | +3.7% |
| 0.25 | +50.3% | +1.0% |
| 0.50 | +17.1% | **−2.3%** |
| 0.75 | +2.2% | **−2.5%** |

ATT falls off fast because BPR is quartic — half as many vehicles on the detour is about
a sixteenth of the excess delay. But Gini *reverses*: a mid-run closure splits the fleet
across two different route sets and spreads load over more edges than everyone following
one route would. The closure acts as a crude load balancer, which is exactly what the
agent is supposed to do voluntarily. **So S3 magnifies the ATT result but is unlikely to
be the lever that reaches the Gini target** — only an early closure raises the baseline
Gini at all.

About 3.8% of S2's trips have no route after the closure. They are **not** scored as
failures: 57 of the 60 nodes that drop out of the arena are still connected on the full
9,904-node network (which the same closure leaves 99.2% connected), so counting them
would report the arena's own pruning as a consequence of the incident — and would push
policy 7 under the 95% served threshold for an unrelated reason. They are dropped from
every policy alike, and the count is printed.

### Climate features: evaluated, then excluded

The proposal (§4.2/§4.3) plans to inject CWA rainfall. `CWA/analyze_rain_speed.py`
measured it first, controlling for section, hour-of-day and weekday/weekend:
0–5 mm/h of rain is associated with a **2.4–2.9% drop in speed** (p < 0.001), but the
relationship is **not monotonic** — above 10 mm/h speeds *rise* 3.0%, plausibly demand
suppression, on only 17 hours of data. The overall effect (~0.66 km/h) is far below the
models' own MAE (3.47 km/h), so the feature was **not integrated** and the decision is
disclosed with its numbers. It is not correct to say rainfall has no effect.

## Status & roadmap

**Prediction (Track A)**

- [x] STGCN / STGAT reproduced on METR-LA (STGAT test MAE ≈ 3.16, beats the proposal target)
- [x] Taichung TDX pipeline + masked evaluation (persistence baseline, PASS/FAIL verdict)
- [x] Road network rebuilt: all 244 TDX sections now map, 202 survive (was 175)
- [x] Both models trained on Taichung, on the 175- and 202-section datasets
- [x] Ensemble weight measured on validation, **confirmed on test** (0.20 vs 0.15, 0.07% apart)
- [x] Alignment guard: both models must predict the same timestamp or the ensemble refuses
- [x] HA (Historical Average) baseline — proposal §4.6, with a persistence cross-check
      confirming window alignment to four decimals
- [x] **Gated Fusion (proposal eq. 3)** — joint dual-path training, one model emitting
      15/30/60 min. Beats STGAT by 1.9% at 60 min, level at 15
- [x] Rainfall evaluated and, with evidence, excluded
- [ ] Ablation: rerun fusion without the time-of-day channel, to separate its
      contribution from the gate's
- [ ] A3 transfer experiment (METR-LA pretrain → Taichung fine-tune) — proposal §5

**Decision (Track B)**

- [x] Prediction → decision pipeline + the seven-policy ablation (`integration/`)
- [x] Global-penalty oracle demonstrating herding mitigation (eq. 4)
- [x] PPO + Residual E-GAT routing agent (GPU-enabled)
- [x] Multi-demand reporting protocol (paired deltas, mean ± std)
- [x] Routing arena rebuilt and re-simplified: 840 nodes / 1,690 edges, 24.6% coverage
- [x] Reward scale fixed — the terminal bonus is relative to a trip, not one edge
- [x] Congestion-aware to-go (`--togo-refresh`): closes 18% of the gap to the oracle and
      pushes worst-link ρ past the proposal's target
- [x] **Arterial-closure scenario S3**, 10 seeds — the proposal's motivating case
- [x] **Beam-search decoding**, with a width-1 equivalence test against greedy
- [x] **Policy 7 fully comparable in both scenarios** (served 100% / 96.4%)
- [ ] Widen the hotspot funnel (`N_HOTSPOTS`) — the last untried lever on Gini
- [ ] `tpred_fallback` sensitivity: `free_flow` vs `network_mean`, both reported
- [ ] One-horizon vs three-horizon agent ablation (now feasible: fusion emits all three)

**Cross-track**

- [ ] SUMO microscopic simulation + TraCI integration
- [ ] Web dashboard (event injection, live re-routing visualization)

Detailed design and history: [`paper_work/實驗設計.md`](paper_work/實驗設計.md) ·
[`paper_work/實驗記錄_DRL決策模組.md`](paper_work/實驗記錄_DRL決策模組.md)

---

## Team

東海大學 畢業專題 — **S12350312 黃子修 · S12350302 黃少鯤 · S12350131 江彥萱**

## Credits & references

This project builds on three open-source implementations; the corresponding papers are
included in this folder as PDFs.

| Module | Code (cloned from) | Paper |
|---|---|---|
| STGCN prediction | [hazdzz/STGCN](https://github.com/hazdzz/STGCN) | Yu, Yin & Zhu — *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*, IJCAI 2018 |
| STGAT prediction | [xyk0058/STGAT](https://github.com/xyk0058/STGAT) | Kong et al. — *STGAT: Spatial-Temporal Graph Attention Networks for Traffic Flow Forecasting*, IEEE Access 2020 |
| DRL routing | [Lei-Kun/DRL-and-graph-neural-network-for-routing-problems](https://github.com/Lei-Kun/DRL-and-graph-neural-network-for-routing-problems) | Lei et al. — *Solve routing problems with a residual edge-graph attention neural network*, Neurocomputing 2022 |

Traffic data from **TDX (Transport Data eXchange)**, road network from
**OpenStreetMap** via OSMnx. Additional methods referenced: Schulman et al. 2017 (PPO) ·
Wardrop 1952 (User-Equilibrium vs System-Optimum) · Lopez et al. 2018 (SUMO).
