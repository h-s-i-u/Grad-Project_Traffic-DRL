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
| **`integration/`** | **The core of this project**: prediction → decision pipeline, routing policies, the PPO/E-GAT agent, and the evaluation harness. See its [README](integration/README.md). |
| **`fusion/`** | **Gated Fusion (proposal §4.3)**: the dual-path model with the eq. 3 gate, plus the unified dataloader both backbones share. See its [README](fusion/README.md). |
| **`TDX_Data/`** | Taichung traffic-data pipeline: fetch TDX → build network → build speed matrix → convert for both models. See its [README](TDX_Data/README.md). |
| `Map/` | Taichung OSM road network (`Map_fined/` = the raw export from 黃少鯤) + the simplified routing graph, speed matrix, mask and adjacency |
| `CWA/` | Central Weather Administration rainfall — evaluated, then **deliberately excluded**; see below |
| `paper_work/` | Experimental design, development log, architecture diagram, written reports |
| `*.pdf` | The three method papers + the project proposal — see Credits |
| `requirements.txt` | Python dependencies (pip freeze of the `traffic_rl` env) |

> `STGAT/` and `STGCN/` are **vendored upstream repositories** and keep their
> own licences — see [`NOTICE.md`](NOTICE.md). Our work is in `integration/`, `fusion/`,
> `TDX_Data/` and `CWA/`, plus these files, which sit inside the upstream directories
> only because they import the upstream packages:
>
> | Ours, inside an upstream directory | What it does |
> |---|---|
> | `STGAT/evaluate_masked_taichung.py` · `STGCN/evaluate_masked.py` | masked evaluation (real observations only) + persistence baseline + PASS/FAIL verdict |
> | `STGAT/run_infer_taichung.py` · `STGCN/run_infer_taichung.py` | single-row inference with a `.meta.json` sidecar, and `--dump-all` for full splits |
> | `STGAT/transfer_taichung.py` | the A3 transfer experiment (no gradient step) |
>
> The three documented fixes we made *to* upstream files are listed in `NOTICE.md`.

---

## Setup

Python 3.10, CUDA 12.1. A GPU is needed to (re)train the models and the DRL agent; the
routing comparison itself runs on CPU in ~1 s.

[Data](https://drive.google.com/drive/folders/1m0iOxlsheqmTdWPemat-1QlAfaugG6lU?usp=sharing)

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
cd ../STGAT
python train.py --cuda --data data/taichung/ \
    --adj_filename data/taichung/adj_mx_dijsk.pkl \
    --num_of_vertices 202 --params_dir experiment_taichung \
    --lr 3e-4 --epoch 500 --early_stop_maxtry 40
python evaluate_masked_taichung.py

# A3 transfer (proposal §5), which needs the from-scratch run above to compare against.
# transfer_taichung.py takes NO gradient step. All four variants matter: --no-transfer
# is the floor, and without it 8.76 reads as mediocre rather than worse than nothing.
python transfer_taichung.py --node-init mean
python transfer_taichung.py --node-init fresh --out-name init_fresh.pth
python transfer_taichung.py --no-transfer     --out-name init_random.pth
python transfer_taichung.py --node-init mean --recalibrate-bn 8192 --out-name init_bnrecal.pth
python evaluate_masked_taichung.py --checkpoint transfer_experiment/init_model.pth

# A3-b DOES train on the target. Every other flag must match the from-scratch run, and
# --params_dir must not be experiment_taichung or it overwrites what it is compared to.
python train.py --cuda --data data/taichung/ \
    --adj_filename data/taichung/adj_mx_dijsk.pkl --num_of_vertices 202 \
    --params_dir transfer_experiment \
    --init-from transfer_experiment/init_model.pth \
    --lr 3e-4 --epoch 500 --early_stop_maxtry 40
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

# ablation C₀ — the same command minus one flag
python train.py --freeze none --epochs 60 --batch 32 --patience 12 \
                --out checkpoints/fusion_c0.pt
python evaluate.py --checkpoint checkpoints/fusion_c0.pt --split test   # no --dump-all
```

**Never pass `--dump-all` for an ablation.** `evaluate.py` writes
`integration/dump_fusion_<split>_p<N>.npz` under a fixed name that does not encode the
checkpoint, and those three files are what `make_drl_input.py --source fusion` reads to
build the decision layer's input. An ablation that dumps over them makes the reported
10-seed routing results silently irreproducible. If an ablation's anomaly buckets are
genuinely needed, back the three files up and verify the restore with `md5sum -c`.

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

### 7. Inference latency (proposal §5: "< 50 ms per decision")

```bash
cd integration
python bench_latency.py --graph taichung --vehicles 800 \
       --drl checkpoints/taichung/drl_fusion_togo25.pt --beam 8 --verify --scale
python bench_latency.py --graph taichung --vehicles 800 \
       --drl checkpoints/taichung/drl_fusion_togo25.pt --beam 8 --device cpu --scale
```

Run **both devices**. On a 1,690-edge graph the encoder is throughput-bound (GPU wins
8.9x) while the decoder is kernel-launch-bound (CPU wins 2.9x), so the better device
depends on how many times you decode per vehicle — the crossover is around 68–74 calls.
Greedy does 23, beam-8 does 261.

`policies.py` is not modified: the timers are installed on the env and agent
*instances*. `--verify` re-runs the same demand through the untouched `policy_drl` and
requires every path to match, because the timing loop is otherwise a copy of it.

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

| anomaly bucket | HA | STGCN | STGAT | fixed 0.2/0.8 | fusion C | **fusion C₀** |
|---|---:|---:|---:|---:|---:|---:|
| Q1 most routine | 1.6674 | 1.7568 | 1.6015 | 1.6044 | **1.5328** | 1.5419 |
| Q2 | 2.4533 | 2.5181 | 2.3105 | 2.3100 | **2.2446** | 2.2521 |
| Q3 | 3.6379 | 3.6363 | 3.3363 | 3.3293 | **3.2644** | 3.2707 |
| Q4 most unusual | 7.9150 | 7.2665 | 6.6047 | **6.5875** | 6.6268 | 6.6067 |

The improvement is concentrated at the **routine** end and vanishes at the anomalous one
— the opposite of what a better gate would produce, and exactly what giving STGCN a
time-of-day channel would produce. That reading was wrong, and the ablation that was
written down to test it says so.

#### The time-of-day channel contributes nothing

`C₀` retrains with one flag removed (`--stgcn-tod`, i.e. the STGCN path drops back to
speed alone) and everything else — freeze mode, gate type, loss, lr, decay, batch, seed
— identical. The criterion was fixed in advance against the 1% noise floor that §13.7's
fixed-weight ceiling defines: land near 3.61 and the attribution holds, land near 3.56
and it does not.

| | C₀ (no tod) | C (tod) | C − C₀ |
|---|---:|---:|---:|
| test 15 / 30 / 60 min | 3.3932 / 3.4825 / **3.5590** | 3.3786 / 3.4799 / **3.5579** | 0.43% / 0.07% / 0.03% |
| val 12-step | 3.1423 | 3.1391 | 0.10% |

Six of seven comparisons land under 0.2%, and the whole −1.9% survives without the
channel. The bucket table above settles it: of Q1's 4.29pp gain over STGAT, the channel
accounts for 0.57pp — **13%** — and at **weekday peak**, the one period where a
time-of-day feature is by construction most informative, **C₀ is 0.45% better than C.**
Both sit inside the noise floor, but there is no cell anywhere in which the signal
points the way the attribution predicted.

The mechanism was predictable from the gate. It sits at **0.78–0.83**, so the fused
output is roughly four-fifths STGAT and one-fifth STGCN; improving the input of a path
that carries 20% of the weight cannot move the result. That is the same fact §13.7
measured as a best fixed weight of 0.2/0.8, arrived at from the other direction.
"The effect concentrates in X-shaped samples" does not license "X's feature caused it" —
the bucket *shape* is inherited from STGAT, which had the channel all along.

What survives unchanged is the measurement underneath: STGCN **on its own**, without a
time-of-day channel, loses to a plain historical average on the most routine cells and
at weekday peak, while the proposal assigns it exactly the *regular* component
("尖峰時段、星期週期"). So half the dual-path premise does not hold as shipped. What no
longer follows is that adding the channel is what fixes it. The honest statement is that
we implemented the input the proposal asks for and measured it to contribute nothing.

#### Neither does the second path, or the gate

`--single-path stgat` bypasses the gate and sends one path to the head, leaving the
dataloader, normalisation, loss, head and horizon count untouched. (Written as
`tanh(W2·t)` rather than pinning the gate to 1 — a pinned gate still carries W3's *bias*,
so the discarded path would keep contributing a learned constant.)

| 60 min MAE | | vs the row above | what that row adds |
|---|---:|---:|---|
| STGAT, trained on its own | 3.6276 | — | — |
| **STGAT alone through fusion's training regime** | **3.5557** | **−2.0%** | **the training regime** |
| + second path + learned gate (C₀) | 3.5590 | +0.09% | **dual path + gate** |
| + time-of-day into STGCN (C) | 3.5579 | −0.03% | **time-of-day** |

One path, no gate, and it matches the full dual-path model at every horizon — at 60 min
it is 0.06% *ahead*. **All three components of the proposal's §4.3 measure zero. The
entire 1.9–2.0% is the training regime** (masked MAE, fusion's head, its early-stopping
criterion), which has nothing to do with eq. 3.

This was foreseeable from a number measured three weeks earlier and not followed
through: the two models' errors correlate at **0.934**, and the best fixed blend of them
beats STGAT alone by 0.43%. Models that are wrong in the same places cannot be rescued
by any rule for combining them, and a learned gate is just a rule for combining them —
which is why it converged on 0.78–0.83, the same 0.2/0.8 the weight search had already
found. **This is a property of the data, not a defect in the implementation.**

The reported model stays C: it is what §4.3 asks for, and it is what feeds the decision
layer. What changes is the claim. Not "gated fusion cut 60-min MAE by 1.9%", but
**"we implemented §4.3's gated fusion and three ablations show none of its components
contributes measurably"** — with the error correlation as the reason. The remaining 2.0%
is not decomposed further: the baseline STGAT was trained by a different person with a
different loss and stopping rule, so the honest scope is "this regime beats that one",
and going deeper is prediction-module tuning rather than an architectural claim.

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

### Transfer, METR-LA → Taichung: the architecture is inductive, the transfer is not

The proposal (§5) expects the METR-LA model to run zero-shot on Taichung and to reach a
usable level after a few fine-tuning epochs. `STGAT/transfer_taichung.py` tests the first
half on its own: it builds a Taichung-shaped STGAT, copies across whatever survives the
change of road network, and writes the result out without taking a single gradient step.

**95.6% of the parameters transfer by shape** — 298 of 332 tensors, 5,153,952 weights:
the gated temporal convolutions, the GAT's shared W / a1 / a2, downsample, EndConv. The
other 34 tensors are bound to the node count (18 per-node GAT biases, 16 BatchNorms over
the node axis). So the architecture really is inductive; what blocks the transfer is 4.4%
of the parameters and an implementation choice, not the method.

Supplying that 4.4% three different ways does not rescue it:

| MAE, real observations only | 15 min | 30 min | 60 min | 12-step mean |
|---|---:|---:|---:|---:|
| **random init — nothing transferred** | **7.3684** | **7.3679** | **7.5950** | **8.0661** |
| transfer + source's mean node | 8.7621 | 10.3458 | 11.2600 | 10.6411 |
| transfer + BatchNorm recalibrated on target | 23.3704 | 42.4420 | 27.3279 | 69.5339 |
| transfer + fresh node stats | 56.0498 | 70.0450 | 71.8151 | 90.5810 |
| persistence | 4.2852 | 4.6712 | 5.1233 | 4.7328 |
| trained from scratch (A2) | 3.3802 | 3.5127 | 3.6276 | — |

**The best of the four is the one that transfers nothing.** That is not a sign of a bug —
a random network emits something close to a constant, so its MAE is roughly the mean
absolute deviation and it is *safe*, because it says nothing. A transferred network says
something structured and wrong, which is easily worse. The conclusion is therefore that
the transferred representation is **actively misaligned** on the target network, not that
it is uninformative.

Three checks were run before drawing that conclusion: the source checkpoint is healthy
(no NaN, sane norms), the two datasets' feature channels mean the same thing (channel 1
is time-of-day in [0,1] in both), and the z-score touches only channel 0, per dataset —
so both channels arrive at the model on the same scale. A predicted explanation for the
failure — that the node-bound BatchNorm statistics were the blocker — was tested by
recalibrating them on Taichung with no gradient step, and it made things four times
*worse*, so that hypothesis is dead.

Scope, which the report has to state: this is STGAT only, since STGCN's spectral
convolution is tied to a fixed Laplacian and is transductive by construction; and
"zero-shot" here means no gradient step, not no target data at all — the z-score
statistics come from Taichung's own train split, because the two cities are in different
units.

#### Fine-tuning from it does work, which is the other half of the claim

| | best epoch | epochs run | wall clock | 15 min | 30 min | 60 min |
|---|---:|---:|---:|---:|---:|---:|
| from scratch (A2) | **23** | 64 | 4024.6 s | **3.3802** | **3.5127** | **3.6276** |
| **fine-tuned from the transfer** | **7** | 47 | 3941.9 s | 3.4137 | 3.5304 | 3.6510 |

Final quality is within the 1% threshold at all three horizons (+0.50 to +0.99%, all
slightly worse), and the best epoch arrives in **7 rather than 23**. The proposal says
"usable after fine-tuning for a few epochs" — seven is a few. **So §5's first clause
fails and its second holds.**

The two results are not in tension once stated properly: the same initialisation is a
*bad predictor* and a *good starting point*. Its weights compute the wrong function, yet
they sit in a region of parameter space from which the target task is quickly reachable.
"Can it be used directly" and "is it easy to learn from" are different questions, and
the proposal wrote them as two halves of one sentence.

Three caveats. "Faster" means epochs, not wall clock — with patience-40 early stopping
both runs cost about the same total time, so the saving only materialises if you stop at
the plateau. The pretrained run converges sooner but to a *slightly* worse plateau, which
is the usual shape. And most importantly: **`STGAT/train.py` had no seed at all until 31
Aug** — no `--seed` argument existed and the seeding lines were commented out, while
`STGCN/main.py` and `fusion/train.py` have defaulted to 42 from the start. A1, A2 and
A3-b are therefore single unseeded draws, so 7-vs-23 looks well outside noise but cannot
be claimed rigorously until the from-scratch run is repeated over several seeds.

### Why the Gini target is not met, measured rather than argued

Gini is the one proposal target the system misses: −10.1% (S2) and −12.9% (S3) against
a −30% goal. The proposal also asked for a grid search over the eq. 4 weights
(alpha, lambda1, lambda2) that had never been run. `integration/sweep_lambda.py` runs it
against policy 6, the analytic optimiser of eq. 4 — if that cannot reach the target, the
objective cannot.

| lambda2 | ATT Δ | **Gini Δ** | worst-ρ Δ |
|---:|---:|---:|---:|
| 0.0 | **−45.4** | −4.7 | −41.6 |
| 0.3 *(proposal default)* | −45.3 | −12.7 | −42.0 |
| **0.8** *(what everything is reported at)* | −44.0 | **−14.5** | −41.9 |
| 3.0 | −41.8 | −17.1 | −42.6 |
| **12.0** | −39.5 | **−19.1** | −43.2 |

At fifteen times the reported weight the optimum reaches −19.1%, and each doubling buys
about 20% less than the last (ratio 0.81), so the series extrapolates to **−22.6%** —
still seven points short. **The target is unreachable within eq. 4, and that is now
measured.** Alpha is not swept and does not need to be: scaling (alpha, lambda1, lambda2)
together scales the whole cost, which Dijkstra's argmin ignores, so only the ratios
matter.

The sweep also separates the two penalty terms cleanly. Without lambda2 the spread term
is gone and Gini improves by only −4.7%; adding it at 0.8 takes it to −14.5%. Raising
lambda1 from 0.5 to 2.0 moves worst-ρ by 4.4 points and Gini by 0.4. **lambda2 buys Gini,
lambda1 buys worst-link saturation** — which is what eq. 4 has two terms for.

#### The reason is topology, and the target does not transfer between graphs

| policy 6 at lambda2 = 12 | out-degree | hops | baseline Gini | **Gini Δ reachable** |
|---|---:|---:|---:|---:|
| **METR-LA, hotspot** | **76.22** | 1.47 | 0.87 | **−50.7%** — target met |
| **Taichung, hotspot** | **2.01** | 28.7 | 0.756 | **−19.1%** |
| Taichung, random *(same graph, demand dispersed)* | 2.01 | 28.7 | 0.641 | −16.0% |

**Changing the demand moves it 3 points. Changing the graph moves it 32.** Spreading load
needs parallel alternatives: a Gaussian-kernel sensor graph offers 76 neighbours and
1.5-hop trips, a real road network offers two exits and 28.7 hops. Dispersing the demand
makes the *relative* figure worse, not better, because the baseline is already more even
and there is less room to improve proportionally.

So the honest statement is not "we missed the target". It is that **the −30% figure was
calibrated on a dense similarity graph and does not carry to a real road network** — and
the earlier explanation offered here, that the four hotspot hubs form an unavoidable
bottleneck, does not hold either: those 16 in-edges carry 3.2% of all edge traversals,
which pins worst-ρ's floor, and worst-ρ already meets its target.

Retraining policy 7 at a higher lambda2 was considered and rejected on arithmetic, not
instinct: it captures ~56% of policy 6's Gini improvement, so the 4.6-point gain would
become ~2.6, still seventeen short, at the cost of invalidating every result downstream
of it.

### Inference latency: the claim holds, the argument behind it does not

The proposal (§5) promises "< 50 ms per single decision". Timing a `run_compare` rollout
and dividing by the vehicle count does not measure that — it mixes a per-vehicle graph
encode with a per-hop decode, and the hop count is itself a variable (32.8 hops in S2,
40.3 in S3, 79 at worst). `bench_latency.py` decomposes it and reports what one **routing
request** costs, end to end, state preparation included.

| one vehicle's route, ms | GPU / S2 | GPU / S3 | CPU / S2 | CPU / S3 |
|---|---:|---:|---:|---:|
| **policy 7, greedy** | **13.87** | **13.76** | 21.50 | 22.13 |
| under 50 ms | ✅ **100%** | ✅ **100%** | ✅ **100%** | ✅ **100%** |
| policy 7, beam-8 | 90.02 | 103.89 | 43.95 | 51.24 |
| under 50 ms | ❌ 22.5% | ❌ 22.2% | ❌ 67.1% | ❌ 52.5% |
| 1 / 4, one Dijkstra request | ~0.29 | ~0.29 | ~0.28 | ~0.30 |
| **6 oracle, one request** | **~0.31** | **~0.32** | **~0.31** | **~0.33** |

Greedy passes on every one of 2,668 vehicles, on both devices, in both scenarios — not
just on average. **Three things follow, and two of them are uncomfortable.**

**Beam-8 fails, and beam-8 is the row we quote for quality.** The decoder runs once per
live beam per hop: 261 calls per vehicle against greedy's 23. The quality numbers and
the latency number therefore come from two different settings, and the report has to say
so in the same breath rather than quoting whichever is favourable. There is an
unimplemented fix — batching the 8 beams into one forward would cut the GPU figure to an
estimated ~35 ms.

**The oracle is 43–68x faster and better on all three metrics.** That retires the
argument, made earlier in this project, that policy 7 trades coordination quality for
responsiveness. It does not: getting under 50 ms needs no learning at all. Policy 6 wins
here because **the arena's cost function is its own** — BPR plus eq.4 is exactly what it
optimises directly, so it is the answer rather than an approximation of it. Policy 7's
case rests on settings where that cost is not available in closed form, which is what
SUMO would provide and what has not been tested. What can be claimed today is that it
learns **56% (S2) / 74% (S3)** of the oracle's Gini improvement from reward alone.

**Which device is faster depends on the decoding width, not on the model.** The encoder
(1,690 edges, 3 GATv2 layers) is 8.9x faster on GPU; the decoder (≤3 candidates) is 2.9x
*slower*, being pure kernel-launch overhead. Crossover: ~68–74 decoder calls per vehicle.
And the GPU's mean is better while its tail is worse — max 45.09 ms against the CPU's
30.66 ms, leaving under 10% headroom. **For a deadline, the CPU is the safer device**,
which is the opposite of the usual instinct.

#### At full-network scale the bottleneck is state preparation, not the model

The E-GAT weights carry no node count, so the same checkpoint loads onto the full
9,904-node / 27,022-edge network. Timing the forward there (routing quality at that
scale is untested and not claimed):

| 840 → 9,904 nodes (×11.8), medians | GPU | CPU |
|---|---|---|
| **encoder (E-GAT)** | 4.209 → **7.686** ms | 8.527 → **170.160** ms |
| `_compute_enc_ctx` (Python loop) | 0.751 → 11.948 ms | 1.155 → 11.910 ms |
| to-go Dijkstra | 2.082 → 36.782 ms | 2.995 → 36.430 ms |
| 1 static Dijkstra | 0.519 → **6.329** ms | 0.751 → **7.074** ms |

**The GPU's lead over the CPU widens from 8.9x to 22.1x** — 1,690 edges never saturate
it, 27,022 begin to. Of the 21.1 ms a vehicle costs before its first hop on the full
network, **64% is state preparation** (the per-node Python loop plus the Dijkstra) and
only 36% is the model. Vectorising `_compute_enc_ctx` is what buys headroom at scale;
a smaller model would not. On CPU the same fixed cost is 183.6 ms, so at full scale the
GPU stops being merely faster and becomes required — the device choice flips with
network size as well as with decoding width.

A Dijkstra request on the full network still costs only 6–7 ms, so §2.1's premise is not
rescued by scale either.

*Caveat on the ratios:* the same encoder measures 1.786 ms (GPU) in the 800-vehicle
rollout and 4.209 ms in the scale block, with the CPU moving the other way (15.884 →
8.527) — clock states differ between a 30-vehicle and an 800-vehicle run. Ratios are
taken within one block, and the two claims above survive either denominator. The claim
that does not survive, and is therefore not made, is "the encoder scales worst on CPU".

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
      15/30/60 min. 1.9% ahead of the separately-trained STGAT at 60 min, level at 15 —
      but see the ablations: none of that comes from the architecture
- [x] Rainfall evaluated and, with evidence, excluded
- [x] Ablation: fusion without the time-of-day channel — **it contributes nothing**
      (0.03–0.43%), refuting the attribution that had been inferred from the bucket table
- [x] C₀'s anomaly buckets — the channel accounts for 13% of the Q1 gain, and at
      weekday peak removing it is *better*
- [x] Control: STGAT alone through fusion's dataloader/head/loss — **it reaches −2.0% by
      itself**, so the dual path and the gate contribute nothing. All three components
      of proposal §4.3 measure zero; the gain is the training regime
- [x] **A3-a: zero-shot transfer, METR-LA → Taichung** (proposal §5) — **it does not hold.**
      95.6% of the weights transfer by shape, but no way of supplying the remaining 4.4%
      beats a randomly initialised network, and all of them lose badly to persistence
- [x] **A3-b: fine-tune from the transferred weights — it does hold.** Best epoch 7
      against 23 from scratch, final MAE within 1% at every horizon
- [ ] Run the from-scratch STGAT over several seeds, to establish how much best-epoch
      varies naturally — `--seed` only exists as of 31 Aug, so 7-vs-23 is still two
      single unseeded runs

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
- [x] **Inference latency measured** (proposal §5) — greedy clears 50 ms on 100% of
      2,668 vehicles, on both devices. Beam-8 does not, and the Dijkstra baselines are
      43–68x faster than either
- [ ] Batch the 8 beams into one decoder forward — would put beam-8 back under 50 ms
      (estimated ~35 ms) and remove the quality/latency conflict
- [ ] Vectorise `_compute_enc_ctx`: a per-node Python loop, now a third of the GPU
      per-vehicle model cost
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

東海大學 (Tunghai University) graduate project.

| Member | Responsible for |
|---|---|
| **黃子修** | Prediction models (`STGCN/`, `STGAT/`, `fusion/`), the TDX processing chain and routing arena (`TDX_Data/`), and the decision layer (`integration/`) — everything documented in this README |
| **黃少鯤** | Taichung OSM road-network export (`Map/Capture_Road_Node.py`, `Map/Map_fined/`) and TDX data collection (`TDX_Data/fetch_tdx_section_*.py`) |
| **江彥萱** | SUMO microscopic simulation and TraCI integration (in progress) |

## Credits & references

This project builds on three open-source implementations; the corresponding papers are
included in this folder as PDFs.

| Module | Code (cloned from) | Paper |
|---|---|---|
| STGCN prediction | [hazdzz/STGCN](https://github.com/hazdzz/STGCN) | Yu, Yin & Zhu — *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting*, IJCAI 2018 |
| STGAT prediction | [xyk0058/STGAT](https://github.com/xyk0058/STGAT) | Kong et al. — *STGAT: Spatial-Temporal Graph Attention Networks for Traffic Flow Forecasting*, IEEE Access 2020 |
| DRL routing | [Lei-Kun/DRL-and-graph-neural-network-for-routing-problems](https://github.com/Lei-Kun/DRL-and-graph-neural-network-for-routing-problems) — **re-implemented, not vendored** | Lei et al. — *Solve routing problems with a residual edge-graph attention neural network*, Neurocomputing 2022 |

Additional methods referenced: Schulman et al. 2017 (PPO) · Wardrop 1952
(User-Equilibrium vs System-Optimum) · Lopez et al. 2018 (SUMO).

## Licensing

Our own code is **MIT** ([`LICENSE`](LICENSE)). The vendored upstream directories keep
their own terms and are **not** covered by it — `STGCN/` is LGPL-2.1, and **`STGAT/` upstream ships no licence file at all**. The datasets carry
their providers' terms. [`NOTICE.md`](NOTICE.md) has the per-directory breakdown, which
files inside the upstream repos we changed, and what to do if you want to reuse any of it.

> Road network © **OpenStreetMap** contributors, under the **Open Database Licence
> (ODbL) 1.0** — the derived road-graph CSVs in `Map/` are an ODbL derived database.
> Traffic speed data 資料來源：**交通部運輸資料流通服務平臺（TDX）**.
> Rainfall data 資料來源：**中央氣象署**.
