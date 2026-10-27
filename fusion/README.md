# Gated Fusion — the dual-path model of 計劃書 §4.3

> STGCN path (regular structure) + STGAT path (anomalous structure), joined by the
> proposal's eq. 3 gate, with an FC head emitting 15 / 30 / 60 min.
>
> See also [`../paper_work/實驗設計.md`](../paper_work/實驗設計.md) §3.7 and
> [`../paper_work/實驗記錄_DRL決策模組.md`](../paper_work/實驗記錄_DRL決策模組.md)
> §13.7 (fixed-weight ensemble), §13.19 (pipeline), §13.22 (training and ablations),
> §13.24 (wired into the decision layer).

```
gate   = σ( W₁ · (s + t) )
Fusion = tanh( W₂ · t · gate + (I − gate) · W₃ · s )
output = FC(Fusion)                    → 15 / 30 / 60 min
```

`s` and `t` are the two paths' **hidden features**, not their finished forecasts. That is
where the proposal puts the fusion: between the backbones and the output head.

As implemented (`model.py`), the gate multiplies the features *before* the linear maps:
`Fusion = tanh(W₂(gate ⊙ t) + W₃((1 − gate) ⊙ s))`, after `proj_cn` has taken STGCN's
256-dim feature down to 64. Same parameters, a different function from the equation as
written, and one consequence that matters: W₃'s bias is never gated, so a gate pinned at 1
would still let the STGCN side add a learned constant. That is why control E bypasses the
gate (`tanh(W₂ t)`) instead of pinning it.

---

## Why this folder exists

The two paths come from separate repositories, and today their data pipelines have
nothing in common:

| | STGCN | STGAT |
|---|---|---|
| data files | `vel.csv` + `adj.npz` | `train/val/test.npz` + `adj_mx_dijsk.pkl` |
| windows (test, p12) | 7,518 | 7,539 |
| input shape | `[B, 1, 12, N]` | `[B, N, 12, 2]` |
| second channel | none | time-in-day (fraction of the day, UTC+8) |
| **normalisation** | **per section** | **one global scalar** |
| output | `[B, 1, 1, N]`, **one step** | `[B, 12, N]`, twelve steps |
| horizons | **one model per horizon** | one model covers all |

Training both under one loss means they have to see the **same window at the same
moment**. That is what this folder does.

### What is already the same

The window definition. STGCN's `data_transform` gives sample *i* the input rows
`i..i+11` and the target row `i+11+n_pred`; STGAT's offsets (x: −11..0, y: 1..12) give
the window at *t* the input rows `t−11..t` with horizon *p* at row `t+p`. Substituting
`g = i = t−11`, both read **"input `g..g+11`, horizon *p* targets `g+11+p`"**. The window
counts differ only because each repo trims the split edges its own way.

### What has to stay different

**The normalisation cannot be unified.** Feeding globally-scaled input to an STGCN
trained on per-section scaling **does not raise — it just returns nonsense**. So
`data.py` builds **two** tensors from the same rows, each on its own scale.

---

## Files

| File | What it does |
|---|---|
| `paths.py` | Loads both `model/` directories into one process under **aliases** (both repos ship a top-level package called `model`; 實驗記錄 §4.1 hit this collision before). Exposes each path's `features()` on a common `[B, N, C]`. |
| `data.py` | The unified dataloader: same rows → two input formats, two normalisations, time-in-day, target mask. |
| `model.py` | The eq. 3 gate + FC head. `--freeze` selects joint training or gate-only; `--single-path` bypasses the gate (control E). |
| **`verify.py`** | **Run this before training anything.** See below. |
| `train.py` | Training loop (masked loss, Adam 3e-4, 0.96 decay per epoch, early stopping, sidecar). |
| `evaluate.py` | Masked scoring, gate statistics, and `--dump-all`. |

### Hidden feature shapes (measured, not assumed)

```
STGCN   st_blocks(x)   [B, 64, Ko=4, N]     time not fully collapsed
        .output        [B, 1, 1, N]         one step -> hence one model per horizon

STGAT   blocks(x)      -> reshape ->        [B, N, 64]
        .output        [B, 12, N]           twelve steps from one model
```

**Both carry 64 channels**, so `W₁(s + t)` works directly with no projection to align
them. STGCN's extra time axis (Ko=4) is flattened and projected down; that `proj_cn` is
the only adapter in the model.

---

## Run `verify.py` first

```bash
cd fusion
python verify.py --device cpu
```

`data.py` rebuilds the windowing, both normalisations, the tensor layouts and the mask
from scratch. Get any one of them wrong — a transpose, the wrong scaler, an off-by-one in
the target row — and **training still runs, the loss still falls**, and the resulting MAE
is simply not comparable with anything in 實驗記錄. **None of those failures raise.**

So `verify.py` pushes the two **already-trained** backbones, with their own output heads,
through this dataloader and checks that they reproduce what the standalone evaluators
reported:

```
STGAT        3.3802 / 3.5127 / 3.6276      at 15 / 30 / 60 min
STGCN        3.5560 / 3.7535 / 3.9549
persistence  4.2872 / 4.6744 / 5.1281
```

If the pipeline is right they must; if it is not, they cannot. **Do not train until this
passes.** On the 202-section data it did: all nine within 0.1% (實驗記錄 §13.19).

> Persistence is the more sensitive of the two, because it uses no model at all: if it is
> off, the fault is in the rows or the mask.

> **One known window of difference.** Upstream's `num = len − n_his − n_pred` is short by
> one, so the last row of every split is never predicted. That off-by-one is not copied
> here, which leaves one extra window per split (7,519 vs 7,518 on test — under 0.02% of
> the cells, far inside the 2% tolerance).

---

## Order of operations

```bash
cd fusion

# 0. verify the pipeline (required)
python verify.py --device cpu

# 1. the reported configuration C: the proposal's joint training, plus a time-of-day
#    channel on the STGCN path (~91 s per epoch; best epoch 25, stopped at 37)
python train.py --freeze none --stgcn-tod --epochs 60 --batch 32 --patience 12 \
                --out checkpoints/fusion_c.pt

# 2. score it, and emit the dump the decision layer and integration/ha_baseline.py read
python evaluate.py --checkpoint checkpoints/fusion_c.pt --split test --dump-all
```

**Reported configuration** is `--freeze none --stgcn-tod` (checkpoint `fusion_c.pt`).
Two controls strip it back, one component at a time:

```bash
# C₀ — the same command minus the time-of-day channel
python train.py --freeze none --epochs 60 --batch 32 --patience 12 \
                --out checkpoints/fusion_c0.pt
python evaluate.py --checkpoint checkpoints/fusion_c0.pt --split test   # NOT --dump-all

# E — gate bypassed, one path only
python train.py --freeze none --single-path stgat \
                --epochs 60 --batch 32 --patience 12 \
                --out checkpoints/fusion_e.pt
python evaluate.py --checkpoint checkpoints/fusion_e.pt --split test    # NOT --dump-all
```

| 60 min MAE (test) | | what the row adds |
|---|---:|---|
| STGAT, trained on its own | 3.6276 | — |
| **E** — one path, no gate | **3.5557** | the training regime |
| **C₀** — + second path + gate | 3.5590 | dual path + gate |
| **C** — + time-of-day (reported) | 3.5579 | time-of-day |

**Every component of eq. 3 measures zero**; the whole 2.0% is the training regime. The
ceiling was visible from the start — the two models' errors correlate at 0.934 — and a
gate is only a rule for combining them. `--single-path` records itself in the meta and
`evaluate.py` reads it back: a single-path checkpoint still carries W1 and W3, so a
default rebuild would load cleanly and then score a gate that was never trained.

Two more settings exist but have no recorded run: `--freeze both` (gate and head only, with
`--stgcn-ckpt ../STGCN/STGCN_taichung_p12.pt`) and `--extended-gate` (the gate also sees
`s − t`, a per-section embedding and time-of-day). Control E answered the question they
were meant to probe more directly.

> **`--dump-all` overwrites the decision layer's input.** The dump name
> `integration/dump_fusion_<split>_p<N>.npz` does not encode which checkpoint produced
> it, and `integration/make_drl_input.py --source fusion` reads exactly those three
> files to build the graph the DRL agent was trained against. An ablation that dumps
> over them makes the reported 10-seed routing results silently irreproducible — no
> error, just different numbers. Only the reported checkpoint should ever dump. If an
> ablation's anomaly buckets are needed, back the three files up first and verify the
> restore with `md5sum -c`.

---

## Deviations from the proposal (state these in the report)

| Item | Proposal | Here | Why |
|---|---|---|---|
| Climate features into STGAT's node input | yes | **not used** | Measured effect 0.66 km/h, far below the models' own MAE of 3.47, and non-monotonic (實驗記錄 §13.9) |
| Loss | L2 (MSE) | **masked MAE** by default (`--loss mse` restores it) | MAE is what every number in 實驗記錄 is reported in; optimising a different quantity than the one reported lets the two diverge quietly |
| Masking | not specified | **external `mask.npy`** | 24.6% of cells are imputed and scoring on them is nearly free. METR-LA's `null_val=0.0` mechanism does **not** transfer — imputed cells hold real numbers, so there is no sentinel to test for (實驗設計 §2.3) |
| Per-path normalisation | not specified | **each path keeps its own** | Unifying it invalidates the pretrained weights and makes warm-starting impossible |
| Time-of-day into the STGCN path | implied — §4.3 assigns that path "尖峰時段、星期週期" | **added** (`--stgcn-tod`), and **measured to contribute nothing** | The shipped STGCN feeds on speed alone, which cannot express what the proposal asks that path to extract. Adding it is faithful to §4.3, but ablation C₀ puts its contribution at 0.03–0.43% on test and 0.06–0.18% on val — under the 1% noise floor. The −1.9% at 60 min comes from the training regime instead: control E reaches it with one path and no gate (實驗記錄 §13.22 ⑥–⑦) |
| Order inside eq. 3 | gate applied after W₂ and W₃ | gate applied first, then W₂ and W₃ (see the note under the equation) | Same parameters; W₃'s bias stays ungated. The report states eq. 3 as written and gives the implemented form beside it |

### The two `--freeze` settings answer different questions

| | `--freeze both` | `--freeze none` (the proposal) |
|---|---|---|
| what learns | gate + head only | everything |
| can it break the 0.934 error correlation | **no** — two frozen backbones are wrong in the same places, and no gate invents independent information | **in principle** — the gate gives the paths a reason to specialise. Measured, it did not: the gate settled on the fixed weight and the second path added nothing (§13.22 ③⑦) |
| one model, three horizons | **no** — the STGCN backbone is bound to a single horizon | **yes** — the new FC head emits all twelve steps |
| cost | minutes | one full training run |

實驗記錄 §13.7 already measured the ceiling for the frozen case: the best **fixed** weight
beats STGAT alone by **0.40%**, and the two models' errors correlate at **0.934**. So
`--freeze both` is expected to gain almost nothing — its value would be to **confirm that
ceiling cheaply**, not to produce a result. It was not needed in the end: control E
settled the question with the jointly trained model.

---

## How to read the result

`evaluate.py` prints **gate statistics**, and that is the real answer:

```
mean opening                  0.xxxx    0 = all STGCN, 1 = all STGAT
spread ACROSS SECTIONS        0.xxxx    <- is the per-node gating doing anything
spread ACROSS TIME            0.xxxx    <- is the per-timestep gating doing anything
```

**Both spreads near zero → the model has learned a fixed weighted average**, reproducing
the constant §13.7 already measured, and any MAE difference is noise. The script says so
in the output rather than leaving it to be noticed.

If the spreads are clearly non-zero **and** MAE improves by more than 1%, that is
evidence that per-node, per-timestep gating does something a global constant cannot —
which is the reason the proposal specifies this design.

> The criterion is written here in advance, to avoid rationalising afterwards:
> **an improvement below 1% is reported as "no measurable gain"**, because the best fixed
> weight is already worth 0.40%.

**Outcome (C, test split).** Mean opening 0.8304, spread 0.0350 across sections and
0.0515 across time: the gate moves, but its average is the fixed weight §13.7 found
(0.2/0.8, so the output is about four-fifths STGAT). On the same 7,519 windows C beats
STGAT by 0.14 / 1.03 / 2.01% at 15 / 30 / 60 min, so by the criterion above 15 min shows
no gain, 30 min just crosses the line and 60 min gains about 2%. Controls C₀ and E then
put all of it on the training regime rather than on the gate (§13.22 ③⑥⑦).

---

## What this folder does not touch

- **No changes to `STGCN/` or `STGAT/`.** The backbones are loaded under aliases rather
  than vendored, so nothing drifts from upstream.
- **No changes to the decision layer's code in `integration/`.**
- **One file connects it to the decision layer.** `evaluate.py --dump-all` writes
  `integration/dump_fusion_<split>_p<N>.npz`, and `integration/make_drl_input.py
  --source fusion` turns the 15-minute dump into baseline 4's forecast and policy 7's
  `tpred` (實驗記錄 §13.24). The agent was retrained on it and all six paired deltas
  (S2 and S3, three metrics) stayed within one standard deviation, as §13.13 predicted:
  at 14.1% coverage *by length*, prediction has no measurable effect on routing. That
  dump is also why `--dump-all` must only ever be run on the reported checkpoint.
