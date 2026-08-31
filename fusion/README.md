# Gated Fusion — the dual-path model of 計劃書 §4.3

> STGCN path (regular structure) + STGAT path (anomalous structure), joined by the
> proposal's eq. 3 gate, with an FC head emitting 15 / 30 / 60 min.
>
> See also [`../paper_work/實驗設計.md`](../paper_work/實驗設計.md) §3.2 and
> [`../paper_work/實驗記錄_DRL決策模組.md`](../paper_work/實驗記錄_DRL決策模組.md)
> §13.7, §13.11, §13.18.

```
gate   = σ( W₁ · (s + t) )
Fusion = tanh( W₂ · t · gate + (I − gate) · W₃ · s )
output = FC(Fusion)                    → 15 / 30 / 60 min
```

`s` and `t` are the two paths' **hidden features**, not their finished forecasts. That is
where the proposal puts the fusion: between the backbones and the output head.

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
| `model.py` | The eq. 3 gate + FC head. `--freeze` selects joint training or gate-only. |
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

## 🔴 Run `verify.py` first

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
passes.**

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

# 1. the proposal: both paths train
python train.py --freeze none --out checkpoints/fusion_joint.pt

# 2. control: gate only, backbones frozen
python train.py --freeze both --stgcn-ckpt ../STGCN/STGCN_taichung_p12.pt \
                --out checkpoints/fusion_frozen.pt

# 3. ablation: the gate also sees a per-section embedding and time-of-day
python train.py --freeze none --extended-gate --out checkpoints/fusion_ext.pt

# 4. score, and emit a dump integration/ha_baseline.py can read
python evaluate.py --checkpoint checkpoints/fusion_joint.pt --split test --dump-all
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

> 🔴 **`--dump-all` overwrites the decision layer's input.** The dump name
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
| Climate features into STGAT's node input | ✅ | ❌ **not used** | Measured effect 0.66 km/h, far below the models' own MAE of 3.47, and non-monotonic (實驗記錄 §13.9) |
| Loss | L2 (MSE) | **masked MAE** by default (`--loss mse` restores it) | MAE is what every number in 實驗記錄 is reported in; optimising a different quantity than the one reported lets the two diverge quietly |
| Masking | not specified | **external `mask.npy`** | 24.6% of cells are imputed and scoring on them is nearly free. METR-LA's `null_val=0.0` mechanism does **not** transfer — imputed cells hold real numbers, so there is no sentinel to test for (實驗設計 §2.3) |
| Per-path normalisation | not specified | **each path keeps its own** | Unifying it invalidates the pretrained weights and makes warm-starting impossible |
| Time-of-day into the STGCN path | implied — §4.3 assigns that path "尖峰時段、星期週期" | **added** (`--stgcn-tod`), and **measured to contribute nothing** | The shipped STGCN feeds on speed alone, which cannot express what the proposal asks that path to extract. Adding it is faithful to §4.3, but ablation C₀ puts its contribution at 0.03–0.43% on test and 0.06–0.18% on val — under the 1% noise floor. The −1.9% at 60 min comes from end-to-end joint training instead (實驗記錄 §13.22 ⑥) |

### The two `--freeze` settings answer different questions

| | `--freeze both` | `--freeze none` (the proposal) |
|---|---|---|
| what learns | gate + head only | everything |
| can it break the 0.934 error correlation | ❌ **no** — two frozen backbones are wrong in the same places, and no gate invents independent information | ✅ **yes** — the gate finally gives the paths a reason to specialise |
| one model, three horizons | ❌ the STGCN backbone is bound to a single horizon | ✅ the new FC head emits all twelve steps |
| cost | minutes | one full training run |

實驗記錄 §13.7 already measured the ceiling for the frozen case: the best **fixed** weight
beats STGAT alone by **0.40%**, and the two models' errors correlate at **0.934**. So
`--freeze both` is expected to gain almost nothing — its value is to **confirm that
ceiling cheaply**, not to produce a result.

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

---

## What this folder does not touch

- **No changes to `STGCN/` or `STGAT/`.** The backbones are loaded under aliases rather
  than vendored, so nothing drifts from upstream.
- **No changes to the decision layer in `integration/`.**
- **Not wired into `make_drl_input.py`.** 實驗記錄 §13.13 established that at 14.1%
  coverage *by length* prediction has no measurable effect on routing (86% of routes are
  identical), so feeding fusion into the decision layer would change no number there and
  would only add another setting that can drift. **Fusion is a Track A result.**
