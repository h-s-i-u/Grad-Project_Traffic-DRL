#!/usr/bin/env python3
"""HA (Historical Average) baseline for the Taichung prediction module.

計劃書 §4.6 requires the prediction module to be compared against "STGCN, STGAT, HA
(Historical Average)". Persistence has been in place since 實驗記錄 §5 (it is what
exposed STGAT's masking bug), but HA was missing. This adds it.

HA is not a model: it needs no checkpoint, no GPU and no inference run. It predicts,
for each (section, target row), the mean speed observed on the SAME weekday and the
SAME time of day, averaged over the TRAIN split only:

    HA(i, r) = mean{ v[i, r'] : r' in train,
                     weekday(r') == weekday(r), timeofday(r') == timeofday(r),
                     mask[i, r'] == True }

Two things it is easy to get wrong, both silent:

  TIMEZONE   DataCollectTime is UTC and Taiwan is UTC+8. Relabelling the time-of-day
             slots does not change MAE on its own, but the WEEKDAY boundary moves by
             8 hours, so Monday 00:00-08:00 local would be filed under Sunday. Rush
             hour is a weekday phenomenon, so that smears exactly the structure HA
             exists to capture. Everything below runs on UTC+8.

  WINDOWS    The score is only comparable with STGCN/STGAT if it is computed on the
             SAME target rows. STGCN's dataloader.data_transform yields
             `len(split) - n_his - n_pred` samples, sample i targeting split row
             `i + n_his + n_pred - 1`. That formula is reproduced here -- and then
             CHECKED, by computing persistence on the same rows and comparing it with
             the value STGCN/evaluate_masked.py already reported. A mismatch means
             the rows are wrong, and the run aborts instead of printing a plausible
             but incomparable number.

Metrics follow STGCN/evaluate_masked.py exactly, including the MAPE denominator floor
(a true 0 km/h survives the float32 z-score round trip as ~1.3e-6 and one such cell
pushed MAPE to 6356%).

--------------------------------------------------------------------------------
WHY HA IS MORE THAN A CHECKBOX

HA and persistence fail in opposite ways: persistence knows the current state and
nothing about structure, HA knows the weekly structure and nothing about today.
Beating only one is weak evidence -- a model that beat persistence alone could be a
smoothed weekly average, and one that beat HA alone could be copying the last frame.
Beating both, at every horizon, is what shows the model is combining current
observation with learned spatio-temporal structure.

HA also caps the long-horizon claim honestly. Against persistence the margin GROWS
with horizon (-21/-25/-29%); against HA it SHRINKS (-16.5/-13.2/-10.4%), because HA
does not degrade with horizon at all. Quoting only the first would credit the model
for persistence getting worse.

And it lets us test the proposal's own premise. 計劃書 §4.3 assigns the STGCN path
the "規則性" (regular) component and the STGAT path the "異常性" (anomalous) one. HA
is a pure regularity predictor, so if the dual-path design is doing what it claims,
the model's advantage over HA should CONCENTRATE in the moments when today departs
from the weekly norm. That is what the anomaly-bucket table below measures.

    TRAP AVOIDED: bucketing by |truth - HA| would be circular -- HA is bad in the
    high bucket by construction. Anomaly is therefore defined on the INPUT WINDOW
    (how far the last 12 observed steps ran from their own weekly norm), which every
    predictor can see at prediction time and which says nothing about the target.
--------------------------------------------------------------------------------

Usage:
    cd integration
    python ha_baseline.py                      # test split, all three horizons
    python ha_baseline.py --split val
    python ha_baseline.py --n-pred 3           # just 15 min

    # add the model columns (needs the dumps search_ensemble_weight.py also uses)
    cd ../STGCN && python run_infer_taichung.py --split test --dump-all --n-pred 3 \
                          --checkpoint STGCN_taichung_p3.pt --device cpu
    cd ../STGAT && python run_infer_taichung.py --split test --dump-all --n-pred 3 \
                          --device cpu
    cd ../integration && python ha_baseline.py --n-pred 3
"""
import argparse
import math

import numpy as np
import pandas as pd

import config as C

MAP_DIR = C.ROOT / "Map"

N_HIS = 12                  # STGCN/STGAT both train on 12 input steps (1 hour)
VAL_AND_TEST_RATE = 0.15    # mirrors STGCN/main.py
STEPS_PER_DAY = 288         # 5-minute sampling
TAIWAN_OFFSET_H = 8

# MAPE denominator floor, copied from STGCN/evaluate_masked.py so the two tools cannot
# drift apart. See that file for why `!= 0` is not enough.
MAPE_MIN_SPEED = 1.0

# Masked persistence MAE already reported by STGCN/evaluate_masked.py on the TEST split
# (實驗記錄 §13.11 ②). Used only to verify that the rows selected here are the same
# ones -- if this tool's persistence differs, its HA number is not comparable either.
# STGAT's own evaluator gets 4.2852 / 4.6712 / 5.1233 from a slightly different window
# count (7,539 vs 7,527), which is why the tolerance is 1% rather than exact.
KNOWN_PERSISTENCE = {"test": {3: 4.2872, 6: 4.6744, 12: 5.1281}}
PERSISTENCE_TOL = 0.01


def metrics(y_true, y_pred, valid, mape_floor=MAPE_MIN_SPEED):
    """MAE / RMSE / MAPE / WMAPE. `valid=None` means no masking."""
    if valid is not None:
        y_true, y_pred = y_true[valid], y_pred[valid]
    d = np.abs(y_true - y_pred)
    ok = y_true >= mape_floor
    return {
        "n": int(y_true.size),
        "MAE": float(d.mean()),
        "RMSE": float(np.sqrt((d ** 2).mean())),
        "MAPE": float((d[ok] / y_true[ok]).mean()) if ok.any() else float("nan"),
        "WMAPE": float(d.sum() / y_true.sum()) if y_true.sum() else float("nan"),
        "mape_excluded": int((~ok).sum()),
    }


def split_sizes(n_rows, rate=VAL_AND_TEST_RATE):
    """Mirror STGCN/main.py's split exactly."""
    len_val = int(math.floor(n_rows * rate))
    len_test = int(math.floor(n_rows * rate))
    len_train = int(n_rows - len_val - len_test)
    return len_train, len_val, len_test


def target_rows(n_rows, split, n_pred, n_his=N_HIS):
    """Absolute rows of vel.csv that a model predicts for this (split, horizon).

    data_transform makes `len(split) - n_his - n_pred` samples and sample i targets
    split row `i + n_his + n_pred - 1`; adding the split's offset maps it back to an
    absolute row.
    """
    len_train, len_val, len_test = split_sizes(n_rows)
    if split == "val":
        offset, length = len_train, len_val
    else:
        offset, length = len_train + len_val, n_rows - len_train - len_val
    n_sample = length - n_his - n_pred
    if n_sample <= 0:
        raise ValueError(f"{split} split too short for n_his={n_his}, n_pred={n_pred}")
    return offset + np.arange(n_sample) + n_his + n_pred - 1, len_train


def slot_index(timestamps):
    """(weekday, time-of-day) collapsed to one integer per row, in Taiwan time.

    weekday * 288 + step_of_day, so the table is a plain 2-D array instead of a dict
    of tuples -- which also makes the "unseen slot" fallback a simple NaN test.
    """
    t = pd.to_datetime(timestamps, utc=True) + pd.Timedelta(hours=TAIWAN_OFFSET_H)
    step = (t.dt.hour * 60 + t.dt.minute) // 5
    return (t.dt.dayofweek.to_numpy() * STEPS_PER_DAY + step.to_numpy()).astype(np.int64)


def build_ha_table(vel, mask, slots, len_train):
    """Per-(slot, section) mean over REAL observations in the train split.

    Imputed cells are excluded for the same reason the scoring is masked: ~25% of the
    matrix is ffill, i.e. a copy of the previous step, and averaging those in would
    make HA a smoothed version of persistence rather than a periodic baseline.
    """
    n_slots = 7 * STEPS_PER_DAY
    n_sec = vel.shape[1]
    tr_slots = slots[:len_train]
    v, m = vel[:len_train], mask[:len_train]

    total = np.zeros((n_slots, n_sec), dtype=np.float64)
    count = np.zeros((n_slots, n_sec), dtype=np.int64)
    np.add.at(total, tr_slots, np.where(m, v, 0.0))
    np.add.at(count, tr_slots, m.astype(np.int64))

    table = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)

    # Fallbacks, in order: this section's train mean, then the global train mean. A
    # slot with no observation at all is not hypothetical -- some sections are 50%
    # imputed, and a 7x288 grid over 24 weeks leaves roughly 24 samples per cell.
    sec_mean = np.divide(total.sum(0), count.sum(0),
                         out=np.full(n_sec, np.nan), where=count.sum(0) > 0)
    global_mean = float(total.sum() / max(1, count.sum()))
    sec_mean = np.where(np.isnan(sec_mean), global_mean, sec_mean)
    empty = np.isnan(table)
    table = np.where(empty, sec_mean[None, :], table)

    return table, {
        "empty_cells": int(empty.sum()),
        "total_cells": int(empty.size),
        "obs_per_cell": float(count[count > 0].mean()) if (count > 0).any() else 0.0,
        "global_mean": global_mean,
    }


def load_dumps(split, n_pred):
    """Model predictions, if the relevant --dump-all has been run.

    Returns {name: (rows, pred)} or {} -- the model columns are optional so that the
    HA/persistence floor can always be produced without a GPU or a checkpoint.

    "fusion" comes from fusion/evaluate.py --dump-all and is absent until the dual-path
    model of 計劃書 §4.3 has been trained; the other two come from the two
    run_infer_taichung.py scripts.
    """
    out = {}
    for name in ("stgcn", "stgat", "fusion"):
        p = C.HERE / f"dump_{name}_{split}_p{n_pred}.npz"
        if p.is_file():
            d = np.load(p)
            out[name] = (d["rows"], d["pred"].astype(np.float64))
    return out


def input_anomaly(vel, mask, table, slots, rows, n_pred, n_his=N_HIS):
    """How far each window's INPUT ran from its own weekly norm, per (window, section).

    Causal by construction: it reads only rows r-n_pred-n_his+1 .. r-n_pred, which are
    exactly the 12 steps the models were fed, so it is available at prediction time and
    carries no information about the target. Bucketing by |truth - HA| instead would be
    circular -- HA is bad in the high bucket by definition, and the "finding" would be
    a restatement of the bucketing rule.

    Imputed cells are excluded from the average; a window with no real observation at
    all for a section comes back NaN and is dropped from the buckets rather than being
    silently filed as "not anomalous".
    """
    num = np.zeros((len(rows), vel.shape[1]), dtype=np.float64)
    den = np.zeros_like(num)
    for k in range(n_his):
        rr = rows - n_pred - n_his + 1 + k
        d = np.abs(vel[rr] - table[slots[rr]])
        m = mask[rr]
        num += np.where(m, d, 0.0)
        den += m
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def bucket_table(name, groups, preds, truth, keep):
    """MAE per group for every predictor, plus each model's margin over HA.

    `groups` is (label, boolean-mask-over-cells); `preds` maps predictor name -> array
    aligned with `truth`. Everything is restricted to `keep` (real observations with a
    usable anomaly score) before grouping.
    """
    names = list(preds)
    head = f"  {name:<28}{'cells':>9}" + "".join(f"{n:>10}" for n in names)
    has_model = any(n not in ("HA", "persist.") for n in names)
    if has_model:
        head += f"{'best vs HA':>12}"
    lines = [head, "  " + "-" * (len(head) - 2)]
    for label, sel in groups:
        g = sel & keep
        if not g.any():
            continue
        maes = {n: float(np.abs(preds[n][g] - truth[g]).mean()) for n in names}
        row = f"  {label:<28}{int(g.sum()):>9,}" + "".join(f"{maes[n]:>10.4f}" for n in names)
        if has_model:
            best = min(v for n, v in maes.items() if n not in ("HA", "persist."))
            row += f"{100 * (best - maes['HA']) / maes['HA']:>11.1f}%"
        lines.append(row)
    return "\n".join(lines)


def anomaly_report(vel, mask, table, slots, rows, n_pred, cli, truth, valid, ha, pers):
    """Does the model's edge over HA concentrate where today departs from the norm?

    計劃書 §4.3 splits the architecture into a "規則性" path and an "異常性" path. HA
    is regularity with nothing else, so this is the cheapest available test of that
    premise -- and it needs no training, only the dumps the ensemble search already
    uses. Without them the table still prints, with HA and persistence alone.
    """
    anom = input_anomaly(vel, mask, table, slots, rows, n_pred, cli.n_his)
    preds = {"HA": ha, "persist.": pers}

    dumps = load_dumps(cli.split, n_pred)
    if dumps:
        # The two models' window counts differ (7,527 vs 7,539), so align on the
        # ABSOLUTE row rather than assuming index-for-index correspondence -- the same
        # trap the single-shot inference path hit (實驗記錄 §13.8).
        shared = rows
        for r, _ in dumps.values():
            shared = np.intersect1d(shared, r)
        keep_local = np.isin(rows, shared)
        for name, (r, p) in dumps.items():
            aligned = np.full((len(rows), vel.shape[1]), np.nan)
            idx = np.searchsorted(r, rows[keep_local])
            aligned[keep_local] = np.clip(p[idx], C.TAICHUNG_SPEED_MIN_KMH,
                                          C.TAICHUNG_SPEED_MAX_KMH)
            preds[name] = aligned
        if "stgcn" in preds and "stgat" in preds:
            w = C.W_STGCN / (C.W_STGCN + C.W_STGAT)
            preds["hybrid"] = w * preds["stgcn"] + (1 - w) * preds["stgat"]
        # Unchanged when stgat or stgcn is present, which is every existing run. The
        # fallback only covers a dumps set that has neither -- i.e. fusion on its own.
        anchor = ("stgat" if "stgat" in preds else
                  "stgcn" if "stgcn" in preds else next(iter(dumps)))
        usable = np.isfinite(preds[anchor])
        print(f"\n  model columns: {', '.join(dumps)} on {int(keep_local.sum()):,} of "
              f"{len(rows):,} windows shared with the dumps"
              + (f"; hybrid at w={w:.2f}/{1 - w:.2f} (chosen on val, 實驗記錄 §13.7)"
                 if "hybrid" in preds else ""))
    else:
        usable = np.ones_like(valid)
        print(f"\n  (no dump_stg*_{cli.split}_p{n_pred}.npz -- HA and persistence only. "
              f"Produce them with run_infer_taichung.py --dump-all --device cpu to add "
              f"the model columns.)")

    keep = valid & np.isfinite(anom) & usable
    if not keep.any():
        print("  no scorable cells for the anomaly breakdown")
        return

    edges = np.quantile(anom[keep], np.linspace(0, 1, cli.buckets + 1))
    edges[-1] = np.inf
    groups = []
    for b in range(cli.buckets):
        sel = (anom >= edges[b]) & (anom < edges[b + 1])
        lo, hi = edges[b], edges[b + 1]
        tag = ("Q1 most routine" if b == 0 else
               f"Q{cli.buckets} most unusual" if b == cli.buckets - 1 else f"Q{b + 1}")
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:.1f}"
        groups.append((f"{tag} ({lo:.1f}-{hi_s})", sel))

    print(f"\n  --- input-window anomaly, MAE per bucket ---")
    print(bucket_table("anomaly bucket (km/h)", groups, preds, truth, keep))

    # A second cut that needs no anomaly score and is easier to read in a report.
    t = slots[rows] // STEPS_PER_DAY                     # 0=Mon .. 6=Sun
    tod = (slots[rows] % STEPS_PER_DAY) * 5              # minutes since local midnight
    dow = np.repeat(t[:, None], vel.shape[1], axis=1)
    mins = np.repeat(tod[:, None], vel.shape[1], axis=1)
    peak = (((mins >= 420) & (mins < 540)) | ((mins >= 1020) & (mins < 1140)))
    weekday = dow < 5
    print(f"\n  --- by period (Taiwan time; peak = 07-09 and 17-19) ---")
    print(bucket_table("period", [
        ("weekday peak", weekday & peak),
        ("weekday off-peak", weekday & ~peak),
        ("weekend", ~weekday),
    ], preds, truth, keep))
    print("  A model whose margin over HA widens toward the unusual buckets is doing "
          "what 計劃書\n  §4.3 assigns to the STGAT path; a flat margin would mean it "
          "is mostly reproducing\n  the weekly pattern that HA already has.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="report on test (default); val only for sanity checks")
    ap.add_argument("--n-pred", type=int, default=None, choices=[3, 6, 12],
                    help="single horizon; default runs 3, 6 and 12 (15/30/60 min)")
    ap.add_argument("--n-his", type=int, default=N_HIS)
    ap.add_argument("--no-check", action="store_true",
                    help="skip the persistence cross-check (not recommended: it is the "
                         "only thing verifying that these rows match the models')")
    ap.add_argument("--buckets", type=int, default=4,
                    help="quantile buckets for the anomaly analysis (default 4)")
    ap.add_argument("--no-anomaly", action="store_true",
                    help="skip the anomaly / time-of-day breakdown")
    cli = ap.parse_args()

    vel = pd.read_csv(MAP_DIR / "taichung_vel.csv",
                      encoding="utf-8-sig").to_numpy(np.float64)
    mask = np.load(MAP_DIR / "taichung_mask.npy")
    stamps = pd.read_csv(MAP_DIR / "taichung_timestamps.csv",
                         encoding="utf-8-sig").iloc[:, 0]
    if vel.shape != mask.shape:
        raise SystemExit(f"ERROR  vel {vel.shape} and mask {mask.shape} disagree -- both "
                         f"come from build_speed.py, so re-run it")
    if len(stamps) != len(vel):
        raise SystemExit(f"ERROR  {len(stamps)} timestamps for {len(vel)} rows")

    n_rows, n_sec = vel.shape
    len_train, len_val, len_test = split_sizes(n_rows)
    slots = slot_index(stamps)
    table, info = build_ha_table(vel, mask, slots, len_train)

    print("=== data ===")
    print(f"  {n_rows:,} rows x {n_sec} sections | train {len_train:,} / "
          f"val {len_val:,} / test {len_test:,} (70/15/15, by time)")
    print(f"  real observations overall {mask.mean():.1%}; "
          f"train only {mask[:len_train].mean():.1%}")
    print(f"  HA table 7 x {STEPS_PER_DAY} slots x {n_sec} sections, built on TRAIN "
          f"observations only, Taiwan time (UTC+{TAIWAN_OFFSET_H})")
    print(f"  mean observations per filled cell {info['obs_per_cell']:.1f}; "
          f"{info['empty_cells']:,} of {info['total_cells']:,} cells "
          f"({info['empty_cells'] / info['total_cells']:.2%}) had none and fall back to "
          f"the section mean")

    horizons = [cli.n_pred] if cli.n_pred else [3, 6, 12]
    rows_out = []
    for n_pred in horizons:
        rows, _ = target_rows(n_rows, cli.split, n_pred, cli.n_his)
        truth = vel[rows]
        valid = mask[rows]
        ha = table[slots[rows]]
        pers = vel[rows - n_pred]          # last observed input frame, as evaluate_masked

        m_ha = metrics(truth, ha, valid)
        m_pe = metrics(truth, pers, valid)
        u_ha = metrics(truth, ha, None)

        ref = KNOWN_PERSISTENCE.get(cli.split, {}).get(n_pred)
        if ref is not None:
            rel = abs(m_pe["MAE"] - ref) / ref
            ok = rel <= PERSISTENCE_TOL
            flag = "ok" if ok else "MISMATCH"
            print(f"\n=== horizon {n_pred} ({n_pred * 5} min) ===")
            print(f"  {len(rows):,} windows, rows {rows.min()}..{rows.max()} | "
                  f"persistence check {m_pe['MAE']:.4f} vs {ref:.4f} "
                  f"({rel:+.2%}) -> {flag}")
            if not ok and not cli.no_check:
                raise SystemExit(
                    f"\nERROR  persistence on these rows is {m_pe['MAE']:.4f} but "
                    f"STGCN/evaluate_masked.py reported {ref:.4f} for the same split "
                    f"and horizon.\n"
                    f"       The rows selected here are therefore NOT the ones the "
                    f"models were scored on, so\n"
                    f"       the HA number would not be comparable. Check n_his "
                    f"({cli.n_his}), the split rate\n"
                    f"       ({VAL_AND_TEST_RATE}) and whether build_speed.py has been "
                    f"re-run since §13.11.")
        else:
            print(f"\n=== horizon {n_pred} ({n_pred * 5} min) ===")
            print(f"  {len(rows):,} windows, rows {rows.min()}..{rows.max()} | "
                  f"persistence {m_pe['MAE']:.4f} (no reference value for "
                  f"split={cli.split})")

        print(f"  {'':<14}{'MAE':>9}{'RMSE':>9}{'MAPE':>9}{'WMAPE':>9}")
        print(f"  {'HA (masked)':<14}{m_ha['MAE']:>9.4f}{m_ha['RMSE']:>9.4f}"
              f"{m_ha['MAPE']:>8.2%}{m_ha['WMAPE']:>9.2%}")
        print(f"  {'HA (unmasked)':<14}{u_ha['MAE']:>9.4f}{u_ha['RMSE']:>9.4f}"
              f"{u_ha['MAPE']:>8.2%}{u_ha['WMAPE']:>9.2%}")
        print(f"  {'persistence':<14}{m_pe['MAE']:>9.4f}{m_pe['RMSE']:>9.4f}"
              f"{m_pe['MAPE']:>8.2%}{m_pe['WMAPE']:>9.2%}")
        rows_out.append((n_pred, m_ha, m_pe))

        if not cli.no_anomaly:
            anomaly_report(vel, mask, table, slots, rows, n_pred, cli, truth, valid,
                           ha, pers)

    # --- Track A floor, assembled ---------------------------------------------
    # The model numbers are quoted from 實驗記錄 §13.11 (202-section models, masked,
    # test split) purely so this prints a complete table. They are NOT recomputed here:
    # if the prediction models are retrained, re-read them from evaluate_masked.py
    # rather than trusting these.
    MODEL_REF = {3: (3.3802, 3.5560), 6: (3.5127, 3.7535), 12: (3.6276, 3.9549)}
    if cli.split == "test":
        print(f"\n=== Track A floor (計劃書 §4.6) ===")
        print(f"  {'horizon':<10}{'HA':>9}{'persist.':>10}{'STGAT*':>9}{'STGCN*':>9}"
              f"{'STGAT vs HA':>13}")
        print("  " + "-" * 60)
        for n_pred, m_ha, m_pe in rows_out:
            at, cn = MODEL_REF.get(n_pred, (float("nan"),) * 2)
            gain = 100 * (at - m_ha["MAE"]) / m_ha["MAE"]
            print(f"  {n_pred * 5:>3} min   {m_ha['MAE']:>9.4f}{m_pe['MAE']:>10.4f}"
                  f"{at:>9.4f}{cn:>9.4f}{gain:>12.1f}%")
        print("  * from 實驗記錄 §13.11, not recomputed here -- re-read them if the "
              "models are retrained.")
        print("\n  HA and persistence are the two floors the prediction module has to "
              "clear. They\n  are not bounds: a model CAN score worse than either. "
              "Read them per horizon --\n  persistence is strong at 15 min (inertia) "
              "and HA at 60 min (weekly periodicity),\n  so a 12-step average hides "
              "which one was actually beaten.")


if __name__ == "__main__":
    main()
