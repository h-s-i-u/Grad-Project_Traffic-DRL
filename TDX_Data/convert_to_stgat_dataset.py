# -*- coding: utf-8 -*-
"""
convert_to_stgat_dataset.py
──────────────────────────
Turn build_speed.py's output into the DCRNN-style dataset STGAT reads.

STGAT and STGCN want completely different layouts:

    STGCN  data/taichung/vel.csv + adj.npz      (time x section) matrix + scipy sparse
    STGAT  data/taichung/{train,val,test}.npz   pre-windowed (samples, 12, sections, 2)
           data/taichung/adj_mx_dijsk.pkl       (sensor_ids, id_to_ind, adj) triple

Two consequences worth knowing:
  1. STGAT emits ALL 12 steps at once, so ONE training run covers 15/30/60 min.
     STGCN is single-step, so it needs one model per horizon.
  2. Feature 1 is time-of-day (fraction of a day). The raw timestamps are UTC;
     they are shifted to UTC+8 here so the daily cycle lines up with Taiwan's
     actual peak hours.

Inputs (all from Map/, produced by build_speed.py):
    taichung_vel.csv, taichung_mask.npy, taichung_timestamps.csv,
    taichung_section_index.csv, taichung_adj.npy

Outputs (STGAT/data/taichung/):
    train.npz / val.npz / test.npz   x, y      : (samples, 12, sections, 2)
                                     x_mask, y_mask : (samples, 12, sections) bool
    adj_mx_dijsk.pkl                 (sensor_ids, sensor_id_to_ind, adj_mx)

Usage:
    python convert_to_stgat_dataset.py

    then:
    cd STGAT && python train.py --cuda --data data/taichung/ \\
        --adj_filename data/taichung/adj_mx_dijsk.pkl \\
        --num_of_vertices 202 --params_dir experiment_taichung \
        --lr 3e-4 --epoch 500 --early_stop_maxtry 40
"""

import argparse
import os
import pickle
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
MAP_DIR = os.path.join(ROOT_DIR, "Map")

DATASET_NAME = "taichung"
N_HIS, N_PRED = 12, 12       # STGAT: 12 steps in, 12 steps out
TZ_OFFSET_HOURS = 8          # TDX DataCollectTime is UTC; Taiwan is UTC+8
# Match STGCN's main.py (which hard-codes 0.15/0.15) so both models are tested on the
# same stretch of time -- otherwise their MAEs cannot be put side by side.
VAL_RATE = TEST_RATE = 0.15


def build_windows(speed, tod, mask, n_his, n_pred):
    """Slide DCRNN-style windows over the series.

    x = t-11..t (inclusive of now), y = t+1..t+12, matching DCRNN's
    generate_training_data.py. Returns x, y as (samples, steps, sections, 2) and
    the masks as (samples, steps, sections).
    """
    T, N = speed.shape
    num = T - n_his - n_pred + 1
    if num <= 0:
        raise ValueError(f"only {T} timesteps -- too few for a {n_his}+{n_pred} window")

    # Stack (speed, time-of-day) into (T, N, 2) once; time-of-day is shared by all nodes.
    feat = np.stack([speed, np.repeat(tod[:, None], N, axis=1)], axis=-1).astype(np.float32)

    x = np.empty((num, n_his, N, 2), dtype=np.float32)
    y = np.empty((num, n_pred, N, 2), dtype=np.float32)
    xm = np.empty((num, n_his, N), dtype=bool)
    ym = np.empty((num, n_pred, N), dtype=bool)
    for i in range(num):
        x[i] = feat[i:i + n_his]
        y[i] = feat[i + n_his:i + n_his + n_pred]
        xm[i] = mask[i:i + n_his]
        ym[i] = mask[i + n_his:i + n_his + n_pred]
    return x, y, xm, ym


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stgat-dir", default=os.path.join(ROOT_DIR, "STGAT"),
                    help="STGAT project path (defaults to this repo's STGAT/)")
    ap.add_argument("--source-dir", default=MAP_DIR, help="build_speed.py output folder")
    ap.add_argument("--val-rate", type=float, default=VAL_RATE)
    ap.add_argument("--test-rate", type=float, default=TEST_RATE)
    ap.add_argument("--missing-as-zero", action="store_true",
                    help="write imputed cells back as 0 so STGAT's built-in "
                         "masked_mae(null_val=0.0) skips them during training. NOTE: this "
                         "makes STGAT's training data differ from STGCN's, so their MAEs "
                         "are no longer comparable (STGCN uses an unmasked MSELoss and "
                         "would learn to predict 0). Default off = both models see the "
                         "same imputed data and honesty comes from the external mask.")
    args = ap.parse_args()

    src = args.source_dir
    paths = {n: os.path.join(src, f"taichung_{n}") for n in
             ("vel.csv", "mask.npy", "timestamps.csv", "section_index.csv", "adj.npy")}
    for p in paths.values():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing {os.path.basename(p)}: {p}\nRun build_speed.py first.")

    print("[1/5] reading build_speed.py output...")
    speed = pd.read_csv(paths["vel.csv"], encoding="utf-8-sig").to_numpy(dtype=np.float32)
    mask = np.load(paths["mask.npy"])
    ts = pd.read_csv(paths["timestamps.csv"], encoding="utf-8-sig")
    sec = pd.read_csv(paths["section_index.csv"], encoding="utf-8-sig").sort_values("matrix_index")
    adj = np.load(paths["adj.npy"])
    T, N = speed.shape
    print(f"      -> {T:,} timesteps x {N} sections; adjacency {adj.shape}")
    for arr, name, shape in ((mask, "mask", (T, N)), (adj, "adj", (N, N))):
        if arr.shape != shape:
            raise ValueError(f"{name} is {arr.shape}, expected {shape} -- all files must "
                             f"come from the same build_speed.py run")
    if len(sec) != N:
        raise ValueError(f"section_index has {len(sec)} rows but the data has {N} columns")

    if args.missing_as_zero:
        n_zeroed = int((~mask).sum())
        speed = np.where(mask, speed, 0.0).astype(np.float32)
        print(f"      -> --missing-as-zero: {n_zeroed:,} imputed cells written back as 0 "
              f"(STGAT's masked_mae will skip them)")

    print("[2/5] building the time-of-day feature (UTC -> UTC+8)...")
    # tz_convert rather than adding a Timedelta: the values come out the same, but the
    # offset is carried on the dtype, so the printout below is not mislabelled +00:00.
    # stdlib timezone, not pd.FixedOffset: the latter is not public pandas API and is
    # missing in some builds (AttributeError under WSL).
    t = pd.to_datetime(ts.iloc[:, 0], utc=True).dt.tz_convert(
        timezone(timedelta(hours=TZ_OFFSET_HOURS)))
    if len(t) != T:
        raise ValueError(f"{len(t)} timestamps but {T} data rows")
    # Fraction of the day in [0, 1), the same definition DCRNN uses for feature 1.
    tod = ((t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second) / 86400.0).to_numpy(np.float32)
    print(f"      -> local time {t.iloc[0]} .. {t.iloc[-1]}")

    print(f"[3/5] windowing ({N_HIS} steps in -> {N_PRED} steps out)...")
    x, y, xm, ym = build_windows(speed, tod, mask, N_HIS, N_PRED)
    num = x.shape[0]
    print(f"      -> {num:,} samples; x{x.shape} y{y.shape}")

    print("[4/5] chronological split (not shuffled)...")
    n_test = round(num * args.test_rate)
    n_val = round(num * args.val_rate)
    n_train = num - n_test - n_val
    bounds = {"train": (0, n_train),
              "val": (n_train, n_train + n_val),
              "test": (n_train + n_val, num)}
    for k, (a, b) in bounds.items():
        print(f"      -> {k:<5} {b - a:>7,} samples ({(b - a) / num:.0%})")

    print("[5/5] writing...")
    out_dir = os.path.join(args.stgat_dir, "data", DATASET_NAME)
    os.makedirs(out_dir, exist_ok=True)
    for k, (a, b) in bounds.items():
        p = os.path.join(out_dir, f"{k}.npz")
        # STGAT's util.load_dataset only reads 'x' and 'y'; the extra mask arrays are
        # ignored there but are what makes a masked evaluation possible later.
        np.savez_compressed(p, x=x[a:b], y=y[a:b], x_mask=xm[a:b], y_mask=ym[a:b])
        print(f"OK  {p}  x{x[a:b].shape}")

    # STGAT's util.load_adj expects a (sensor_ids, sensor_id_to_ind, adj_mx) triple.
    sensor_ids = [str(s) for s in sec["SectionID"]]
    p_adj = os.path.join(out_dir, "adj_mx_dijsk.pkl")
    with open(p_adj, "wb") as f:
        pickle.dump((sensor_ids, {s: i for i, s in enumerate(sensor_ids)},
                     adj.astype(np.float32)), f)
    print(f"OK  {p_adj}  {adj.shape}, {float((adj > 0).mean()):.1%} non-zero")

    print(f"\n{num:,} samples x {N_HIS}->{N_PRED} steps x {N} sections x 2 features "
          f"(speed, time-of-day)")
    if not args.missing_as_zero:
        print(f"NOTE  the data still contains {1 - mask.mean():.1%} imputed cells. STGAT's "
              f"built-in masked_mae(null_val=0.0) will NOT catch them\n"
              f"      (imputed values are not 0) -- score with the y_mask stored in the npz.")
    print(f"\nNext:")
    print(f"   cd {args.stgat_dir}")
    print(f"   python train.py --cuda --data data/{DATASET_NAME}/ \\")
    print(f"       --adj_filename data/{DATASET_NAME}/adj_mx_dijsk.pkl \\")
    # --params_dir is not optional in practice: train.py defaults it to
    # 'experiment_METR_LA', so a Taichung run launched without it overwrites the
    # METR-LA checkpoint, and the two have different shapes -- it cannot be recovered.
    print(f"       --num_of_vertices {N} --params_dir experiment_taichung \\")
    print(f"       --lr 3e-4 --epoch 500 --early_stop_maxtry 40")


if __name__ == "__main__":
    main()
