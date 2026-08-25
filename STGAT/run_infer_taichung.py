# -*- coding: utf-8 -*-
"""
run_infer_taichung.py
──────────────────────────
Run the trained Taichung STGAT and save its per-section speed prediction.

Counterpart to STGCN/run_infer_taichung.py: writes a raw, unclamped .npy into
integration/, where make_drl_input.py ensembles the two models, clamps, and maps the
section speeds onto OSM edges.

Horizon indexing (easy to get wrong):
    STGAT emits 12 steps at once, y = t+1 .. t+12. So "n steps ahead" is index n-1:
        --n-pred 3  -> index 2  -> 15 min      (matches STGCN --n_pred 3)
        --n-pred 6  -> index 5  -> 30 min
        --n-pred 12 -> index 11 -> 60 min
    The existing METR-LA run_infer.py takes index 0, i.e. 5 minutes, while its STGCN
    counterpart uses n_pred=3, i.e. 15 minutes -- so that ensemble mixes two horizons.
    This script keeps --n-pred consistent with STGCN so the Taichung ensemble does not.

Aligning with STGCN (read this before ensembling):
    `--index -1` means "the last test window", and the two models build windows
    differently, so it does NOT mean the same moment:

        STGAT  TDX_Data/convert_to_stgat_dataset.build_windows: y[i] spans
               feat[i+n_his : i+n_his+12], horizon p is index p-1  -> last target = T-13+p
        STGCN  script/dataloader.data_transform: num = len - n_his - n_pred  (no +1)
               and y[i] = test[i + n_his + n_pred - 1]             -> last target = T-2

    At the 15 min horizon that is an 8-step (40 min) gap. Averaging two snapshots taken
    40 minutes apart yields a perfectly plausible -- and wrong -- edge CSV, with nothing
    raising an error. Use --target-row instead of --index whenever the output is going
    to be ensembled; both scripts print the row they predicted, and make_drl_input.py
    refuses to combine two .npy files whose target rows disagree.

Output:
    integration/taichung_pred_stgat.npy         [n_sections] km/h, raw model output
    integration/taichung_pred_stgat.meta.json   which row/time this snapshot is

Usage:
    cd STGAT
    python run_infer_taichung.py                                  # prints its target row
    python run_infer_taichung.py --n-pred 12 --checkpoint experiment_taichung/best_model.pth

    # aligned pair: let this script pick the row, then match it in STGCN
    python run_infer_taichung.py --n-pred 3                        # prints target row R
    cd ../STGCN && python run_infer_taichung.py --n-pred 3 --target-row R \
                       --checkpoint STGCN_taichung_p3.pt

    Note: train.py now honours --params_dir when saving, but that flag still DEFAULTS to
    'experiment_METR_LA', so a Taichung run launched without it silently overwrites the
    METR-LA checkpoint. Always pass --params_dir experiment_taichung when training.
"""

import argparse
import contextlib
import io
import json
import os

import numpy as np
import pandas as pd
import torch

import util
from model.stgat import STGAT

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
MAP_DIR = os.path.join(ROOT, "Map")
OUT_DIR = os.path.join(ROOT, "integration")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/taichung/")
    ap.add_argument("--adj", default=None, help="defaults to <data>/adj_mx_dijsk.pkl")
    ap.add_argument("--checkpoint", default="experiment_taichung/best_model.pth")
    ap.add_argument("--n-pred", type=int, default=3,
                    help="steps ahead (3/6/12 = 15/30/60 min); one STGAT model covers all")
    ap.add_argument("--index", type=int, default=None,
                    help="which test window to predict from; -1 = most recent (default)")
    ap.add_argument("--target-row", type=int, default=None,
                    help="predict this absolute row of Map/taichung_timestamps.csv. "
                         "Mutually exclusive with --index. This is the shared coordinate "
                         "between the two models -- use it when the output will be "
                         "ensembled, since --index -1 means a different moment in each.")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="which chronological split to draw windows from")
    ap.add_argument("--dump-all", action="store_true",
                    help="predict EVERY window of --split and write an npz of "
                         "{pred, rows} instead of one .npy. Feeds "
                         "integration/search_ensemble_weight.py.")
    ap.add_argument("--batch-size", type=int, default=32, help="only used by --dump-all")
    ap.add_argument("--nhid", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--nheads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = ap.parse_args()

    if cli.index is not None and cli.target_row is not None:
        ap.error("pass either --index or --target-row, not both")
    if cli.dump_all and (cli.index is not None or cli.target_row is not None):
        ap.error("--dump-all predicts every window, so --index/--target-row mean nothing")

    adj_path = cli.adj or os.path.join(cli.data, "adj_mx_dijsk.pkl")
    if not os.path.isfile(cli.checkpoint):
        raise FileNotFoundError(
            f"checkpoint not found: {cli.checkpoint}\n"
            f"Train first with --params_dir experiment_taichung; without that flag "
            f"train.py saves into experiment_METR_LA/ and clobbers the METR-LA model.")
    if not os.path.isfile(adj_path):
        raise FileNotFoundError(f"adjacency not found: {adj_path} "
                                f"(run TDX_Data/convert_to_stgat_dataset.py first)")

    # --- data: load_dataset normalises x/y in place and hands back the fitted scaler ---
    data = util.load_dataset(cli.data, 1, 1, 1)
    scaler = data["scaler"]
    x_split = data[f"x_{cli.split}"]              # (samples, T, N, F), already z-scored
    n_sample, n_his, n_vertex, n_feat = x_split.shape

    if not 1 <= cli.n_pred <= 12:
        raise ValueError(f"--n-pred must be 1..12 (STGAT emits 12 steps), got {cli.n_pred}")

    # Absolute row of the full timeline (== row of Map/taichung_timestamps.csv) that
    # window i of this split predicts at this horizon. build_windows() sets
    # y[g] = feat[g+n_his : g+n_his+12] for the GLOBAL window g, so horizon p (1-based)
    # of window g lands on row g + n_his + p - 1; the splits are chronological, so where
    # this one starts is a plain offset in window counts.
    n_train, n_val = data["x_train"].shape[0], data["x_val"].shape[0]
    split_start = {"val": n_train, "test": n_train + n_val}[cli.split]
    row0 = split_start + n_his + cli.n_pred - 1
    row_last = row0 + n_sample - 1
    _, _, adj_list = util.load_adj(adj_path, "symnadj")
    adj_mx = torch.from_numpy(np.array(adj_list, dtype=np.float32))[0].to(cli.device)
    if adj_mx.shape[0] != n_vertex:
        raise ValueError(f"adjacency is {tuple(adj_mx.shape)} but the data has "
                         f"{n_vertex} nodes -- both must come from the same conversion run")

    # --- model ---
    net = STGAT(cli.device.startswith("cuda"), n_vertex, n_feat, n_his, 12,
                nheads=cli.nheads, nhid=cli.nhid, layers=cli.layers).to(cli.device)
    net.load_state_dict(torch.load(cli.checkpoint, map_location=cli.device))
    net.eval()

    # --- bulk mode: every window of the split, keyed by absolute row ---
    if cli.dump_all:
        preds = []
        with torch.no_grad():
            for i in range(0, n_sample, cli.batch_size):
                xb = torch.Tensor(
                    x_split[i:i + cli.batch_size].transpose(0, 2, 1, 3)).to(cli.device)
                # model/stgat.py still has a debug print of every layer's shape; without
                # this the dump loop buries the report under tens of thousands of lines.
                with contextlib.redirect_stdout(io.StringIO()):
                    out = net(adj_mx, xb)
                if isinstance(out, tuple):
                    out = out[0]
                preds.append(out[:, :, cli.n_pred - 1, 0].cpu().numpy())
        speeds = scaler.inverse_transform(np.concatenate(preds))
        rows = row0 + np.arange(n_sample)

        os.makedirs(OUT_DIR, exist_ok=True)
        dump = os.path.join(OUT_DIR, f"dump_stgat_{cli.split}_p{cli.n_pred}.npz")
        np.savez(dump, pred=speeds.astype(np.float32), rows=rows.astype(np.int64),
                 n_pred=cli.n_pred, split=cli.split)
        print("=== STGAT bulk inference ===")
        print(f"  checkpoint {cli.checkpoint} | n_vertex {n_vertex} | "
              f"{cli.n_pred * 5} min ahead (horizon index {cli.n_pred - 1})")
        print(f"  split {cli.split} | {n_sample} windows | rows {row0}..{row_last}")
        print(f"  predicted speed {speeds.min():.1f}..{speeds.max():.1f} km/h (raw)")
        print(f"\nOK  {dump}")
        print("\nNext: the same for STGCN, then")
        print(f"  cd ../integration && python search_ensemble_weight.py "
              f"--n-pred {cli.n_pred} --split {cli.split}")
        return

    if cli.target_row is not None:
        idx = cli.target_row - row0
        if not 0 <= idx < n_sample:
            raise IndexError(
                f"--target-row {cli.target_row} is out of this model's reach: at the "
                f"{cli.n_pred * 5} min horizon it can only predict rows {row0}..{row_last}. "
                f"A row past {row_last} needs more history than the data has; a row below "
                f"{row0} falls inside the train/val split. Note the reachable range moves "
                f"with --n-pred, so a row STGCN can hit at 15 min may need a different "
                f"horizon here.")
    else:
        want = -1 if cli.index is None else cli.index
        idx = want if want >= 0 else n_sample + want
        if not 0 <= idx < n_sample:
            raise IndexError(f"--index {want} out of range (0..{n_sample - 1})")
    target_row = row0 + idx

    # NetDataSet feeds the model (N, T, F), so transpose the stored (T, N, F) sample.
    x = torch.Tensor(x_split[idx].transpose(1, 0, 2)).unsqueeze(0).to(cli.device)
    with torch.no_grad():
        out = net(adj_mx, x)
        if isinstance(out, tuple):                # train() mode also returns logits
            out = out[0]
    # out: (1, N, 12, 1) -> pick the requested horizon
    speed = scaler.inverse_transform(out[0, :, cli.n_pred - 1, 0].cpu().numpy())

    # Which moment this snapshot represents. Timestamps are UTC; Taiwan is UTC+8.
    ts_path = os.path.join(MAP_DIR, "taichung_timestamps.csv")
    when = "unknown"
    if os.path.isfile(ts_path):
        ts = pd.read_csv(ts_path, encoding="utf-8-sig")
        if target_row < len(ts):
            when = str(ts.iloc[target_row, 0])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "taichung_pred_stgat.npy")
    np.save(out_path, speed)

    # Sidecar so make_drl_input.py can verify the two models predicted the same moment.
    # Without it the alignment is unverifiable after the fact -- the .npy is a bare
    # float array with no record of when it is supposed to be.
    meta = {"model": "stgat", "checkpoint": cli.checkpoint, "n_pred": cli.n_pred,
            "minutes_ahead": cli.n_pred * 5, "split": cli.split,
            "window_index": int(idx), "target_row": int(target_row),
            "target_time": when, "n_sections": int(speed.shape[0])}
    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=== STGAT inference ===")
    print(f"  checkpoint {cli.checkpoint} | n_vertex {n_vertex} | "
          f"{cli.n_pred * 5} min ahead (horizon index {cli.n_pred - 1})")
    print(f"  {cli.split} window {idx} of {n_sample - 1} "
          f"(reachable rows {row0}..{row_last})")
    print(f"  target row  {target_row} | target time {when} (UTC, +8 = Taiwan)")
    print(f"  predicted speed {speed.min():.1f}..{speed.max():.1f} km/h (raw, unclamped)")
    print(f"\nOK  {out_path}  [{speed.shape[0]}]")
    print(f"OK  {meta_path}")
    print("\nNext, to predict the SAME moment with STGCN:")
    print(f"  cd ../STGCN && python run_infer_taichung.py --n-pred {cli.n_pred} "
          f"--target-row {target_row} --checkpoint STGCN_taichung_p{cli.n_pred}.pt")
    print("  cd ../integration && python make_drl_input.py")


if __name__ == "__main__":
    main()
