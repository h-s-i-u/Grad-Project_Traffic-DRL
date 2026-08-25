# -*- coding: utf-8 -*-
"""
run_infer_taichung.py
──────────────────────────
Run the trained Taichung STGCN and save its per-section speed prediction.

Responsibilities are split the same way as the METR-LA pipeline: each model writes a
raw, unclamped .npy into integration/, and integration/make_drl_input.py does the
ensembling, clamping and section->edge mapping. Keeping the clamp downstream means an
out-of-range prediction stays visible in this file instead of being silently hidden
(that is how STGAT's 244 mph bug surfaced).

Aligning with STGAT (read this before ensembling):
    `--index -1` means "the last test window", and the two models build windows
    differently, so it does NOT mean the same moment:

        STGCN  script/dataloader.data_transform: num = len - n_his - n_pred  (no +1)
               and y[i] = test[i + n_his + n_pred - 1]   -> last target = T-2
        STGAT  TDX_Data/convert_to_stgat_dataset.build_windows: y[i] spans
               feat[i+n_his : i+n_his+12], horizon p is index p-1  -> last target = T-13+p

    At the 15 min horizon that is an 8-step (40 min) gap. Averaging two snapshots taken
    40 minutes apart yields a perfectly plausible -- and wrong -- edge CSV, with nothing
    raising an error. Use --target-row instead of --index whenever the output is going
    to be ensembled; both scripts print the row they predicted, and make_drl_input.py
    refuses to combine two .npy files whose target rows disagree.

Output:
    integration/taichung_pred_stgcn.npy         [n_sections] km/h, raw model output
    integration/taichung_pred_stgcn.meta.json   which row/time this snapshot is

Usage:
    cd STGCN
    python run_infer_taichung.py --checkpoint STGCN_taichung_p3.pt          # latest window
    python run_infer_taichung.py --n-pred 6 --checkpoint STGCN_taichung_p6.pt

    # aligned pair: let STGAT pick the row, then match it here
    cd ../STGAT && python run_infer_taichung.py --n-pred 3         # prints target row R
    cd ../STGCN && python run_infer_taichung.py --n-pred 3 --target-row R \
                       --checkpoint STGCN_taichung_p3.pt

    Note: main.py saves to STGCN_<dataset>.pt regardless of --n_pred, so rename the
    checkpoint between horizons or the next training run overwrites it.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

from model import models
from script import dataloader, utility

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
MAP_DIR = os.path.join(ROOT, "Map")
OUT_DIR = os.path.join(ROOT, "integration")


def build_model_args(dataset, n_pred, n_his, device):
    """Rebuild the exact config main.py trained with; a mismatch fails to load."""
    args = argparse.Namespace(
        dataset=dataset, n_his=n_his, n_pred=n_pred,
        Kt=3, Ks=3, stblock_num=2,
        act_func="glu", graph_conv_type="cheb_graph_conv",
        gso_type="sym_norm_lap", enable_bias=True, droprate=0.5,
    )
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    blocks = [[1]]
    for _ in range(args.stblock_num):
        blocks.append([64, 16, 64])
    blocks.append([128] if Ko == 0 else [128, 128])
    blocks.append([1])

    adj, n_declared = dataloader.load_adj(dataset)
    n_vertex = adj.shape[0]            # trust the matrix, not the hard-coded table
    if n_vertex != n_declared:
        print(f"WARNING  dataloader declares n_vertex={n_declared} but the adjacency is "
              f"{n_vertex}; using {n_vertex}. Update script/dataloader.py to match.")

    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == "cheb_graph_conv":
        gso = utility.calc_chebynet_gso(gso)
    args.gso = torch.from_numpy(gso.toarray().astype(np.float32)).to(device)
    return args, blocks, n_vertex


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="taichung")
    ap.add_argument("--n-pred", type=int, default=3,
                    help="steps ahead (3/6/12 = 15/30/60 min; must match the checkpoint)")
    ap.add_argument("--n-his", type=int, default=12)
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
    ap.add_argument("--batch-size", type=int, default=64, help="only used by --dump-all")
    ap.add_argument("--checkpoint", default=None, help="defaults to STGCN_<dataset>.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = ap.parse_args()

    if cli.index is not None and cli.target_row is not None:
        ap.error("pass either --index or --target-row, not both")
    if cli.dump_all and (cli.index is not None or cli.target_row is not None):
        ap.error("--dump-all predicts every window, so --index/--target-row mean nothing")

    ckpt = cli.checkpoint or f"STGCN_{cli.dataset}.pt"
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt} "
                                f"(train first: python main.py --dataset {cli.dataset})")

    # --- data: same load/split/scaler as main.py, so the model sees what it expects ---
    vel = pd.read_csv(os.path.join("./data", cli.dataset, "vel.csv"))
    n_rows = vel.shape[0]
    len_val = int(math.floor(n_rows * 0.15))
    len_test = int(math.floor(n_rows * 0.15))
    len_train = int(n_rows - len_val - len_test)

    train, val, test = dataloader.load_data(cli.dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)          # scaler must come from train only
    chosen = zscore.transform({"val": val, "test": test}[cli.split])

    args, blocks, n_vertex = build_model_args(cli.dataset, cli.n_pred, cli.n_his, cli.device)
    x, _ = dataloader.data_transform(chosen, args.n_his, args.n_pred, cli.device)

    # Absolute row of the full timeline (== row of Map/taichung_timestamps.csv) that
    # window i of this split predicts. data_transform sets
    # y[i] = split[i + n_his + n_pred - 1], and load_data slices chronologically, so
    # where the split starts is a plain row offset.
    split_start = {"val": len_train, "test": len_train + len_val}[cli.split]
    row0 = split_start + args.n_his + args.n_pred - 1
    row_last = row0 + x.shape[0] - 1

    model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(cli.device)
    model.load_state_dict(torch.load(ckpt, map_location=cli.device))
    model.eval()

    # --- bulk mode: every window of the split, keyed by absolute row ---
    if cli.dump_all:
        preds = []
        with torch.no_grad():
            for i in range(0, x.shape[0], cli.batch_size):
                xb = x[i:i + cli.batch_size]
                preds.append(model(xb).view(xb.shape[0], -1).cpu().numpy())
        speeds = zscore.inverse_transform(np.concatenate(preds))
        rows = row0 + np.arange(x.shape[0])

        os.makedirs(OUT_DIR, exist_ok=True)
        dump = os.path.join(OUT_DIR, f"dump_stgcn_{cli.split}_p{args.n_pred}.npz")
        np.savez(dump, pred=speeds.astype(np.float32), rows=rows.astype(np.int64),
                 n_pred=args.n_pred, split=cli.split)
        print("=== STGCN bulk inference ===")
        print(f"  checkpoint {ckpt} | n_vertex {n_vertex} | {args.n_pred * 5} min ahead")
        print(f"  split {cli.split} | {x.shape[0]} windows | rows {row0}..{row_last}")
        print(f"  predicted speed {speeds.min():.1f}..{speeds.max():.1f} km/h (raw)")
        print(f"\nOK  {dump}")
        print("\nNext: the same for STGAT, then")
        print(f"  cd ../integration && python search_ensemble_weight.py "
              f"--n-pred {args.n_pred} --split {cli.split}")
        return

    if cli.target_row is not None:
        idx = cli.target_row - row0
        if not 0 <= idx < x.shape[0]:
            raise IndexError(
                f"--target-row {cli.target_row} is out of this model's reach: with "
                f"n_his={args.n_his} and n_pred={args.n_pred} it can only predict rows "
                f"{row0}..{row_last}. A row past {row_last} needs more history than the "
                f"data has; a row below {row0} falls inside the train/val split.")
    else:
        want = -1 if cli.index is None else cli.index
        idx = want if want >= 0 else x.shape[0] + want
        if not 0 <= idx < x.shape[0]:
            raise IndexError(f"--index {want} out of range (0..{x.shape[0] - 1})")
    target_row = row0 + idx

    # --- predict ---
    with torch.no_grad():
        pred = model(x[idx:idx + 1]).view(1, -1).cpu().numpy()
    speed = zscore.inverse_transform(pred)[0]

    # Which moment this snapshot represents. Timestamps are UTC; Taiwan is UTC+8.
    ts_path = os.path.join(MAP_DIR, "taichung_timestamps.csv")
    when = "unknown"
    if os.path.isfile(ts_path):
        ts = pd.read_csv(ts_path, encoding="utf-8-sig")
        if target_row < len(ts):
            when = str(ts.iloc[target_row, 0])

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "taichung_pred_stgcn.npy")
    np.save(out, speed)

    # Sidecar so make_drl_input.py can verify the two models predicted the same moment.
    # Without it the alignment is unverifiable after the fact -- the .npy is a bare
    # float array with no record of when it is supposed to be.
    meta = {"model": "stgcn", "checkpoint": ckpt, "n_pred": args.n_pred,
            "minutes_ahead": args.n_pred * 5, "split": cli.split,
            "window_index": int(idx), "target_row": int(target_row),
            "target_time": when, "n_sections": int(speed.shape[0])}
    meta_path = os.path.splitext(out)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=== STGCN inference ===")
    print(f"  checkpoint {ckpt} | n_vertex {n_vertex} | {args.n_pred * 5} min ahead")
    print(f"  {cli.split} window {idx} of {x.shape[0] - 1} "
          f"(reachable rows {row0}..{row_last})")
    print(f"  target row  {target_row} | target time {when} (UTC, +8 = Taiwan)")
    print(f"  predicted speed {speed.min():.1f}..{speed.max():.1f} km/h (raw, unclamped)")
    print(f"\nOK  {out}  [{speed.shape[0]}]")
    print(f"OK  {meta_path}")
    print("\nNext, to predict the SAME moment with STGAT:")
    print(f"  cd ../STGAT && python run_infer_taichung.py "
          f"--n-pred {args.n_pred} --target-row {target_row}")
    print("  cd ../integration && python make_drl_input.py")


if __name__ == "__main__":
    main()
