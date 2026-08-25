# -*- coding: utf-8 -*-
"""
evaluate_masked.py
──────────────────────────
Score a trained STGCN on REAL OBSERVATIONS ONLY, printing the unmasked numbers
side by side so the gap between them is visible.

Why this exists:
    After resampling, ~24% of the Taichung TDX cells are imputed (ffill/bfill/column
    mean), against only 7.13% missing in METR-LA. Scoring on imputed cells is nearly
    free for the model -- an ffill value simply equals the previous step -- so an
    unmasked MAE looks far better than the model deserves and cannot be compared with
    METR-LA or the published baselines.

    This failure is SILENT: nothing errors, nothing crashes, the score just looks
    good. STGAT's MAE plateau at 8.5 was the same class of bug (an ineffective mask)
    and took two rounds to find.

    `build_speed.py` saves the mask BEFORE imputation; this script uses it to drop
    the imputed cells from the metrics.

Usage:
    cd STGCN
    python evaluate_masked.py --dataset taichung
    python evaluate_masked.py --dataset taichung --n-pred 6     # 30 min
    python evaluate_masked.py --dataset taichung --n-pred 12    # 60 min

    Note: STGCN emits a single step, so each horizon needs its own trained model.
    The proposal's 15/30/60 min correspond to --n-pred 3/6/12.
"""

import argparse
import math
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

from model import models
from script import dataloader, utility


def build_args(dataset, n_pred, n_his, device):
    """Rebuild the exact model config main.py trained with.

    Any mismatch either fails to load the checkpoint or silently produces wrong
    numbers, so these values must track main.py's defaults.
    """
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

    adj, n_vertex_declared = dataloader.load_adj(dataset)
    # Trust the adjacency's real shape: dataloader hard-codes n_vertex per dataset,
    # and build_speed.py drops high-missing sections, so the two drift apart.
    n_vertex = adj.shape[0]
    if n_vertex != n_vertex_declared:
        print(f"WARNING  dataloader.load_adj declares n_vertex={n_vertex_declared} but the "
              f"adjacency is {n_vertex}. Using {n_vertex} -- update the '{dataset}' branch "
              f"in script/dataloader.py to match, or training will shape-mismatch.")

    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == "cheb_graph_conv":
        gso = utility.calc_chebynet_gso(gso)
    args.gso = torch.from_numpy(gso.toarray().astype(np.float32)).to(device)
    return args, blocks, n_vertex


def split_sizes(n_rows, val_and_test_rate=0.15):
    """Mirror main.py's split exactly -- otherwise the mask rows would not line up
    with the samples the model was tested on."""
    len_val = int(math.floor(n_rows * val_and_test_rate))
    len_test = int(math.floor(n_rows * val_and_test_rate))
    len_train = int(n_rows - len_val - len_test)
    return len_train, len_val, len_test


# MAPE denominator floor, in speed units. A `!= 0` test is NOT enough: dataloader's
# data_transform casts to float32, so a true 0 km/h comes back from the z-score
# round-trip as ~1.3e-6 rather than exactly 0. Dividing by that yields ~1e6 from a
# single cell and swamps the mean (observed: MAPE 6356%). Below ~1 km/h the road is
# standstill anyway and a percentage error is meaningless, so floor the denominator.
MAPE_MIN_SPEED = 1.0


def metrics(y_true, y_pred, valid, mape_floor=MAPE_MIN_SPEED):
    """MAE / RMSE / MAPE / WMAPE. `valid=None` means no masking.

    MAPE drops cells whose true speed is below `mape_floor` (see above); WMAPE
    normalises by the total instead, so it needs no floor and corroborates MAPE.
    """
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="taichung")
    ap.add_argument("--n-pred", type=int, default=3,
                    help="which step ahead (3/6/12 = 15/30/60 min; must match training)")
    ap.add_argument("--n-his", type=int, default=12)
    ap.add_argument("--checkpoint", default=None, help="defaults to STGCN_<dataset>.pt")
    ap.add_argument("--mask", default=None,
                    help="defaults to data/<dataset>/mask.npy, then ../Map/<dataset>_mask.npy")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    cli = ap.parse_args()

    dataset = cli.dataset
    ckpt = cli.checkpoint or f"STGCN_{dataset}.pt"
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt} "
                                f"(train first: python main.py --dataset {dataset})")

    # STGCN's architecture does not depend on n_pred -- Ko is derived from n_his alone,
    # and n_pred only picks WHICH future step data_transform uses as the target. So a
    # 30-minute checkpoint loads into a 15-minute evaluation without any shape error
    # and is quietly scored against the wrong horizon. The giveaway is that the
    # persistence baseline comes out identical across horizons. Since the checkpoints
    # are named STGCN_<dataset>_p<N>.pt, the intended horizon is right there.
    m = re.search(r"_p(\d+)\.pt$", os.path.basename(ckpt))
    if m and int(m.group(1)) != cli.n_pred:
        raise SystemExit(
            f"ERROR  {os.path.basename(ckpt)} was trained for n_pred={m.group(1)} "
            f"({int(m.group(1)) * 5} min) but --n-pred is {cli.n_pred} "
            f"({cli.n_pred * 5} min).\n"
            f"  The model would load fine and be scored against the wrong horizon.\n"
            f"  Run:  python evaluate_masked.py --dataset {dataset} "
            f"--checkpoint {os.path.basename(ckpt)} --n-pred {m.group(1)}")

    # --- data: load and split exactly as main.py does ---
    vel = pd.read_csv(os.path.join("./data", dataset, "vel.csv"))
    n_rows = vel.shape[0]
    len_train, len_val, _ = split_sizes(n_rows)

    train, val, test = dataloader.load_data(dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)                 # scaler fits on train only
    target = zscore.transform(val if cli.split == "val" else test)
    offset = len_train if cli.split == "val" else len_train + len_val

    args, blocks, n_vertex = build_args(dataset, cli.n_pred, cli.n_his, cli.device)
    x, y = dataloader.data_transform(target, args.n_his, args.n_pred, cli.device)
    n_sample = x.shape[0]

    # data_transform's sample i targets row (i + n_his + n_pred - 1) of the SPLIT;
    # adding the split's offset maps it back to a row of vel.csv / mask.
    rows = offset + np.arange(n_sample) + args.n_his + args.n_pred - 1

    # Persistence: predict the target from the last observed frame, i.e. the row
    # n_pred steps earlier. It costs nothing to compute and is the single most
    # useful sanity check -- a model that cannot beat it has not learned the
    # dynamics, only the mean (this is exactly how STGAT's broken run was caught).
    y_pers = vel.to_numpy(dtype=float)[rows - args.n_pred]

    # --- model ---
    model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(cli.device)
    model.load_state_dict(torch.load(ckpt, map_location=cli.device))
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, n_sample, cli.batch_size):
            preds.append(model(x[i:i + cli.batch_size]).view(-1, n_vertex).cpu().numpy())
    y_pred = zscore.inverse_transform(np.concatenate(preds))
    y_true = zscore.inverse_transform(y.cpu().numpy())

    # --- mask ---
    mask_path = cli.mask
    if mask_path is None:
        for cand in (os.path.join("./data", dataset, "mask.npy"),
                     os.path.join("..", "Map", f"{dataset}_mask.npy")):
            if os.path.isfile(cand):
                mask_path = cand
                break

    valid = None
    if mask_path and os.path.isfile(mask_path):
        mask = np.load(mask_path)
        if mask.shape != (n_rows, n_vertex):
            raise ValueError(f"mask shape {mask.shape} != data ({n_rows}, {n_vertex}). "
                             f"mask.npy and vel.csv must come from the same build_speed.py run.")
        valid = mask[rows]
    else:
        print("WARNING  no mask.npy found -- only unmasked numbers below. "
              "Do NOT put these in the report.\n")

    # --- report ---
    print(f"\n=== setup ===")
    print(f"  dataset={dataset} | n_vertex={n_vertex} | checkpoint={ckpt}")
    print(f"  n_his={args.n_his}, n_pred={args.n_pred} ({args.n_pred * 5} min ahead) | "
          f"split={cli.split}")
    print(f"  {n_sample:,} samples | source rows {rows[0]:,}..{rows[-1]:,}")

    raw = metrics(y_true, y_pred, None)
    if valid is None:
        pers = metrics(y_true, y_pers, None)
        print(f"\n  model       MAE {raw['MAE']:.4f} | RMSE {raw['RMSE']:.4f} | "
              f"WMAPE {raw['WMAPE']:.2%}")
        print(f"  persistence MAE {pers['MAE']:.4f} | RMSE {pers['RMSE']:.4f} | "
              f"WMAPE {pers['WMAPE']:.2%}")
        return

    m = metrics(y_true, y_pred, valid)
    p = metrics(y_true, y_pers, valid)
    ratio = valid.mean()
    print(f"\n=== mask ===")
    print(f"  target cells      {valid.size:,}")
    print(f"  real observations {int(valid.sum()):,} ({ratio:.1%})")
    print(f"  imputed (dropped) {int((~valid).sum()):,} ({1 - ratio:.1%})")

    print(f"\n=== metrics ===")
    print(f"  {'':<8}{'unmasked':>12}{'real only':>14}{'delta':>10}"
          f"{'persistence':>14}{'vs pers.':>11}")
    print("  " + "-" * 69)
    for key, fmt in (("MAE", "{:.4f}"), ("RMSE", "{:.4f}"),
                     ("MAPE", "{:.2%}"), ("WMAPE", "{:.2%}")):
        a, b, c = raw[key], m[key], p[key]
        delta = (b - a) / a * 100 if a else float("nan")
        gain = (b - c) / c * 100 if c else float("nan")
        print(f"  {key:<8}{fmt.format(a):>12}{fmt.format(b):>14}{delta:>+9.1f}%"
              f"{fmt.format(c):>14}{gain:>+10.1f}%")

    print(f"\n  MAPE drops cells with true speed < {MAPE_MIN_SPEED} "
          f"({raw['mape_excluded']:,} unmasked / {m['mape_excluded']:,} masked); "
          f"WMAPE needs no floor and corroborates it.")

    # The verdict that matters: a model that cannot beat "copy the last frame" has
    # learned the mean, not the dynamics -- the failure mode that hid in STGAT for
    # two rounds of debugging.
    beat = m["MAE"] < p["MAE"]
    print(f"\n  {'PASS' if beat else 'FAIL'}: masked MAE {m['MAE']:.4f} "
          f"{'<' if beat else '>='} persistence {p['MAE']:.4f}"
          f"  ({abs(m['MAE'] - p['MAE']) / p['MAE'] * 100:.1f}% "
          f"{'better' if beat else 'WORSE'})")
    if not beat:
        print("  -> Do not tune hyper-parameters yet. Check data alignment, masking, "
              "and the training loop first.")

    print(f"\nReport the 'real only' column. The unmasked figures include "
          f"{1 - ratio:.1%} imputed cells\n(mostly ffill, i.e. equal to the previous "
          f"step), which the model predicts at almost no cost -- they flatter the score\n"
          f"and make it incomparable with METR-LA (7.13% missing).")


if __name__ == "__main__":
    main()
