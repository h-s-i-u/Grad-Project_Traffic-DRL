# -*- coding: utf-8 -*-
"""
evaluate_masked_taichung.py
──────────────────────────
Score the trained Taichung STGAT on REAL OBSERVATIONS ONLY, alongside the unmasked
numbers and a persistence baseline.

Why train.py's own numbers are not enough:
    train.py scores with util.masked_mae(pred, real, 0.0), which drops cells whose
    value is 0. That is the METR-LA convention, where 0 marks a missing reading. The
    Taichung data was IMPUTED by build_speed.py, so its missing cells hold ffill'd
    values, not 0 -- the built-in mask sees nothing to drop and all ~24% of imputed
    cells count towards the score. An ffill'd value equals the previous step, so the
    model predicts it at almost no cost and the MAE comes out flattered.

    convert_to_stgat_dataset.py therefore stores `y_mask` inside the npz (captured
    before imputation). This script uses it.

    The failure is silent -- nothing errors, the score just looks good. That is how
    STGAT's first mask bug hid for two rounds of debugging.

Usage:
    cd STGAT
    python evaluate_masked_taichung.py
    python evaluate_masked_taichung.py --checkpoint experiment_taichung/best_model.pth
"""

import argparse
import contextlib
import io
import os

import numpy as np
import torch

import util
from model.stgat import STGAT

# MAPE denominator floor, km/h. A `!= 0` test is not enough: the float32 round-trip
# through the scaler turns a true 0 into ~1e-6, and dividing by that yields ~1e6 from a
# single cell (this exact bug produced a 6356% MAPE in the STGCN evaluator). Below
# ~1 km/h the road is standstill and a percentage error is meaningless anyway.
MAPE_MIN_SPEED = 1.0

# Horizons highlighted in the detail table: the proposal reports 15/30/60 min.
KEY_HORIZONS = (3, 6, 12)


def metrics(y_true, y_pred, valid, floor=MAPE_MIN_SPEED):
    """MAE / RMSE / MAPE / WMAPE. `valid=None` means no masking."""
    if valid is not None:
        y_true, y_pred = y_true[valid], y_pred[valid]
    if y_true.size == 0:
        return {k: float("nan") for k in ("MAE", "RMSE", "MAPE", "WMAPE")}
    d = np.abs(y_true - y_pred)
    ok = y_true >= floor
    return {
        "MAE": float(d.mean()),
        "RMSE": float(np.sqrt((d ** 2).mean())),
        "MAPE": float((d[ok] / y_true[ok]).mean()) if ok.any() else float("nan"),
        "WMAPE": float(d.sum() / y_true.sum()) if y_true.sum() else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/taichung/")
    ap.add_argument("--adj", default=None, help="defaults to <data>/adj_mx_dijsk.pkl")
    ap.add_argument("--checkpoint", default="experiment_taichung/best_model.pth")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--nhid", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--nheads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = ap.parse_args()

    adj_path = cli.adj or os.path.join(cli.data, "adj_mx_dijsk.pkl")
    npz_path = os.path.join(cli.data, f"{cli.split}.npz")
    for p, name in ((cli.checkpoint, "checkpoint"), (adj_path, "adjacency"),
                    (npz_path, f"{cli.split}.npz")):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{name} not found: {p}")

    # load_dataset applies the same scaler training used; the npz is reopened for the
    # mask because load_dataset only reads 'x' and 'y'.
    data = util.load_dataset(cli.data, 1, 1, 1)
    scaler = data["scaler"]
    x = data[f"x_{cli.split}"]                       # (S, T_in, N, F), z-scored
    y = data[f"y_{cli.split}"][..., 0]               # (S, T_out, N), z-scored speed
    with np.load(npz_path) as z:
        if "y_mask" not in z:
            raise KeyError(f"{npz_path} has no 'y_mask' -- regenerate it with "
                           f"TDX_Data/convert_to_stgat_dataset.py")
        y_mask = z["y_mask"]                         # (S, T_out, N) bool
    S, T_in, N, F = x.shape
    T_out = y.shape[1]

    _, _, adj_list = util.load_adj(adj_path, "symnadj")
    adj_mx = torch.from_numpy(np.array(adj_list, dtype=np.float32))[0].to(cli.device)
    if adj_mx.shape[0] != N:
        raise ValueError(f"adjacency is {tuple(adj_mx.shape)} but the data has {N} nodes")

    net = STGAT(cli.device.startswith("cuda"), N, F, T_in, T_out,
                nheads=cli.nheads, nhid=cli.nhid, layers=cli.layers).to(cli.device)
    net.load_state_dict(torch.load(cli.checkpoint, map_location=cli.device))
    net.eval()

    # --- predict every window ---
    preds = []
    with torch.no_grad():
        for i in range(0, S, cli.batch_size):
            # NetDataSet hands the model (N, T, F); the stored layout is (T, N, F).
            xb = torch.Tensor(x[i:i + cli.batch_size].transpose(0, 2, 1, 3)).to(cli.device)
            # model/stgat.py has a leftover debug print of the layer shapes; silence it
            # so it does not bury the report under thousands of lines.
            with contextlib.redirect_stdout(io.StringIO()):
                out = net(adj_mx, xb)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.squeeze(-1).cpu().numpy())      # (B, N, T_out)
    y_pred = scaler.inverse_transform(np.concatenate(preds))  # (S, N, T_out)

    y_true = scaler.inverse_transform(y.transpose(0, 2, 1))   # (S, N, T_out)
    valid = y_mask.transpose(0, 2, 1)                         # (S, N, T_out)

    # Persistence: hold the last observed frame for every horizon. Free to compute and
    # the sharpest check that the model learned dynamics rather than the mean.
    last = scaler.inverse_transform(x[:, -1, :, 0])           # (S, N)
    y_pers = np.repeat(last[:, :, None], T_out, axis=2)

    ratio = valid.mean()
    print(f"\n=== setup ===")
    print(f"  checkpoint {cli.checkpoint} | n_vertex {N} | split {cli.split}")
    print(f"  {S:,} windows x {T_out} horizons = {valid.size:,} target cells")
    print(f"\n=== mask ===")
    print(f"  real observations {int(valid.sum()):,} ({ratio:.1%})")
    print(f"  imputed (dropped) {int((~valid).sum()):,} ({1 - ratio:.1%})")

    # --- per-horizon MAE ---
    print(f"\n=== MAE by horizon ===")
    print(f"  {'h':>3}{'min':>5}{'unmasked':>11}{'real only':>12}{'delta':>9}"
          f"{'persist.':>11}{'vs pers.':>10}")
    print("  " + "-" * 59)
    rows = []
    for h in range(T_out):
        a = metrics(y_true[..., h], y_pred[..., h], None)["MAE"]
        b = metrics(y_true[..., h], y_pred[..., h], valid[..., h])["MAE"]
        c = metrics(y_true[..., h], y_pers[..., h], valid[..., h])["MAE"]
        rows.append((a, b, c))
        mark = " *" if (h + 1) in KEY_HORIZONS else ""
        print(f"  {h + 1:>3}{(h + 1) * 5:>5}{a:>11.4f}{b:>12.4f}"
              f"{(b - a) / a * 100:>+8.1f}%{c:>11.4f}{(b - c) / c * 100:>+9.1f}%{mark}")
    ua, ub, uc = (np.mean([r[i] for r in rows]) for i in range(3))
    print("  " + "-" * 59)
    print(f"  {'avg':>8}{ua:>11.4f}{ub:>12.4f}{(ub - ua) / ua * 100:>+8.1f}%"
          f"{uc:>11.4f}{(ub - uc) / uc * 100:>+9.1f}%")
    print(f"\n  * = the 15/30/60 min horizons the proposal reports")

    # --- full metrics at the reported horizons ---
    print(f"\n=== real observations only, at 15/30/60 min ===")
    print(f"  {'horizon':>8}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}{'WMAPE':>10}")
    print("  " + "-" * 48)
    for h in KEY_HORIZONS:
        if h > T_out:
            continue
        m = metrics(y_true[..., h - 1], y_pred[..., h - 1], valid[..., h - 1])
        print(f"  {h * 5:>5} min{m['MAE']:>10.4f}{m['RMSE']:>10.4f}"
              f"{m['MAPE']:>10.2%}{m['WMAPE']:>10.2%}")

    # --- verdict ---
    beat = ub < uc
    print(f"\n  {'PASS' if beat else 'FAIL'}: masked MAE {ub:.4f} "
          f"{'<' if beat else '>='} persistence {uc:.4f} "
          f"({abs(ub - uc) / uc * 100:.1f}% {'better' if beat else 'WORSE'})")
    if not beat:
        print("  -> Do not tune hyper-parameters yet. Check data alignment, the mask, "
              "and the training loop first.")
    inc = rows[-1][1] > rows[0][1]
    print(f"  {'PASS' if inc else 'FAIL'}: MAE grows with horizon "
          f"({rows[0][1]:.4f} -> {rows[-1][1]:.4f})"
          f"{'' if inc else '  -> a flat curve means the model only learned the mean'}")

    print(f"\nReport the 'real only' column. The unmasked figures include "
          f"{1 - ratio:.1%} imputed cells\n(mostly ffill, i.e. equal to the previous "
          f"step), which the model predicts at almost no cost.")


if __name__ == "__main__":
    main()
