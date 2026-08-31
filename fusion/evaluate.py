#!/usr/bin/env python3
"""Score a trained fusion model, and answer the question it exists to answer.

Two outputs:

  1. Masked MAE / RMSE / MAPE / WMAPE per horizon, beside the numbers the separate
     models, HA and persistence already have. Metric definitions are copied from
     STGCN/evaluate_masked.py, MAPE floor included, so the columns are comparable.

  2. GATE STATISTICS. A gate that never moves means the model settled on a fixed blend,
     and the whole exercise has reproduced the constant weight 實驗記錄 §13.7 already
     measured (best mix beats STGAT alone by 0.40%, errors correlate 0.934). The spread
     of the gate ACROSS SECTIONS and ACROSS TIME is what says whether per-node,
     per-timestep gating found anything a global constant could not.

`--dump-all` writes integration/dump_fusion_<split>_p<N>.npz in the same format the two
run_infer_taichung.py scripts use, so integration/ha_baseline.py picks it up as another
column in the anomaly breakdown and search_ensemble_weight.py can read it too.

Usage:
    cd fusion
    python evaluate.py --checkpoint checkpoints/fusion.pt --split test
    python evaluate.py --checkpoint checkpoints/fusion.pt --split test --dump-all
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths as P
from data import FusionDataset
from model import DualPathModel

MAPE_MIN_SPEED = 1.0        # STGCN/evaluate_masked.py: a true 0 survives the float32
                            # round trip as ~1.3e-6 and one cell pushed MAPE to 6356%

# 實驗記錄 §13.11 / §13.18, masked, test split.
REFERENCE = {
    "STGAT": {3: 3.3802, 6: 3.5127, 12: 3.6276},
    "STGCN": {3: 3.5560, 6: 3.7535, 12: 3.9549},
    "HA": {3: 4.0486, 6: 4.0489, 12: 4.0496},
    "persistence": {3: 4.2872, 6: 4.6744, 12: 5.1281},
}


def metrics(y_true, y_pred, valid, floor=MAPE_MIN_SPEED):
    y_true, y_pred = y_true[valid], y_pred[valid]
    d = np.abs(y_true - y_pred)
    ok = y_true >= floor
    return {"n": int(y_true.size), "MAE": float(d.mean()),
            "RMSE": float(np.sqrt((d ** 2).mean())),
            "MAPE": float((d[ok] / y_true[ok]).mean()) if ok.any() else float("nan"),
            "WMAPE": float(d.sum() / y_true.sum()) if y_true.sum() else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/fusion.pt")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--horizons", default="3,6,12")
    ap.add_argument("--dump-all", action="store_true",
                    help="also write integration/dump_fusion_<split>_p<N>.npz")
    cli = ap.parse_args()
    horizons = [int(h) for h in cli.horizons.split(",")]

    meta_path = os.path.splitext(cli.checkpoint)[0] + ".meta.json"
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        gate_desc = (f"BYPASSED ({meta['single_path']} only)" if meta.get("single_path")
                     else ("extended" if meta.get("extended_gate") else "faithful"))
        print(f"  checkpoint: epoch {meta.get('epoch')}, freeze={meta.get('freeze')}, "
              f"gate={gate_desc}, "
              f"val 12-step {meta.get('val_mae_12step', float('nan')):.4f}")
    else:
        print(f"  ⚠ no {os.path.basename(meta_path)} beside the checkpoint -- the model "
              f"is being rebuilt with DEFAULT settings, which silently mis-loads if it "
              f"was trained with --extended-gate or --head-hidden.")

    n_cn = int(meta.get("stgcn_channels", 1))
    ds = FusionDataset(cli.split, stgcn_channels=n_cn)
    print(" ", ds.describe())
    gso, n_vertex, _ = P.stgcn_gso()
    adj = P.stgat_adj()
    model = DualPathModel(gso.to(cli.device), adj.to(cli.device), n_vertex,
                          n_pred=ds.n_pred, freeze=meta.get("freeze", "none"),
                          extended_gate=bool(meta.get("extended_gate", False)),
                          head_hidden=int(meta.get("head_hidden", 0)),
                          stgcn_channels=n_cn,
                          # 🔴 Load-bearing. A single-path checkpoint still carries W1
                          # and W3, so rebuilding with the default learned gate would
                          # load_state_dict cleanly and then score a gate that was never
                          # trained -- no error, and a perfectly plausible number.
                          single_path=meta.get("single_path"),
                          cuda=cli.device.startswith("cuda")).to(cli.device)
    model.load_state_dict(torch.load(cli.checkpoint, map_location=cli.device))
    model.eval()

    loader = DataLoader(ds, batch_size=cli.batch, shuffle=False, num_workers=0)
    preds, ys, ms, gates = [], [], [], []
    with torch.no_grad():
        for x_cn, x_at, y, m, row0 in loader:
            tod = torch.as_tensor(ds.tod[row0.numpy() - 1], device=cli.device)
            xc, xa = x_cn.to(cli.device), x_at.to(cli.device)
            with contextlib.redirect_stdout(io.StringIO()):    # stgat.py debug prints
                s, t = model.paths(xc, xa)
                z = model.fusion(s, t, tod)
                gates.append(model.fusion.gate_value(s, t, tod).cpu().numpy())
            preds.append(ds.denorm_cn(z.permute(0, 2, 1)).cpu().numpy())   # [B,12,N]
            ys.append(y.numpy())
            ms.append(m.numpy())
    pred = np.concatenate(preds)
    y_all = np.concatenate(ys)
    m_all = np.concatenate(ms)
    gate = np.concatenate(gates)                     # [W, N]

    print(f"\n=== masked scores ({cli.split}) ===")
    print(f"  {'horizon':<10}{'MAE':>9}{'RMSE':>9}{'MAPE':>9}{'WMAPE':>9}{'cells':>11}")
    print("  " + "-" * 57)
    got = {}
    for h in horizons:
        s = metrics(y_all[:, h - 1, :], pred[:, h - 1, :], m_all[:, h - 1, :])
        got[h] = s["MAE"]
        print(f"  {h * 5:>3} min   {s['MAE']:>9.4f}{s['RMSE']:>9.4f}"
              f"{s['MAPE']:>8.2%}{s['WMAPE']:>9.2%}{s['n']:>11,}")

    if cli.split == "test":
        print(f"\n=== against the existing baselines (實驗記錄 §13.11 / §13.18) ===")
        print(f"  {'':<14}" + "".join(f"{h * 5:>9} min" for h in horizons))
        print(f"  {'fusion':<14}" + "".join(f"{got[h]:>13.4f}" for h in horizons))
        for name, ref in REFERENCE.items():
            print(f"  {name:<14}" + "".join(f"{ref.get(h, float('nan')):>13.4f}"
                                            for h in horizons))
        print(f"  {'vs STGAT':<14}"
              + "".join(f"{100 * (got[h] - REFERENCE['STGAT'][h]) / REFERENCE['STGAT'][h]:>12.1f}%"
                        for h in horizons))
        print("  Negative on the last row means the fusion beat the better single model.")
        print("  實驗記錄 §13.7 puts the ceiling low: the best FIXED weight beats STGAT "
              "alone by\n  0.40%, and the two models' errors correlate at 0.934. Anything "
              "under ~1% here is\n  within that noise and should be reported as "
              "'no measurable gain'.")

    print(f"\n=== gate ===")
    print(f"  mean opening                  {gate.mean():.4f}   "
          f"(0 = all STGCN, 1 = all STGAT)")
    print(f"  spread ACROSS SECTIONS        {gate.mean(axis=0).std():.4f}")
    print(f"  spread ACROSS TIME (per sec.) {gate.std(axis=0).mean():.4f}")
    per_sec = gate.mean(axis=0)
    print(f"  most STGCN-leaning section    idx {per_sec.argmin():>3}  gate {per_sec.min():.3f}")
    print(f"  most STGAT-leaning section    idx {per_sec.argmax():>3}  gate {per_sec.max():.3f}")
    if meta.get("single_path"):
        # Constant BY CONSTRUCTION here, so the collapse warning below would be a
        # tautology -- and its second clause is measurably false for this run: a
        # retrained single path does beat the constant-weight ensemble of the two
        # UPSTREAM models, because that ensemble cannot retrain anything.
        print(f"  (constant by construction: the gate is bypassed, "
              f"{meta['single_path']} only)")
    elif max(gate.mean(axis=0).std(), gate.std(axis=0).mean()) < 0.02:
        print("  ⚠ the gate is essentially constant. This model has reproduced a fixed "
              "weighted\n    average, so it cannot beat the constant-weight ensemble by "
              "anything but noise.")

    if cli.dump_all:
        out_dir = os.path.join(P.ROOT, "integration")
        os.makedirs(out_dir, exist_ok=True)
        for h in horizons:
            rows = ds.target_rows(h)
            path = os.path.join(out_dir, f"dump_fusion_{cli.split}_p{h}.npz")
            np.savez(path, pred=pred[:, h - 1, :].astype(np.float32),
                     rows=np.asarray(rows, dtype=np.int64), n_pred=h, split=cli.split)
            print(f"\nOK  {path}  ({len(rows)} windows, rows {rows.min()}..{rows.max()})")
        print("  integration/ha_baseline.py will pick these up as a 'fusion' column.")


if __name__ == "__main__":
    main()
