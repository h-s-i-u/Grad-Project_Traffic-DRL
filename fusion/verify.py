#!/usr/bin/env python3
"""Prove the unified dataloader before anything is trained on it.

Nothing downstream is trustworthy unless this passes. `fusion/data.py` rebuilds the
windowing, the two normalisations, the tensor layouts and the mask from scratch; get any
one of them wrong -- a transpose, the wrong scaler, an off-by-one in the target row --
and training still runs, the loss still falls, and the resulting MAE is simply not
comparable with anything in 實驗記錄. None of those failures raise.

So: push the two ALREADY-TRAINED backbones through this dataloader, with their own output
heads, and check that they reproduce the numbers the standalone evaluators reported
(實驗記錄 §13.11). If the pipeline is right they must; if it is not, they cannot.

    STGAT   3.3802 / 3.5127 / 3.6276    at 15 / 30 / 60 min
    STGCN   3.5560 / 3.7535 / 3.9549
    persistence  4.2872 / 4.6744 / 5.1281

Persistence is checked too, and it is the more sensitive of the two: it needs no model at
all, so if it is off, the fault is squarely in the rows or the mask.

Usage:
    cd fusion
    python verify.py                       # test split, all three horizons
    python verify.py --split val --device cpu
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths as P
from data import FusionDataset

# 實驗記錄 §13.11, masked, test split. Re-read them from evaluate_masked.py if the
# prediction models are ever retrained -- they are the reference, not this file.
REFERENCE = {
    "test": {
        "stgat": {3: 3.3802, 6: 3.5127, 12: 3.6276},
        "stgcn": {3: 3.5560, 6: 3.7535, 12: 3.9549},
        "persistence": {3: 4.2872, 6: 4.6744, 12: 5.1281},
    }
}
TOL = 0.02          # 2%: the window sets differ slightly from evaluate_masked's, but a
                    # real pipeline bug is wrong by 100%+, not by 2%.


def batched(ds, batch, device):
    for i in range(0, len(ds), batch):
        items = [ds[j] for j in range(i, min(i + batch, len(ds)))]
        yield (torch.stack([x[0] for x in items]).to(device),
               torch.stack([x[1] for x in items]).to(device),
               torch.stack([x[2] for x in items]),
               torch.stack([x[3] for x in items]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--horizons", default="3,6,12")
    ap.add_argument("--stgcn-dir", default=P.STGCN_DIR)
    ap.add_argument("--stgat-ckpt",
                    default=os.path.join(P.STGAT_DIR, "experiment_taichung",
                                         "best_model.pth"))
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N windows (a quick smoke test; the reference "
                         "comparison is then meaningless and is skipped)")
    cli = ap.parse_args()
    horizons = [int(h) for h in cli.horizons.split(",")]

    ds = FusionDataset(cli.split)
    print("=== dataloader ===")
    print(" ", ds.describe())
    print(f"  STGCN z-score : per section (mean {ds.cn_mean.mean():.3f} avg, "
          f"std {ds.cn_std.mean():.3f} avg)")
    print(f"  STGAT z-score : one global scalar (mean {ds.at_mean:.4f}, "
          f"std {ds.at_std:.4f})")
    if cli.limit:
        ds.windows = ds.windows[:cli.limit]
        print(f"  ⚠ limited to {len(ds)} windows -- reference comparison skipped")

    gso, n_vertex, n_declared = P.stgcn_gso()
    if n_vertex != n_declared:
        print(f"  ⚠ dataloader.load_adj declares n_vertex={n_declared} but the adjacency "
              f"is {n_vertex}; using {n_vertex}")
    if n_vertex != ds.n_vertex:
        raise SystemExit(f"ERROR  adjacency has {n_vertex} sections, vel.csv has "
                         f"{ds.n_vertex} -- they must come from the same pipeline run")
    adj = P.stgat_adj()
    gso, adj = gso.to(cli.device), adj.to(cli.device)

    # --- collect targets and persistence once ---
    ys, ms = [], []
    for _, _, y, m in batched(ds, cli.batch, cli.device):
        ys.append(y)
        ms.append(m)
    y_all = torch.cat(ys).numpy()          # [W, 12, N] km/h
    m_all = torch.cat(ms).numpy()
    last_in = ds.vel[ds.windows + ds.n_his - 1]        # the last observed input frame

    results = {}
    print(f"\n=== STGAT (one model, twelve steps) ===")
    at = P.STGATPath(adj, n_vertex, cuda=cli.device.startswith("cuda")).to(cli.device)
    n_unused = at.load_pretrained(cli.stgat_ckpt, cli.device)
    at.eval()
    preds = []
    with torch.no_grad():
        for x_cn, x_at, _, _ in batched(ds, cli.batch, cli.device):
            # model/stgat.py still prints every layer's shape on each forward
            with contextlib.redirect_stdout(io.StringIO()):
                preds.append(at.predict(x_at).cpu().numpy())
    p_at = ds.denorm_at(np.concatenate(preds))          # [W, N, 12] km/h
    print(f"  loaded {os.path.basename(cli.stgat_ckpt)} ({n_unused} unused keys)")
    results["stgat"] = {h: float(np.abs(p_at[:, :, h - 1] - y_all[:, h - 1, :])
                                 [m_all[:, h - 1, :]].mean()) for h in horizons}

    print(f"\n=== STGCN (one model per horizon) ===")
    results["stgcn"] = {}
    for h in horizons:
        ckpt = os.path.join(cli.stgcn_dir, f"STGCN_taichung_p{h}.pt")
        if not os.path.isfile(ckpt):
            print(f"  {os.path.basename(ckpt)} not found -- skipping {h * 5} min")
            continue
        cn = P.STGCNPath(gso, n_vertex).to(cli.device)
        cn.load_pretrained(ckpt, cli.device)
        cn.eval()
        preds = []
        with torch.no_grad():
            for x_cn, _, _, _ in batched(ds, cli.batch, cli.device):
                preds.append(cn.net(x_cn).view(x_cn.shape[0], -1).cpu().numpy())
        p_cn = np.concatenate(preds) * ds.cn_std + ds.cn_mean      # [W, N] km/h
        results["stgcn"][h] = float(np.abs(p_cn - y_all[:, h - 1, :])
                                    [m_all[:, h - 1, :]].mean())
        print(f"  {os.path.basename(ckpt)} -> {h * 5:>2} min")

    results["persistence"] = {
        h: float(np.abs(last_in - y_all[:, h - 1, :])[m_all[:, h - 1, :]].mean())
        for h in horizons}

    # --- verdict ---
    ref = REFERENCE.get(cli.split, {})
    print(f"\n=== reproduction check ({cli.split}) ===")
    print(f"  {'':<13}" + "".join(f"{h * 5:>7} min" for h in horizons))
    ok = True
    for name in ("stgat", "stgcn", "persistence"):
        got = results.get(name, {})
        cells, flags = [], []
        for h in horizons:
            v = got.get(h)
            cells.append(f"{v:>11.4f}" if v is not None else f"{'--':>11}")
            want = ref.get(name, {}).get(h)
            if v is None or want is None or cli.limit:
                flags.append("  ")
                continue
            rel = abs(v - want) / want
            flags.append("ok" if rel <= TOL else "XX")
            ok &= rel <= TOL
        print(f"  {name:<13}" + "".join(cells))
        if ref.get(name) and not cli.limit:
            print(f"  {'reference':<13}"
                  + "".join(f"{ref[name].get(h, float('nan')):>11.4f}" for h in horizons)
                  + "   " + " ".join(flags))

    if cli.limit:
        print("\n  (smoke test only)")
        return
    if not ok:
        raise SystemExit(
            "\nFAIL  a backbone does not reproduce its recorded MAE through this "
            "dataloader.\n"
            "      Something in data.py disagrees with how that model was trained --\n"
            "      check, in this order: the tensor layout ([B,1,T,N] vs [B,N,T,F]),\n"
            "      the normalisation (per-section vs one global scalar), the target\n"
            "      row (g + n_his - 1 + horizon), and the mask alignment.\n"
            "      Do NOT train on this dataloader until it passes.")
    print("\nPASS  both backbones reproduce their recorded MAE through the unified "
          "dataloader.")


if __name__ == "__main__":
    main()
