#!/usr/bin/env python3
"""Choose the STGCN/STGAT ensemble weight by measurement instead of by guess.

config.py ships W_STGCN=0.7 / W_STGAT=0.3, inherited from the METR-LA runs where STGAT
had not yet converged and deserved the smaller share. On Taichung STGAT wins at every
horizon (15/30/60 min MAE 3.4716 / 3.5905 / 3.6811 against STGCN's 3.6259 / 3.8431 /
4.0288), so that ratio now weights the weaker model more heavily. This script sweeps the
weight on the VALIDATION split and reports the best one.

It also answers the question the weight alone hides: whether the ensemble earns its
keep at all. Averaging two forecasts helps only when their errors are decorrelated -- if
both models are wrong on the same sections at the same moments, the best mix collapses
to "use the better model" and the honest result is to report a single model, or to build
the proposal's Gated Fusion (eq. 3), which weights per node and per timestep rather than
with one global constant. The printed error correlation is what tells them apart.

Scoring is masked, for the same reason the two evaluators are: build_speed.py imputed
23.7% of the matrix (mostly ffill, i.e. a copy of the previous step), and a model
reproduces those cells at almost no cost. Including them would flatter both models
unevenly and could pick the wrong weight.

Inputs:
    dump_stg{cn,at}_<split>_p<N>.npz    from run_infer_taichung.py --dump-all
    ../Map/taichung_vel.csv             ground truth, one row per timestep
    ../Map/taichung_mask.npy            True where the cell is a real observation

Usage:
    cd STGCN
    python run_infer_taichung.py --split val --dump-all --n-pred 3 \
        --checkpoint STGCN_taichung_p3.pt
    cd ../STGAT
    python run_infer_taichung.py --split val --dump-all --n-pred 3
    cd ../integration
    python search_ensemble_weight.py --n-pred 3

    # then confirm the chosen weight on data it was not selected on
    #   (re-dump both with --split test, then)
    python search_ensemble_weight.py --n-pred 3 --split test
"""
import argparse

import numpy as np
import pandas as pd

import config as C

MAP_DIR = C.ROOT / "Map"


def masked_scores(pred, truth):
    """MAE / RMSE over already-masked, already-flattened arrays."""
    d = np.abs(pred - truth)
    return float(d.mean()), float(np.sqrt((d ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pred", type=int, default=3,
                    help="horizon the dumps were made at (3/6/12 = 15/30/60 min)")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="val to choose the weight; test only to confirm it afterwards")
    ap.add_argument("--step", type=float, default=0.05, help="weight grid resolution")
    ap.add_argument("--speed-min", type=float, default=C.TAICHUNG_SPEED_MIN_KMH)
    ap.add_argument("--speed-max", type=float, default=C.TAICHUNG_SPEED_MAX_KMH)
    args = ap.parse_args()

    # --- load the two dumps ---
    dumps = {}
    for name in ("stgcn", "stgat"):
        p = C.HERE / f"dump_{name}_{args.split}_p{args.n_pred}.npz"
        if not p.is_file():
            raise FileNotFoundError(
                f"{p.name} not found. Produce it with:\n"
                f"  cd ../{name.upper()} && python run_infer_taichung.py "
                f"--split {args.split} --dump-all --n-pred {args.n_pred}"
                + (f" --checkpoint STGCN_taichung_p{args.n_pred}.pt"
                   if name == "stgcn" else ""))
        dumps[name] = np.load(p)

    print("=== inputs ===")
    for name, d in dumps.items():
        r = d["rows"]
        print(f"  {name:<6} {len(r):>6,} windows   rows {r.min()}..{r.max()}")

    # The two models' splits start at different rows, so intersect on the absolute row
    # rather than assuming the arrays line up index-for-index. This is the same trap the
    # single-shot path hit: matching window numbers do not mean matching moments.
    rows, i_cn, i_at = np.intersect1d(dumps["stgcn"]["rows"], dumps["stgat"]["rows"],
                                      return_indices=True)
    if rows.size == 0:
        raise SystemExit("ERROR  the two dumps share no rows -- were they made at the "
                         "same horizon and on the same split?")
    pred = {"stgcn": dumps["stgcn"]["pred"][i_cn], "stgat": dumps["stgat"]["pred"][i_at]}
    print(f"  shared {len(rows):,} rows ({rows.min()}..{rows.max()}); "
          f"dropped {len(dumps['stgcn']['rows']) - len(rows)} stgcn / "
          f"{len(dumps['stgat']['rows']) - len(rows)} stgat")

    # --- ground truth + mask, sliced to the shared rows ---
    vel = pd.read_csv(MAP_DIR / "taichung_vel.csv", encoding="utf-8-sig").to_numpy(np.float32)
    mask = np.load(MAP_DIR / "taichung_mask.npy")
    if vel.shape != mask.shape:
        raise ValueError(f"vel {vel.shape} and mask {mask.shape} disagree -- both come "
                         f"from build_speed.py, so re-run it")
    if pred["stgcn"].shape[1] != vel.shape[1]:
        raise ValueError(f"predictions have {pred['stgcn'].shape[1]} sections but "
                         f"taichung_vel.csv has {vel.shape[1]}")
    truth, valid = vel[rows], mask[rows]

    # Clamp exactly as make_drl_input.py does, so the weight is chosen on the numbers the
    # router will actually receive rather than on raw output it never sees.
    y = truth[valid]
    p = {k: np.clip(v, args.speed_min, args.speed_max)[valid] for k, v in pred.items()}

    print(f"\n=== masked scoring ({args.split}) ===")
    print(f"  {valid.sum():,} real observations of {valid.size:,} cells "
          f"({valid.mean():.1%}); the other {1 - valid.mean():.1%} are imputed and dropped")

    single = {k: masked_scores(v, y) for k, v in p.items()}
    for k, (mae, rmse) in single.items():
        print(f"  {k:<6} alone   MAE {mae:.4f}   RMSE {rmse:.4f}")
    best_single = min(single, key=lambda k: single[k][0])
    best_single_mae = single[best_single][0]

    # --- sweep ---
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, args.step), 4)
    print(f"\n=== weight sweep (w = share given to STGCN) ===")
    print(f"  {'w_stgcn':>8}{'w_stgat':>9}{'MAE':>10}{'RMSE':>10}"
          f"{'vs ' + best_single:>14}")
    print("  " + "-" * 51)
    results = []
    for w in grid:
        mae, rmse = masked_scores(w * p["stgcn"] + (1 - w) * p["stgat"], y)
        results.append((float(w), mae, rmse))
    best_w, best_mae, best_rmse = min(results, key=lambda r: r[1])

    cur = C.W_STGCN / (C.W_STGCN + C.W_STGAT)      # config need not be normalised
    cur_w = round(round(cur / args.step) * args.step, 4)

    def mark(w):
        """Both tags can land on the same row, so build the label instead of a lookup."""
        tags = ([] if abs(w - best_w) > 1e-9 else ["best"]) + \
               ([] if abs(w - cur_w) > 1e-9 else ["current config"])
        return "  <- " + " / ".join(tags) if tags else ""

    for w, mae, rmse in results:
        print(f"  {w:>8.2f}{1 - w:>9.2f}{mae:>10.4f}{rmse:>10.4f}"
              f"{(mae - best_single_mae) / best_single_mae * 100:>+13.2f}%{mark(w)}")

    # --- is the ensemble worth having? ---
    gain = (best_single_mae - best_mae) / best_single_mae * 100
    err = {k: v - y for k, v in p.items()}
    corr = float(np.corrcoef(err["stgcn"], err["stgat"])[0, 1])

    print(f"\n=== error correlation ===")
    print(f"  corr(err_stgcn, err_stgat) = {corr:.3f}")
    if corr > 0.9:
        print("  -> the two models are wrong in the same places; a global weight has "
              "almost nothing to exploit")
    elif corr > 0.7:
        print("  -> substantially correlated; expect a small ensemble gain")
    else:
        print("  -> fairly independent errors; ensembling should pay off")

    print(f"\n=== verdict ===")
    print(f"  best mix   w_stgcn {best_w:.2f} / w_stgat {1 - best_w:.2f}   "
          f"MAE {best_mae:.4f}  RMSE {best_rmse:.4f}")
    print(f"  best single {best_single} alone                     MAE {best_single_mae:.4f}")
    if gain < 0.5:
        print(f"  The ensemble beats the better single model by only {gain:.2f}%. That is "
              f"within\n  seed-to-seed noise, so a fixed global weight is not buying "
              f"anything here.\n  Report {best_single} alone, or implement the proposal's "
              f"Gated Fusion (eq. 3):\n  a per-node, per-timestep gate can exploit local "
              f"differences that one constant cannot.")
    else:
        print(f"  The ensemble beats the better single model by {gain:.2f}%. Worth keeping.")
    if args.split == "val":
        print(f"\n  Put this in config.py:")
        print(f"      W_STGCN, W_STGAT = {best_w:.2f}, {1 - best_w:.2f}")
        print(f"  then confirm on data the weight was NOT chosen on: re-dump both models "
              f"with\n  --split test and re-run this with --split test. A weight that "
              f"only wins on val\n  is overfitted to it.")
    else:
        print(f"\n  This is the TEST split -- use it to confirm the weight chosen on val, "
              f"not to pick one.")


if __name__ == "__main__":
    main()
