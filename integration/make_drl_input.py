#!/usr/bin/env python3
"""Turn the two models' section-level predictions into edge-level speeds for routing.

    STGCN/run_infer_taichung.py -> taichung_pred_stgcn.npy  ┐
    STGAT/run_infer_taichung.py -> taichung_pred_stgat.npy  ┴─> this -> taichung_pred_edges.csv

Four things happen here, all of them deliberately downstream of inference so that a
misbehaving model stays visible in its own .npy:

  0. check    both .npy files describe the SAME moment and the SAME horizon, using the
              .meta.json each run_infer_taichung.py writes beside its output
  1. clamp    raw predictions to a physical range (routing cost is length/speed, so a
              near-zero or negative speed would blow it up)
  2. ensemble W_STGCN * stgcn + W_STGAT * stgat
  3. map      each TDX section's speed onto every OSM edge it covers (3.1 on average,
              via section_to_edges.csv); an edge on two sections gets their mean

Step 0 exists because the two models slice their test windows differently, so running
both with the default `--index -1` predicts moments up to 11 steps (55 min) apart. The
ensemble of two mismatched snapshots is a plausible-looking, wrong CSV and nothing else
in the pipeline would notice -- so this refuses to write instead of warning. Pass
--target-row to whichever script is behind (both print the row they predicted).

All three speeds are kept as separate columns, not just the ensemble, because the
proposal's baselines (2) pure-STGCN and (3) pure-STGAT each route on their own
prediction while (4) routes on the ensemble.

Output columns: from_node, to_node, speed_stgcn, speed_stgat, speed_hybrid

--------------------------------------------------------------------------------
--source fusion

Baseline (4) is "every vehicle follows the hybrid prediction", and since fusion/ was
built the hybrid IS the gated-fusion model (計劃書 §4.3) rather than the fixed
0.2/0.8 constant. --source fusion takes speed_hybrid from fusion's own output instead
of re-mixing the two single models.

It reads the dump fusion/evaluate.py --dump-all writes, and selects the requested row
from it: the dump already carries its `rows` array, so no second inference pass is
needed and the provenance stays traceable to one file. speed_stgcn / speed_stgat still
come from the two .npy files, because baselines (2) and (3) are defined as the single
models routing on their own forecast.

⚠️ Changing this changes what the AGENT OBSERVES -- tpred reaches it through
feats[:,1] and through edge_static, which is a state-dict buffer. Swapping the source
under a trained checkpoint is the same class of silent mismatch as togo_refresh
(實驗記錄 §13.16 ⑥), so the agent has to be retrained. The reward itself is unaffected:
_gcost is built from t0, not tpred.

Usage:
    cd integration && python make_drl_input.py
    python make_drl_input.py --source fusion --target-row 50273
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import config as C

MAP_DIR = C.ROOT / "Map"
PRED = {"stgcn": C.HERE / "taichung_pred_stgcn.npy",
        "stgat": C.HERE / "taichung_pred_stgat.npy"}
OUT_CSV = C.HERE / "taichung_pred_edges.csv"


def read_meta(npy_path):
    """The .meta.json beside a prediction, or None for a file written before sidecars."""
    p = npy_path.with_suffix(".meta.json")
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_fusion(split, n_pred, target_row, n_sec, npy_row=None):
    """One row of fusion's dump -> (speeds, meta). See --source fusion above."""
    path = C.HERE / f"dump_fusion_{split}_p{n_pred}.npz"
    if not path.is_file():
        raise SystemExit(
            f"ERROR  {path.name} not found.\n"
            f"       cd ../fusion && python evaluate.py "
            f"--checkpoint checkpoints/fusion_c.pt --split {split} --dump-all")
    d = np.load(path)
    rows = d["rows"]
    hit = np.flatnonzero(rows == target_row)
    if hit.size == 0:
        extra = ""
        if npy_row is not None and npy_row != target_row:
            extra = (f"\n       Note the two .npy predictions are at row {npy_row}; "
                     f"speed_stgcn / speed_stgat\n       come from those, so a different "
                     f"row here would put three different moments in one CSV.")
        raise SystemExit(
            f"ERROR  row {target_row} is not in {path.name} "
            f"(it covers {rows.min()}..{rows.max()}).{extra}")
    v = d["pred"][int(hit[0])].flatten()
    if v.size != n_sec:
        raise ValueError(f"{path.name} has {v.size} sections but the section index has "
                         f"{n_sec} -- both must come from the same build_speed.py run")
    fmeta = C.ROOT / "fusion" / "checkpoints" / "fusion_c.meta.json"
    epoch = None
    if fmeta.is_file():
        with open(fmeta, encoding="utf-8") as f:
            epoch = json.load(f).get("epoch")
    return v, {"model": "fusion", "n_pred": int(n_pred), "split": split,
               "target_row": int(target_row), "n_sections": int(v.size),
               "fusion_epoch": epoch, "dump": path.name}


def rerun_cmd(name, n_pred, row):
    """The command that re-runs one model at a given horizon and row."""
    if name == "stgcn":
        return (f"cd ../STGCN && python run_infer_taichung.py --n-pred {n_pred} "
                f"--target-row {row} --checkpoint STGCN_taichung_p{n_pred}.pt")
    return (f"cd ../STGAT && python run_infer_taichung.py --n-pred {n_pred} "
            f"--target-row {row}")


def check_alignment(metas, strict=True):
    """Refuse to ensemble snapshots of different moments or different horizons."""
    print("\n=== alignment ===")
    missing = [k for k, m in metas.items() if m is None]
    if missing:
        print(f"  no .meta.json for {', '.join(missing)} -- written before the sidecar "
              f"existed, or by hand. Alignment cannot be verified; re-run "
              f"run_infer_taichung.py to regenerate.")
        return
    for k, m in metas.items():
        print(f"  {k:<6} row {m['target_row']:>6}  {m['target_time']}  "
              f"({m['minutes_ahead']} min ahead, window {m['window_index']})")
    if len(metas) < 2:
        return

    rows = {m["target_row"] for m in metas.values()}
    preds = {m["n_pred"] for m in metas.values()}
    if len(rows) == 1 and len(preds) == 1:
        print("  OK  both models predicted the same row at the same horizon")
        return

    problems, fixes = [], []
    if len(preds) > 1:
        # Horizons differ: nothing to reconcile by moving a row, both must be re-run.
        problems.append(f"different horizons: n_pred {sorted(preds)} "
                        f"(= {sorted(p * 5 for p in preds)} min ahead)")
        p = min(preds)
        fixes.append(f"Pick one horizon and re-run both, e.g. at {p * 5} min:")
        fixes += [f"  {rerun_cmd(k, p, min(rows))}" for k in sorted(metas)]
    else:
        # Same horizon, different moments. Move the later model back to the earlier row:
        # the earlier row is inside the later model's reach (it already predicts past
        # it), whereas the later row is often beyond the earlier model's last window.
        gap = max(rows) - min(rows)
        problems.append(f"different moments: target rows {sorted(rows)} "
                        f"({gap} steps = {gap * 5} min apart)")
        shared = min(rows)
        late = max(metas, key=lambda k: metas[k]["target_row"])
        fixes.append(f"Move {late} back to row {shared} (the row both models reach):")
        fixes.append(f"  {rerun_cmd(late, metas[late]['n_pred'], shared)}")

    msg = ("cannot ensemble these predictions:\n    " + "\n    ".join(problems) +
           "\n  " + "\n  ".join(fixes) +
           "\n  Each model reaches a different row range -- both scripts print theirs -- "
           "so the shared row is usually not either model's last window.\n"
           "  Override with --allow-misaligned only for a deliberate diagnostic.")
    if strict:
        raise SystemExit(f"ERROR  {msg}")
    print(f"  WARNING  {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w-stgcn", type=float, default=C.W_STGCN)
    ap.add_argument("--w-stgat", type=float, default=C.W_STGAT)
    ap.add_argument("--speed-min", type=float, default=C.TAICHUNG_SPEED_MIN_KMH)
    ap.add_argument("--speed-max", type=float, default=C.TAICHUNG_SPEED_MAX_KMH)
    ap.add_argument("--allow-misaligned", action="store_true",
                    help="ensemble even if the two models predicted different moments "
                         "or horizons. For diagnostics only -- the result is not a "
                         "snapshot of any real instant.")
    ap.add_argument("--source", choices=["ensemble", "fusion"], default="ensemble",
                    help="where speed_hybrid comes from: the fixed-weight mix of the two "
                         ".npy files (default), or the gated-fusion model's own output. "
                         "Changing this requires retraining the agent -- see the module "
                         "docstring")
    ap.add_argument("--target-row", type=int, default=None,
                    help="--source fusion only: which row of the fusion dump to take. "
                         "Defaults to the row the two .npy files were taken at, so the "
                         "two sources describe the same moment")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="--source fusion only: which dump split to read")
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    idx_csv = MAP_DIR / "taichung_section_index.csv"
    s2e_csv = MAP_DIR / "section_to_edges.csv"
    for p, name in ((idx_csv, "section index"), (s2e_csv, "section->edges mapping")):
        if not p.is_file():
            raise FileNotFoundError(f"{name} not found: {p}\n"
                                    f"Run TDX_Data/build_network.py and build_speed.py first.")

    sec = pd.read_csv(idx_csv, encoding="utf-8-sig").sort_values("matrix_index")
    n_sec = len(sec)

    # --- load whichever predictions exist ---
    raw = {}
    for name, path in PRED.items():
        if path.is_file():
            v = np.load(path).flatten()
            if v.size != n_sec:
                raise ValueError(f"{path.name} has {v.size} values but the section index "
                                 f"has {n_sec} rows -- both must come from the same "
                                 f"build_speed.py run")
            raw[name] = v
        else:
            print(f"  note: {path.name} not found -- skipping that model")
    if not raw:
        raise FileNotFoundError("no predictions found; run the two run_infer_taichung.py first")

    metas = {k: read_meta(PRED[k]) for k in raw}
    check_alignment(metas, strict=not args.allow_misaligned)

    fusion = fusion_meta = None
    if args.source == "fusion":
        known = {m["target_row"] for m in metas.values() if m}
        n_preds = {m["n_pred"] for m in metas.values() if m}
        row = args.target_row if args.target_row is not None else (
            known.pop() if len(known) == 1 else None)
        if row is None:
            raise SystemExit(
                "ERROR  cannot infer which row to read from the fusion dump: the two "
                ".npy sidecars\n       disagree or are missing. Pass --target-row "
                "explicitly.")
        n_pred = n_preds.pop() if len(n_preds) == 1 else 3
        fusion, fusion_meta = load_fusion(args.split, n_pred, row, n_sec,
                                          npy_row=known.copy().pop() if known else None)
        print(f"\n=== fusion ===")
        print(f"  {fusion_meta['dump']} row {row} (n_pred {n_pred}, "
              f"{n_pred * 5} min), fusion epoch {fusion_meta['fusion_epoch']}")
        n_out = int(((fusion < args.speed_min) | (fusion > args.speed_max)).sum())
        print(f"  {fusion.min():.1f}..{fusion.max():.1f} km/h, mean {fusion.mean():.1f}"
              f", out of range: {n_out}")

    print("\n=== raw predictions (km/h) ===")
    for name, v in raw.items():
        n_out = int(((v < args.speed_min) | (v > args.speed_max)).sum())
        print(f"  {name:<6} {v.min():7.1f}..{v.max():7.1f}   mean {v.mean():5.1f}"
              f"   out of range: {n_out}")
        if n_out:
            print(f"         ^ clamped to [{args.speed_min}, {args.speed_max}]; a large "
                  f"count here means the model is misbehaving, not the clamp")

    clamped = {k: np.clip(v, args.speed_min, args.speed_max) for k, v in raw.items()}

    # --- ensemble; renormalise if only one model ran, so its weight is not silently 0.7 ---
    w = {"stgcn": args.w_stgcn, "stgat": args.w_stgat}
    total = sum(w[k] for k in clamped)
    mix = sum(clamped[k] * (w[k] / total) for k in clamped)
    used = " + ".join(f"{w[k] / total:.2f}*{k}" for k in clamped)
    print(f"\n=== hybrid ===")
    if fusion is None:
        hybrid = mix
        print(f"  fixed-weight ensemble: {used}")
    else:
        hybrid = np.clip(fusion, args.speed_min, args.speed_max)
        # The two are genuinely different forecasts, not a re-scaling of each other
        # (measured on the test split at 60 min: corr 0.896, median relative difference
        # 3.34%). Printing the divergence here is what makes the retrain non-optional
        # obvious at the point of switching.
        rel = np.abs(hybrid - mix) / np.maximum(mix, 1e-6)
        print(f"  gated fusion (計劃書 §4.3), replacing the fixed-weight mix {used}")
        print(f"  vs that mix: corr {np.corrcoef(hybrid, mix)[0, 1]:.4f}, "
              f"median relative difference {np.median(rel):.2%}, "
              f"p90 {np.percentile(rel, 90):.2%}")
        print(f"  ⚠ the agent observes tpred (feats[:,1] and edge_static), so a "
              f"checkpoint trained on\n    the other source must be RETRAINED; its "
              f"reward is unaffected (_gcost uses t0).")
    print(f"  {hybrid.min():.1f}..{hybrid.max():.1f} km/h, mean {hybrid.mean():.1f}")

    sec = sec.copy()
    for k, v in clamped.items():
        sec[f"speed_{k}"] = v
    sec["speed_hybrid"] = hybrid

    # --- section -> OSM edges ---
    s2e = pd.read_csv(s2e_csv, encoding="utf-8-sig")
    cols = [f"speed_{k}" for k in clamped] + ["speed_hybrid"]
    merged = s2e.merge(sec[["SectionID"] + cols], on="SectionID", how="inner")
    # One edge can lie on two sections' paths; average so the export has one row per edge
    # (the routing graph stores a single speed per edge anyway).
    per_edge = merged.groupby(["from_node", "to_node"], as_index=False)[cols].mean()

    per_edge.to_csv(args.out, index=False, encoding="utf-8-sig")
    # Sidecar, for the same reason every other artefact here has one: two CSVs written
    # from different sources are indistinguishable on disk, and the one the agent was
    # trained against decides whether its checkpoint is still valid.
    side = {"source": args.source, "n_edges": int(len(per_edge)),
            "speed_min": args.speed_min, "speed_max": args.speed_max,
            "inputs": {k: metas.get(k) for k in raw}}
    if fusion_meta:
        side["fusion"] = fusion_meta
    else:
        side["weights"] = {k: w[k] / total for k in clamped}
    side_path = os.path.splitext(str(args.out))[0] + ".meta.json"
    with open(side_path, "w", encoding="utf-8") as f:
        json.dump(side, f, indent=2, ensure_ascii=False)

    dropped_rows = len(s2e) - len(merged)
    print(f"\n=== section -> edge ===")
    print(f"  {len(sec)} sections -> {len(per_edge)} edges "
          f"({len(merged) - len(per_edge)} overlaps averaged)")
    if dropped_rows:
        print(f"  {dropped_rows} mapping rows belong to sections dropped by build_speed.py "
              f"(high missing rate); those edges keep their free-flow time")
    print(f"\nOK  {args.out}")
    print(f"\nThe decision layer picks this up automatically:")
    print(f"  python run_compare.py --graph taichung --vehicles 800")


if __name__ == "__main__":
    main()
