# -*- coding: utf-8 -*-
"""
transfer_taichung.py
────────────────────
A3-a: carry the METR-LA STGAT onto the Taichung road network WITHOUT training on it.

This is the first half of the proposal's §5 claim, on its own:

    「以 METR-LA 訓練的模型應可直接遷移至台中路網進行 zero-shot 推論，
      並在 fine-tune 數個 epoch 後達到可用水準。」

Nothing here optimises anything. There is no loss, no optimizer, no training loop. The
script builds a Taichung-shaped STGAT, copies across every weight whose shape survives
the change of road network, fills in the ones that cannot, and writes the result out for
`evaluate_masked_taichung.py` to score. The SECOND half (fine-tuning) is
`train.py --init-from`, and it does use the Taichung training split -- that is what
fine-tuning means.

WHAT TRANSFERS, MEASURED
    332 tensors, identical key sets between the two checkpoints.
      298 tensors / 5,153,952 params (95.5%)  same shape, copied verbatim:
          the gated temporal convolutions, the GAT's W / a1 / a2, downsample, EndConv
       34 tensors /   241,776 params (4.5%)   bound to the node count, 207 -> 202:
          18x per-node GAT bias [N, 64] and 16x BatchNorm2d over the node axis [N]

    STGAT is inductive in principle -- attention is computed from shared weights, so a
    different graph is not a problem -- and the measurement says the architecture is 95%
    of the way there. What breaks zero-shot is 34 tensors' worth of implementation
    choice, not the method. 實驗記錄 §14.3 argued that from reading the source; this
    counts it.

WHAT "ZERO-SHOT" MEANS HERE, PRECISELY
    No gradient step is taken on Taichung. It is NOT "no Taichung data at all": the
    z-score statistics come from Taichung's own train split, because the two cities are
    in different units (mph vs km/h) and METR-LA's mean and standard deviation are
    meaningless against km/h. The model works entirely in z-space, which is what makes
    the transfer conceivable at all. This must be stated in the report.

Usage:
    cd STGAT
    python transfer_taichung.py
    python evaluate_masked_taichung.py --checkpoint transfer_experiment/init_model.pth
"""

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
import torch

import util
from model.stgat import STGAT

# The docstring quotes the proposal in Chinese and the report lines use a warning glyph,
# while Windows hands a redirected stdout cp1252 -- so piping this to a file or a pager
# would die on the text rather than on anything real.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def classify(src, dst):
    """Split the target's tensors into (copied, node-bound, absent-from-source)."""
    same, bound, missing = [], [], []
    for k, v in dst.items():
        if k not in src:
            missing.append(k)
        elif src[k].shape == v.shape:
            same.append(k)
        else:
            bound.append(k)
    return same, bound, missing


def fill_node_bound(src, dst, keys, how):
    """Give the target's node-bound tensors a value the source can justify.

    'mean' hands every new node the SOURCE network's average node. Node identity does
    not carry across cities -- sensor 41 in Los Angeles is not section 41 in Taichung --
    so a per-node mean is the only thing the source can offer that does not depend on an
    invented alignment. Slicing the first 202 of 207 would be exactly such an invention.

    'fresh' leaves PyTorch's initialisation, i.e. transfers nothing for these tensors.
    Keep it as the control: the gap between the two is what the source's node statistics
    were worth.
    """
    if how == "fresh":
        return
    for k in keys:
        m = src[k].float().mean(dim=0, keepdim=True)          # [1, ...]
        dst[k] = m.expand_as(dst[k]).contiguous().to(dst[k].dtype)


def recalibrate_bn(net, data, adj_path, cli):
    """Re-estimate the BatchNorm running statistics on the TARGET's train split.

    No gradient step, no weight changes. Only running_mean / running_var move, and those
    are precisely the tensors that cannot cross between road networks -- a BatchNorm over
    the node axis has one statistic per node, and node 41 in Los Angeles is not node 41
    in Taichung.

    Why this is worth separating out: with the statistics left at PyTorch's defaults
    (mean 0, var 1) the layer performs no normalisation at all, while the transferred
    convolutions were trained expecting variances of 0.7-5.7. The transferred weights are
    then unusable for a reason that has nothing to do with whether Los Angeles traffic
    resembles Taichung traffic. This tells the two apart.

    Only the BatchNorm modules go into train mode. Putting the whole net there would also
    switch dropout on, and dropout noise inflates the very variances being estimated.
    `momentum=None` makes each layer accumulate an exact running average over the windows
    it is shown rather than an exponential one.
    """
    import torch.nn as nn

    _, _, adj_list = util.load_adj(adj_path, "symnadj")
    adj_mx = torch.from_numpy(np.array(adj_list, dtype=np.float32))[0].to(cli.device)
    x = data["x_train"]
    n = min(cli.recalibrate_bn, len(x))
    net = net.to(cli.device)
    net.eval()
    bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    print(f"\nBN recalibration: {len(bns)} BatchNorm layers, {n:,} target train windows, "
          f"no gradient step")
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        for i in range(0, n, cli.batch_size):
            xb = torch.Tensor(x[i:i + cli.batch_size].transpose(0, 2, 1, 3)).to(cli.device)
            net(adj_mx, xb)
    net.eval()
    rv = [float(m.running_var.mean()) for m in bns[:4]]
    print(f"  running_var after (first 4 layers): "
          f"{', '.join(f'{v:.3f}' for v in rv)}")
    print(f"  ⚠ report this as ZERO-SHOT + BN ADAPTATION, not as plain zero-shot: "
          f"target data was used,\n    though no weight was trained.")
    net.cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="experiment_METR_LA/best_model.pth",
                    help="the checkpoint trained on the OTHER road network")
    ap.add_argument("--data", default="data/taichung/",
                    help="target dataset; only its shapes and its scaler are read")
    ap.add_argument("--adj", default=None, help="defaults to <data>/adj_mx_dijsk.pkl")
    ap.add_argument("--out-dir", default="transfer_experiment")
    ap.add_argument("--out-name", default="init_model.pth",
                    help="filename inside --out-dir; give the controls their own so they "
                         "do not overwrite the transferred model")
    ap.add_argument("--no-transfer", action="store_true",
                    help="control: build the same architecture and save it UNTRAINED, "
                         "copying nothing. This is the floor -- the gap between it and a "
                         "real transfer is what the shared weights were actually worth, "
                         "and without it 'zero-shot failed' cannot distinguish 'the "
                         "transfer carried nothing' from 'it carried something unusable'")
    ap.add_argument("--node-init", choices=["mean", "fresh"], default="mean",
                    help="what the node-bound tensors get. 'mean' = the source's average "
                         "node (the strongest honest reading of transfer); 'fresh' = "
                         "PyTorch init, transferring nothing for those tensors")
    ap.add_argument("--recalibrate-bn", type=int, default=0, metavar="N",
                    help="after transferring, re-estimate the node-bound BatchNorm "
                         "statistics from N windows of the TARGET's train split. No "
                         "gradient step is taken and no weight changes -- only "
                         "running_mean / running_var, which are exactly the tensors that "
                         "cannot cross between road networks. Separates 'the source's "
                         "knowledge does not apply' from 'the transferred weights had no "
                         "usable normalisation'. Report it as zero-shot + BN adaptation, "
                         "not as plain zero-shot")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--nhid", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--nheads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = ap.parse_args()

    adj_path = cli.adj or os.path.join(cli.data, "adj_mx_dijsk.pkl")
    needed = [(adj_path, "adjacency")]
    if not cli.no_transfer:
        needed.append((cli.source, "source checkpoint"))
    for p, name in needed:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{name} not found: {p}")

    # --- target shapes, from the target data itself ---
    data = util.load_dataset(cli.data, 1, 1, 1)
    x, y = data["x_test"], data["y_test"]
    S, T_in, N, F = x.shape
    T_out = y.shape[1]
    _, _, adj_list = util.load_adj(adj_path, "symnadj")
    adj_n = np.array(adj_list, dtype=np.float32)[0].shape[0]
    if adj_n != N:
        raise ValueError(f"adjacency is {adj_n} nodes but the data has {N} -- these must "
                         f"describe the same road network")

    title = ("A3-a  CONTROL: untrained architecture, nothing copied"
             if cli.no_transfer else
             f"A3-a  zero-shot transfer: {cli.source} -> {cli.data}")
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    print(f"target : {N} nodes, {F} features, {T_in} in -> {T_out} out")
    print(f"NO gradient step is taken here. The z-score statistics DO come from the "
          f"target's\n         train split -- the two cities are in different units, so "
          f"the source's are unusable.")

    with contextlib.redirect_stdout(io.StringIO()):    # model/stgat.py debug prints
        net = STGAT(cli.device.startswith("cuda"), N, F, T_in, T_out,
                    nheads=cli.nheads, nhid=cli.nhid, layers=cli.layers)

    dst = net.state_dict()
    if cli.no_transfer:
        same, bound, n_same, n_bound = [], [], 0, sum(v.numel() for v in dst.values())
        print(f"\ncontrol run: nothing copied. {len(dst)} tensors, {n_bound:,} params, "
              f"all at PyTorch init.\n  This is the floor. Whatever a real transfer "
              f"scores ABOVE this is what the shared\n  weights were worth; scoring the "
              f"same means nothing transferred.")
        src = None
    else:
        src = torch.load(cli.source, map_location="cpu")
        if isinstance(src, dict) and "state_dict" in src:
            src = src["state_dict"]

    same, bound, missing = classify(src, dst) if src is not None else ([], [], [])
    if src is not None and (missing or not same):
        raise SystemExit(
            f"error: architecture mismatch -- {len(missing)} of the target's tensors are "
            f"absent from the source and {len(same)} matched. Is {cli.source} an STGAT "
            f"checkpoint with the same nhid/layers/nheads?\n  first missing: "
            f"{missing[:3]}")

    if src is not None:
        n_same = sum(dst[k].numel() for k in same)
        n_bound = sum(dst[k].numel() for k in bound)
        print(f"\ntransferable : {len(same):>3}/{len(dst)} tensors  {n_same:>10,} params  "
              f"{n_same / (n_same + n_bound):6.1%}")
        print(f"node-bound   : {len(bound):>3}/{len(dst)} tensors  {n_bound:>10,} params  "
              f"{n_bound / (n_same + n_bound):6.1%}   -> --node-init {cli.node_init}")
        shapes = sorted({(tuple(src[k].shape), tuple(dst[k].shape)) for k in bound})
        for a, b in shapes:
            n = sum(1 for k in bound if tuple(dst[k].shape) == b)
            print(f"               {n:>3} x {str(a):>12} -> {str(b)}")
        for k in same:
            dst[k] = src[k].clone()
        fill_node_bound(src, dst, bound, cli.node_init)
        net.load_state_dict(dst)

    if cli.recalibrate_bn:
        recalibrate_bn(net, data, adj_path, cli)

    os.makedirs(cli.out_dir, exist_ok=True)
    out = os.path.join(cli.out_dir, cli.out_name)
    torch.save(net.state_dict(), out)
    meta = {"experiment": ("A3-a control: untrained, nothing copied" if cli.no_transfer
                           else "A3-a zero-shot transfer"),
            "source": None if cli.no_transfer else cli.source,
            "target_data": cli.data,
            "node_init": None if cli.no_transfer else cli.node_init,
            "n_vertex": N, "num_features": F, "n_his": T_in, "n_pred": T_out,
            "tensors_total": len(dst), "tensors_transferred": len(same),
            "tensors_node_bound": len(bound),
            "params_transferred": n_same, "params_node_bound": n_bound,
            "nhid": cli.nhid, "layers": cli.layers, "nheads": cli.nheads,
            "recalibrate_bn": cli.recalibrate_bn,
            "note": ("no gradient step taken on the target; the scaler is the target's"
                     + ("; BatchNorm statistics WERE re-estimated on the target's train "
                        "split -- report as zero-shot + BN adaptation"
                        if cli.recalibrate_bn else ""))}
    with open(os.path.splitext(out)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {out}  (+ .meta.json)")
    if cli.no_transfer:
        print(f"\nScore it as the floor:\n"
              f"    python evaluate_masked_taichung.py --checkpoint {out}")
        return
    print(f"\nThis file IS the proposal's zero-shot claim. Score it before reading any "
          f"fine-tuned\nnumber, because nothing else measures it:\n"
          f"    python evaluate_masked_taichung.py --checkpoint {out}\n"
          f"\nThen A3-b, which DOES train on the target -- every other flag must match "
          f"the\nfrom-scratch run or the comparison measures the flags instead of the "
          f"transfer:\n"
          f"    python train.py --cuda --data {cli.data} --adj_filename {adj_path} \\\n"
          f"                    --num_of_vertices {N} --params_dir {cli.out_dir} \\\n"
          f"                    --init-from {out} \\\n"
          f"                    --lr 3e-4 --epoch 500 --early_stop_maxtry 40")


if __name__ == "__main__":
    main()
