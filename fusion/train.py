#!/usr/bin/env python3
"""Train the dual-path Gated Fusion model (計劃書 §4.3).

    python train.py --freeze none                 # the proposal: both paths learn
    python train.py --freeze both                 # gate only, backbones fixed
    python train.py --freeze none --extended-gate # gate also sees section + time-of-day

RUN fusion/verify.py FIRST. It checks that this dataloader reproduces the MAE the two
standalone models already recorded; if it does not, everything trained here is measured
against a different set of windows than 實驗記錄 and cannot be compared with it.

WHAT --freeze DECIDES
    both  The gate is measured in isolation, and 實驗記錄 §13.7's error correlation of
          0.934 is a hard ceiling -- two fixed backbones are wrong in the same places.
          Minutes to train.
    none  計劃書 §4.3 as written, and the only setting where the paths can DECORRELATE,
          because the gate finally gives them a reason to specialise. Also the only one
          that produces a single model emitting 15/30/60 min, since the fusion head
          replaces STGCN's single-step output block.

LOSS. The proposal specifies L2. The default here is masked MAE, because MAE is what
every number in 實驗記錄 is reported in and optimising a different quantity than the one
reported invites the two to diverge quietly. `--loss mse` restores the proposal's choice.
Either way the loss is MASKED: ~25% of the matrix is imputed and scoring on it is nearly
free for a model (實驗設計 §2.3). METR-LA's `null_val=0.0` trick does not transfer -- the
imputed cells hold real numbers, so the external mask is the only thing that works.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths as P
from data import FusionDataset
from model import DualPathModel


@contextlib.contextmanager
def quiet():
    """Swallow STGAT's leftover per-layer debug prints.

    model/stgat.py prints X_in/X_out for every TimeBlock on every forward -- 24 lines a
    batch here. At ~1,100 batches an epoch that is 2.6M lines over a 100-epoch run, and
    實驗記錄 §14.2 (12) already measured a 28.5 MB log from the same prints during the
    standalone training. Wrapped only around the forward so this script's own output
    still reaches the terminal.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def masked_loss(pred, y, m, kind):
    d = (pred - y)[m]
    if d.numel() == 0:
        return pred.sum() * 0.0
    if kind == "mae":
        return d.abs().mean()
    if kind == "mse":
        return (d ** 2).mean()
    return torch.nn.functional.smooth_l1_loss(d, torch.zeros_like(d))


@torch.no_grad()
def evaluate(model, loader, ds, device, horizons=(3, 6, 12), limit=0):
    """Masked MAE in km/h: per horizon and averaged over all twelve steps.

    `limit` caps the batches for timing runs. Without it, --limit-batches would shorten
    training but still validate over the whole split, and the wall-clock measured would
    be dominated by the part it was meant to skip.
    """
    model.eval()
    num = np.zeros(ds.n_pred)
    den = np.zeros(ds.n_pred)
    for nb, (x_cn, x_at, y, m, row0) in enumerate(loader):
        if limit and nb >= limit:
            break
        tod = torch.as_tensor(ds.tod[row0.numpy() - 1], device=device)
        with quiet():
            z = model(x_cn.to(device), x_at.to(device), tod)       # [B, N, 12]
        pred = ds.denorm_cn(z.permute(0, 2, 1))                    # [B, 12, N]
        d = (pred.cpu() - y).abs().numpy()
        mm = m.numpy()
        num += (d * mm).sum(axis=(0, 2))
        den += mm.sum(axis=(0, 2))
    model.train()
    per = num / np.maximum(den, 1)
    return {h: float(per[h - 1]) for h in horizons}, float(per.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", choices=["none", "both", "stgcn", "stgat"],
                    default="none")
    ap.add_argument("--stgcn-tod", action="store_true",
                    help="give the STGCN path a time-of-day channel. 計劃書 §4.3 assigns "
                         "it 尖峰時段/星期週期, which it cannot represent from speed "
                         "alone -- measured in 實驗記錄 §13.20 ③. Incompatible with "
                         "warm-starting that path (the first block's shape changes)")
    ap.add_argument("--single-path", choices=["stgcn", "stgat"], default=None,
                    help="control: bypass the gate and send ONE path to the head. "
                         "--single-path stgat answers 'what does the second path plus "
                         "the gate buy, given this training regime?' -- if it reaches "
                         "the dual-path score on its own, the answer is nothing. The "
                         "question is live because the learned gate settles at 0.78-0.83 "
                         "(the output is ~4/5 STGAT) while the best FIXED blend beats "
                         "STGAT by only 0.40% (實驗記錄 §13.7, §13.22 ⑥c)")
    ap.add_argument("--extended-gate", action="store_true")
    ap.add_argument("--head-hidden", type=int, default=0,
                    help="0 = a single FC, as the proposal writes it")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4, help="計劃書 §4.3")
    ap.add_argument("--decay", type=float, default=0.96, help="per-epoch, 計劃書 §4.3")
    ap.add_argument("--loss", choices=["mae", "mse", "huber"], default="mae")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--limit-batches", type=int, default=0, metavar="N",
                    help="stop each epoch after N batches. For timing a run before "
                         "committing to it -- the resulting model is not usable")
    ap.add_argument("--init-from-pretrained", action="store_true",
                    help="warm-start the backbones from the separately trained models. "
                         "STGCN's checkpoints are horizon-specific, so this imports a "
                         "bias toward whichever one is used -- off by default")
    ap.add_argument("--stgcn-ckpt", default=None)
    ap.add_argument("--stgat-ckpt",
                    default=os.path.join(P.STGAT_DIR, "experiment_taichung",
                                         "best_model.pth"))
    ap.add_argument("--out", default="checkpoints/fusion.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    cli = ap.parse_args()

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    n_cn = 2 if cli.stgcn_tod else 1
    tr = FusionDataset("train", stgcn_channels=n_cn)
    va = FusionDataset("val", stgcn_channels=n_cn)
    print("=== data ===")
    print(" ", tr.describe())
    print(" ", va.describe())
    print(f"  STGCN path input: {n_cn} channel(s)"
          + ("  [speed + time-of-day]" if n_cn == 2 else "  [speed only]"))
    if n_cn == 1 and "stgcn" not in {"both": ("stgcn",), "stgcn": ("stgcn",)}.get(
            cli.freeze, ()):
        print("  ⚠ the STGCN path is being trained from scratch WITHOUT time-of-day. "
              "計劃書 §4.3\n    assigns it 尖峰時段/星期週期, which speed alone cannot "
              "express: 實驗記錄 §13.20 ③\n    measured it losing to a plain historical "
              "average on the most routine cells and at\n    weekday peak. Consider "
              "--stgcn-tod.")

    gso, n_vertex, _ = P.stgcn_gso()
    if n_vertex != tr.n_vertex:
        raise SystemExit(f"ERROR  adjacency has {n_vertex} sections, vel.csv has "
                         f"{tr.n_vertex}")
    adj = P.stgat_adj()
    model = DualPathModel(gso.to(cli.device), adj.to(cli.device), n_vertex,
                          n_pred=tr.n_pred, freeze=cli.freeze,
                          extended_gate=cli.extended_gate,
                          head_hidden=cli.head_hidden, stgcn_channels=n_cn,
                          single_path=cli.single_path,
                          cuda=cli.device.startswith("cuda")).to(cli.device)
    if cli.single_path:
        print(f"  ⚠ CONTROL RUN: gate bypassed, only the {cli.single_path.upper()} path "
              f"reaches the head.\n    The other path still runs (no gradient) so the "
              f"RNG stream matches a dual-path run.\n    This is not a fusion model -- "
              f"do not report it as one, and do not --dump-all it.")
    frozen = model.frozen_paths()
    # Only warm-start what is frozen (a frozen random backbone is pure noise) unless the
    # caller explicitly asks to warm-start everything.
    want_cn = cli.stgcn_ckpt if (cli.init_from_pretrained or "stgcn" in frozen) else None
    want_at = cli.stgat_ckpt if (cli.init_from_pretrained or "stgat" in frozen) else None
    if want_cn or want_at:
        info = model.load_pretrained(want_cn, want_at, cli.device)
        print(f"  warm-started {info}")
    if "stgcn" in frozen and not want_cn:
        raise SystemExit(
            "error: --freeze includes the STGCN path but no --stgcn-ckpt was given, so "
            "it would be\n       FROZEN AT RANDOM INIT and contribute nothing but noise. "
            "Pass the checkpoint for\n       the horizon you care about, e.g. "
            "--stgcn-ckpt ../STGCN/STGCN_taichung_p12.pt")
    if "stgat" in frozen and not want_at:
        raise SystemExit("error: --freeze includes the STGAT path but no --stgat-ckpt "
                         "was given")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for p in trainable)
    n_all_p = sum(p.numel() for p in model.parameters())
    print(f"\n=== model ===")
    print(f"  freeze={cli.freeze} | gate={'extended' if cli.extended_gate else 'faithful'}"
          f" | head={'FC' if not cli.head_hidden else f'MLP({cli.head_hidden})'}")
    print(f"  {n_train_p:,} trainable of {n_all_p:,} parameters")
    print(f"  loss={cli.loss} (masked) | lr={cli.lr} decay={cli.decay}/epoch")

    opt = torch.optim.Adam(trainable, lr=cli.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cli.decay)
    tl = DataLoader(tr, batch_size=cli.batch, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=cli.batch, shuffle=False, num_workers=0)

    os.makedirs(os.path.dirname(os.path.abspath(cli.out)) or ".", exist_ok=True)
    best, best_epoch, since = float("inf"), -1, 0
    n_batches = (len(tr) + cli.batch - 1) // cli.batch
    print(f"\nTraining -> {cli.out}")
    print(f"  {n_batches:,} batches/epoch at batch {cli.batch}"
          + (f", limited to {cli.limit_batches}" if cli.limit_batches else "")
          + f" | validating on {len(va):,} windows")
    if cli.limit_batches:
        print("  ⚠ --limit-batches is for TIMING ONLY; the checkpoint it writes is not "
              "a trained model")
    print()
    for ep in range(1, cli.epochs + 1):
        t0 = time.time()
        model.train()
        run, nb = 0.0, 0
        for x_cn, x_at, y, m, row0 in tl:
            if cli.limit_batches and nb >= cli.limit_batches:
                break
            tod = torch.as_tensor(tr.tod[row0.numpy() - 1], device=cli.device)
            with quiet():
                z = model(x_cn.to(cli.device), x_at.to(cli.device), tod)
            pred = tr.denorm_cn(z.permute(0, 2, 1))
            loss = masked_loss(pred, y.to(cli.device), m.to(cli.device), cli.loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cli.clip)
            opt.step()
            run += float(loss)
            nb += 1
        sched.step()
        per, mean12 = evaluate(model, vl, va, cli.device, limit=cli.limit_batches)
        tag = ""
        if mean12 < best:
            best, best_epoch, since = mean12, ep, 0
            torch.save(model.state_dict(), cli.out)
            with open(os.path.splitext(cli.out)[0] + ".meta.json", "w",
                      encoding="utf-8") as f:
                json.dump({"epoch": ep, "val_mae_12step": mean12,
                           "val_mae": {str(k): v for k, v in per.items()},
                           "freeze": cli.freeze, "extended_gate": cli.extended_gate,
                           "head_hidden": cli.head_hidden, "loss": cli.loss,
                           "stgcn_channels": n_cn,
                           # Load-bearing: W1/W3 exist in the state dict even when a
                           # single path was trained, so evaluate.py rebuilding with the
                           # default learned gate would load cleanly and then score a
                           # gate that was never trained. Silent, and the number looks
                           # plausible. evaluate.py reads this key for that reason.
                           "single_path": cli.single_path,
                           # Nothing verifies these when a path is trained from scratch
                           # (there is no checkpoint whose shapes must match), so they
                           # are recorded here: "fusion vs STGCN alone" is only a fair
                           # comparison if the architecture is the same one.
                           "stgcn_arch": P.STGCNPath.ARCH,
                           "stgat_arch": {"nhid": 64, "nheads": 4, "layers": 4},
                           "lr": cli.lr, "decay": cli.decay, "batch": cli.batch,
                           "n_vertex": n_vertex, "n_pred": tr.n_pred,
                           "n_his": tr.n_his, "seed": cli.seed}, f, indent=2)
            tag = "  <- saved"
        else:
            since += 1
        print(f"epoch {ep:3d} | train {run / max(1, nb):7.4f} | val 15/30/60 "
              f"{per[3]:.4f} {per[6]:.4f} {per[12]:.4f} | 12-step {mean12:.4f} "
              f"| {time.time() - t0:5.1f}s{tag}")
        if since >= cli.patience:
            print(f"\nEarly stop: no improvement for {cli.patience} epochs.")
            break

    print(f"\nBest 12-step val MAE {best:.4f} at epoch {best_epoch} -> {cli.out}")
    print(f"Next: python evaluate.py --checkpoint {cli.out} --split test")


if __name__ == "__main__":
    main()
