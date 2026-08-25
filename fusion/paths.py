"""Import the STGCN and STGAT backbones into ONE process, and expose their hidden
features on a common shape.

計劃書 §4.3 describes a single model with two paths joined by a gate. That needs both
implementations live at the same time, which is not free: they are separate repositories
and BOTH ship a top-level package called `model`. 實驗記錄 §4.1 hit this collision before
and worked around it by decoupling inference into two processes -- an option the joint
model does not have.

The fix here is to load each `model/` directory under a name of our choosing, rather than
copying the files (a vendored copy silently drifts from the repo it was taken from):

    STGAT/model/stgat.py   uses RELATIVE imports (`from .layers import TimeBlock`), so it
                           only needs its parent package registered under some name.
    STGCN/model/models.py  uses an ABSOLUTE import (`from model import layers`), so the
                           name `model` has to exist for the duration of that import and
                           is restored immediately afterwards.

The leaf modules (layers/readout/discriminator) import nothing from their own repo, so
this is the whole of the problem.

--------------------------------------------------------------------------------
HIDDEN FEATURE SHAPES (measured, not assumed -- see fusion/README.md)

    STGCN   st_blocks(x)   [B, 64, Ko=4, N]     time not fully collapsed
            .output        [B, 1, 1, N]         ONE step; hence one model per horizon

    STGAT   blocks(x)      [B, N, T', C] -> reshape/permute -> [B, 64, N]
            .output        [B, 12, N]           twelve steps from one model

Both carry 64 channels, so the gate of eq. 3 operates on [B, N, 64] for either path with
no projection needed. STGCN's extra time axis (Ko=4) is flattened and projected down;
that projection is the ONLY learned adapter, and it is part of the fusion head rather
than of the path.
"""
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STGCN_DIR = os.path.join(ROOT, "STGCN")
STGAT_DIR = os.path.join(ROOT, "STGAT")

HIDDEN = 64          # both paths' channel count, measured
STGCN_KO = 4         # n_his - (Kt-1)*2*stblock_num = 12 - 2*2*2


def _load_package(alias, path):
    """Import a directory as a package under `alias` instead of its own name."""
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_module(alias, pkg_alias, path):
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_alias
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def stgcn_modules():
    """(models, layers) from STGCN/model, imported without claiming the name `model`."""
    pkg = _load_package("_stgcn_model", os.path.join(STGCN_DIR, "model"))
    layers = _load_module("_stgcn_model.layers", "_stgcn_model",
                          os.path.join(STGCN_DIR, "model", "layers.py"))
    # models.py does `from model import layers`. Lend it the name, then give it back:
    # leaving it in place would make STGAT's own `model` package unreachable.
    saved = sys.modules.get("model")
    sys.modules["model"] = pkg
    pkg.layers = layers
    try:
        models = _load_module("_stgcn_model.models", "_stgcn_model",
                              os.path.join(STGCN_DIR, "model", "models.py"))
    finally:
        if saved is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = saved
    return models, layers


def stgat_module():
    """STGAT/model/stgat.py, whose relative imports resolve against the alias."""
    _load_package("_stgat_model", os.path.join(STGAT_DIR, "model"))
    for name in ("layers", "readout", "discriminator"):
        _load_module(f"_stgat_model.{name}", "_stgat_model",
                     os.path.join(STGAT_DIR, "model", f"{name}.py"))
    return _load_module("_stgat_model.stgat", "_stgat_model",
                        os.path.join(STGAT_DIR, "model", "stgat.py"))


# --------------------------------------------------------------------------- #
# adjacency: the two paths want the same graph in two different forms
# --------------------------------------------------------------------------- #
def stgcn_gso(dataset="taichung"):
    """Chebyshev graph shift operator, exactly as STGCN/main.py builds it."""
    sys.path.insert(0, STGCN_DIR)
    from script import dataloader, utility          # noqa: E402
    cwd = os.getcwd()
    os.chdir(STGCN_DIR)                             # dataloader hard-codes './data'
    try:
        adj, n_declared = dataloader.load_adj(dataset)
    finally:
        os.chdir(cwd)
    gso = utility.calc_chebynet_gso(utility.calc_gso(adj, "sym_norm_lap"))
    return torch.from_numpy(gso.toarray().astype(np.float32)), adj.shape[0], n_declared


def stgat_adj(data_dir=None):
    """Symmetric-normalised adjacency from STGAT's pickle."""
    sys.path.insert(0, STGAT_DIR)
    import util                                      # noqa: E402
    path = os.path.join(data_dir or os.path.join(STGAT_DIR, "data", "taichung"),
                        "adj_mx_dijsk.pkl")
    _, _, adj_list = util.load_adj(path, "symnadj")
    return torch.from_numpy(np.array(adj_list, dtype=np.float32))[0]


# --------------------------------------------------------------------------- #
# the two paths, each exposing features() on a common [B, N, C]
# --------------------------------------------------------------------------- #
class STGCNPath(nn.Module):
    """STGCN ST-Conv blocks. Input [B, 1, T, N] (per-section z-score)."""

    # Architecture constants, mirroring STGCN/main.py's defaults via
    # STGCN/evaluate_masked.py's build_args(). They must match what the standalone model
    # was trained with, or "fusion vs STGCN alone" compares two different architectures.
    # Warm-starting enforces this (a mismatch fails to load); training from scratch does
    # NOT, so train.py records them in the sidecar.
    ARCH = dict(kt=3, ks=3, stblock_num=2, act_func="glu", droprate=0.5,
                graph_conv_type="cheb_graph_conv", gso_type="sym_norm_lap")

    def __init__(self, gso, n_vertex, n_his=12, in_channels=1, kt=3, ks=3,
                 stblock_num=2, act_func="glu", droprate=0.5):
        super().__init__()
        models, _ = stgcn_modules()
        import argparse
        args = argparse.Namespace(
            n_his=n_his, n_pred=1, Kt=kt, Ks=ks, stblock_num=stblock_num,
            act_func=act_func, graph_conv_type="cheb_graph_conv",
            gso_type="sym_norm_lap", enable_bias=True, droprate=droprate, gso=gso)
        ko = n_his - (kt - 1) * 2 * stblock_num
        # in_channels > 1 adds a time-of-day channel. 計劃書 §4.3 assigns this path the
        # "規則性" component -- "尖峰時段、星期週期" -- but the shipped implementation
        # feeds it speed alone, so it cannot see the hour or the weekday that
        # periodicity is defined by. Measured consequence (實驗記錄 §13.20 ③): STGCN
        # LOSES to a plain historical average in the most routine anomaly bucket and at
        # weekday peak. Supplying the channel is what §4.3 already asks for, not a
        # departure from it -- but it changes the first block's shape, so a 1-channel
        # checkpoint can no longer be loaded.
        blocks = [[in_channels]] + [[64, 16, 64]] * stblock_num
        blocks += [[128] if ko == 0 else [128, 128], [1]]
        self.net = models.STGCNChebGraphConv(args, blocks, n_vertex)
        self.in_channels = in_channels
        self.ko = ko
        self.out_dim = 64 * ko          # flattened before the head projects it down

    def load_pretrained(self, path, device="cpu"):
        """Load a trained STGCN checkpoint, keeping only the ST-blocks.

        The checkpoint also holds `output.*` (the single-step head), which the fusion
        model replaces. strict=False is therefore expected here -- but the ST-block keys
        must all match, so the return value is checked rather than ignored.
        """
        if self.in_channels != 1:
            raise RuntimeError(
                f"cannot warm-start a {self.in_channels}-channel STGCN path from "
                f"{os.path.basename(path)}: the shipped checkpoints take speed alone, so "
                f"the first block's shape differs. Train this path from scratch "
                f"(--freeze none or --freeze stgat) or drop --stgcn-tod.")
        sd = torch.load(path, map_location=device)
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        bad = [k for k in missing if k.startswith("st_blocks")]
        if bad:
            raise RuntimeError(f"{os.path.basename(path)} is missing ST-block weights "
                               f"{bad[:3]}... -- wrong architecture or wrong checkpoint")
        return len(unexpected)

    def features(self, x):
        """[B, C, T, N] -> [B, N, 64*Ko]."""
        h = self.net.st_blocks(x)                       # [B, 64, Ko, N]
        return h.permute(0, 3, 1, 2).flatten(2)         # [B, N, 64*Ko]


class STGATPath(nn.Module):
    """STGAT ST-Blocks. Input [B, N, T, F] (global z-score + time-in-day)."""

    def __init__(self, adj, n_vertex, n_feat=2, n_his=12, nhid=64, nheads=4, layers=4,
                 cuda=False):
        super().__init__()
        stgat = stgat_module()
        self.net = stgat.STGAT(cuda, n_vertex, n_feat, n_his, 12,
                               nheads=nheads, nhid=nhid, layers=layers)
        self.register_buffer("adj", adj)
        self.out_dim = HIDDEN

    def load_pretrained(self, path, device="cpu"):
        sd = torch.load(path, map_location=device)
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        bad = [k for k in missing if k.startswith("blocks")]
        if bad:
            raise RuntimeError(f"{os.path.basename(path)} is missing ST-block weights "
                               f"{bad[:3]}... -- wrong architecture or wrong checkpoint")
        return len(unexpected)

    def features(self, x):
        """[B, N, T, F] -> [B, N, 64].

        Replicates STGAT.forward up to `emb`, stopping before `.output`. Copied rather
        than hooked so that the fusion model has no hidden control flow: the four lines
        below are the whole of the backbone's tail.
        """
        out = x
        for i in range(self.net.layers):
            out = self.net.blocks[i](out, self.adj)
        emb = out.reshape((out.shape[0], out.shape[1], -1))     # [B, N, T'*C]
        return emb

    def predict(self, x):
        """Full backbone INCLUDING its own head -> [B, N, 12] km/h-space z-scores.

        Only used by verify.py, to check that this wrapper reproduces the MAE the
        standalone evaluator reported. Nothing in training calls it.
        """
        out = self.net(self.adj, x)
        if isinstance(out, tuple):
            out = out[0]
        return out[..., 0]                                       # [B, N, 12]
