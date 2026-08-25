"""One dataloader feeding both paths from the same rows.

The joint model of 計劃書 §4.3 needs STGCN and STGAT to see the SAME window at the same
moment. Today they do not: the two repositories window, split and normalise the data
differently, which is why 實驗記錄 §13.11 ② reports 7,527 windows for one and 7,539 for
the other. This module reconciles them.

WHAT IS ACTUALLY THE SAME
    Both use a 12-step input and index their target the same way. STGCN's
    `data_transform` gives sample i the input rows i..i+11 and the target row
    i+11+n_pred; STGAT's offsets (x: -11..0, y: 1..12) give the window at t the input
    rows t-11..t and horizon p at row t+p. Substituting g = i = t-11, both read
    "input g..g+11, horizon p targets g+11+p". The window COUNTS differ only because
    each repo trims the split edges its own way.

WHAT IS DIFFERENT, AND HAS TO BE KEPT DIFFERENT
    Normalisation. STGCN z-scores PER SECTION (sklearn StandardScaler on the train rows);
    STGAT z-scores with ONE GLOBAL SCALAR over the train windows. These are not
    interchangeable -- feeding a trained STGCN globally-scaled input produces nonsense
    without erroring. Each path therefore receives its own tensor, built from the same
    rows.

    The split UNIT also differs: STGCN splits rows then windows within each split, while
    TDX_Data/convert_to_stgat_dataset.py splits windows. Rows are used here, because that
    is what STGCN/evaluate_masked.py and integration/ha_baseline.py score on, and those
    are the tools the reported numbers come from.

MASKING
    ~25% of the matrix is imputed and the loss must not be scored on it (實驗設計 §2.3).
    METR-LA's `null_val=0.0` mechanism does NOT transfer -- the imputed cells hold real
    numbers, so there is no sentinel to test for. Every window therefore carries its own
    target mask, and the training loss is masked with it.
"""
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DIR = os.path.join(ROOT, "Map")

N_HIS = 12
N_PRED = 12                  # the head emits all twelve steps, as 計劃書 §4.3 specifies
VAL_AND_TEST_RATE = 0.15     # mirrors STGCN/main.py and TDX_Data/convert_to_stgat_dataset.py
TZ_OFFSET_HOURS = 8          # DataCollectTime is UTC; time-of-day must be Taiwan local
STGAT_TRAIN_WINDOWS = None   # filled in by _stgat_scaler; see there


def split_sizes(n_rows, rate=VAL_AND_TEST_RATE):
    len_val = int(math.floor(n_rows * rate))
    len_test = int(math.floor(n_rows * rate))
    return int(n_rows - len_val - len_test), len_val, len_test


def _stgat_scaler(vel, n_rows):
    """Reproduce STGAT's single global mean/std WITHOUT materialising its .npz.

    `util.load_dataset` fits on `x_train[..., 0]`, i.e. the raw speeds at the INPUT rows
    of every train window -- not on the raw rows. Rows near the edges belong to fewer
    windows, so the two are not the same average. train.npz is ~1.7 GB uncompressed, so
    the weights are derived instead: window g covers rows g..g+11, and
    TDX_Data/convert_to_stgat_dataset.py assigns windows 0..n_train-1 to train.

    If this were wrong the error would not be subtle -- verify.py would report an STGAT
    MAE nowhere near the 3.63 the standalone evaluator gives.
    """
    n_windows = n_rows - N_HIS - N_PRED + 1
    n_test = round(n_windows * VAL_AND_TEST_RATE)
    n_val = round(n_windows * VAL_AND_TEST_RATE)
    n_train = n_windows - n_test - n_val
    last_g = n_train - 1
    r = np.arange(n_rows)
    mult = np.minimum(last_g, r) - np.maximum(0, r - (N_HIS - 1)) + 1
    mult = np.clip(mult, 0, None).astype(np.float64)
    w = np.repeat(mult[:, None], vel.shape[1], axis=1)
    tot = w.sum()
    mean = float((w * vel).sum() / tot)
    var = float((w * (vel - mean) ** 2).sum() / tot)      # population, as numpy .std()
    return mean, math.sqrt(var), n_train


class FusionDataset(Dataset):
    """Windows of one split, in both paths' input formats.

    Item g yields
        x_cn [1, 12, N]    per-section z-score      -> STGCNPath.features
        x_at [N, 12, 2]    global z-score + tod     -> STGATPath.features
        y    [12, N]       raw km/h, horizons 1..12
        m    [12, N]       True where y is a real observation
        row0 int           absolute row of horizon 1 (== g + 12)
    """

    def __init__(self, split="train", map_dir=MAP_DIR, n_his=N_HIS, n_pred=N_PRED,
                 stgcn_channels=1):
        vel = pd.read_csv(os.path.join(map_dir, "taichung_vel.csv"),
                          encoding="utf-8-sig").to_numpy(np.float64)
        mask = np.load(os.path.join(map_dir, "taichung_mask.npy"))
        stamps = pd.read_csv(os.path.join(map_dir, "taichung_timestamps.csv"),
                             encoding="utf-8-sig").iloc[:, 0]
        if vel.shape != mask.shape:
            raise ValueError(f"vel {vel.shape} and mask {mask.shape} disagree -- both "
                             f"come from build_speed.py, so re-run it")
        if len(stamps) != len(vel):
            raise ValueError(f"{len(stamps)} timestamps for {len(vel)} rows")

        self.n_rows, self.n_vertex = vel.shape
        self.n_his, self.n_pred, self.split = n_his, n_pred, split
        if stgcn_channels not in (1, 2):
            raise ValueError(f"stgcn_channels must be 1 or 2, got {stgcn_channels}")
        self.stgcn_channels = stgcn_channels
        len_train, len_val, len_test = split_sizes(self.n_rows)

        # --- the two normalisations, both fitted on TRAIN only ---
        tr = vel[:len_train]
        self.cn_mean = tr.mean(axis=0)                       # per section
        self.cn_std = tr.std(axis=0)
        self.cn_std[self.cn_std == 0] = 1.0
        self.at_mean, self.at_std, _ = _stgat_scaler(vel, self.n_rows)

        self.vel = vel.astype(np.float32)
        self.mask = mask
        self.z_cn = ((vel - self.cn_mean) / self.cn_std).astype(np.float32)
        self.z_at = ((vel - self.at_mean) / self.at_std).astype(np.float32)

        t = pd.to_datetime(stamps, utc=True) + pd.Timedelta(hours=TZ_OFFSET_HOURS)
        self.tod = ((t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second) / 86400.0
                    ).to_numpy(np.float32)

        # --- window range for this split ---
        offset = {"train": 0, "val": len_train, "test": len_train + len_val}[split]
        length = {"train": len_train, "val": len_val,
                  "test": self.n_rows - len_train - len_val}[split]
        # `+ 1` on purpose. STGCN's data_transform uses `num = len - n_his - n_pred`,
        # whose last sample targets row len-2: the final row of every split is never
        # predicted. That is an off-by-one upstream, not a convention, so it is not
        # copied -- at the cost of one extra window per split (1 in 7,519 on test,
        # under 0.02% of the cells, far inside verify.py's tolerance).
        n_windows = length - n_his - n_pred + 1
        if n_windows <= 0:
            raise ValueError(f"{split} split is too short for {n_his}+{n_pred} steps")
        self.offset, self.length = offset, length
        self.windows = offset + np.arange(n_windows)         # global window index g

    def __len__(self):
        return len(self.windows)

    def target_rows(self, horizon=None):
        """Absolute rows this split predicts. `horizon` is 1-based; None = all twelve."""
        g = self.windows[:, None]
        h = np.arange(1, self.n_pred + 1)[None, :] if horizon is None \
            else np.array([[horizon]])
        return (g + self.n_his - 1 + h).squeeze()

    def __getitem__(self, i):
        g = int(self.windows[i])
        a, b = g, g + self.n_his                              # input rows
        c, d = g + self.n_his, g + self.n_his + self.n_pred    # target rows
        tod = np.repeat(self.tod[a:b, None], self.n_vertex, axis=1)   # [T, N]
        # STGCN takes [C, T, N]; the second channel, when asked for, is the same
        # time-of-day STGAT already receives, so the two paths differ in architecture
        # rather than in what they are allowed to know.
        x_cn = (self.z_cn[a:b][None, :, :] if self.stgcn_channels == 1
                else np.stack([self.z_cn[a:b], tod], axis=0))        # [C, T, N]
        x_at = np.stack([self.z_at[a:b], tod], axis=-1).transpose(1, 0, 2)   # [N, T, F]
        return (torch.from_numpy(np.ascontiguousarray(x_cn)),
                torch.from_numpy(np.ascontiguousarray(x_at)),
                torch.from_numpy(self.vel[c:d]),
                torch.from_numpy(self.mask[c:d]),
                g + self.n_his)

    # --- helpers shared with train/evaluate -------------------------------------
    def denorm_cn(self, z):
        """[..., N] per-section z-score -> km/h."""
        mean = torch.as_tensor(self.cn_mean, dtype=z.dtype, device=z.device)
        std = torch.as_tensor(self.cn_std, dtype=z.dtype, device=z.device)
        return z * std + mean

    def denorm_at(self, z):
        return z * self.at_std + self.at_mean

    def describe(self):
        return (f"{self.split}: {len(self)} windows, target rows "
                f"{self.target_rows(1).min()}..{self.target_rows(self.n_pred).max()}, "
                f"{self.n_vertex} sections | real observations "
                f"{self.mask[self.target_rows().reshape(-1)].mean():.1%}")


def masked_mae(pred_kmh, y_kmh, m):
    """MAE in km/h over real observations only. Returns 0 if a batch has none."""
    d = (pred_kmh - y_kmh).abs()[m]
    return d.mean() if d.numel() else pred_kmh.sum() * 0.0
