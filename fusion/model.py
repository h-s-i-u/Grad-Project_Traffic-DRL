"""The dual-path model of 計劃書 §4.3: STGCN path + STGAT path joined by a gate.

    gate      = sigma( W1 . (s + t) )
    Fusion    = tanh( W2 . t . gate + (I - gate) . W3 . s )
    output    = FC(Fusion)            -> 15 / 30 / 60 min

s and t are the two paths' HIDDEN features, which is what the proposal specifies -- the
formula sits between the backbones and the output head, not after two finished forecasts.
Both paths carry 64 channels (measured, see paths.py), so the gate needs no projection;
STGCN's extra time axis (Ko=4) is flattened and projected down by `proj_cn`, the only
adapter in the model.

    OUTPUT SPACE. The head emits per-section z-scores and `data.denorm_cn` converts to
    km/h. Predicting a z-score rather than raw km/h only sets the initialisation scale --
    the loss is computed in km/h either way, so what is optimised is what is reported.

    WHY tanh IS SAFE HERE. Applied to a finished forecast, tanh would clip every speed
    into [-1, 1]; that is what makes prediction-level fusion awkward. Applied to hidden
    features with an FC after it -- the proposal's own arrangement -- it is an ordinary
    activation and the FC restores the range.

TWO GATE VARIANTS
    faithful  gate = sigma(W1(s + t)), exactly as written.
    extended  the gate additionally sees [s - t, a per-section embedding, time-of-day].

The second is not decoration. 實驗記錄 §13.7 measured that the MAE-optimal and
RMSE-optimal ensemble weights sit at OPPOSITE ends of the spectrum -- STGCN makes fewer
large errors, STGAT is better on average -- which says which path to trust probably
varies by section and by time of day. The faithful gate can only see the SUM of the two
feature vectors, so it cannot express "this section's STGCN branch is usually the
reliable one". Report the faithful variant; the extended one measures what that
restriction costs.
"""
import torch
import torch.nn as nn

from paths import STGCNPath, STGATPath


class GatedFusion(nn.Module):
    """eq. 3, plus the FC head. Operates on [B, N, C] from each path."""

    def __init__(self, stgcn_dim, stgat_dim, n_vertex, hidden=64, n_pred=12,
                 extended_gate=False, node_emb=8, head_hidden=0):
        super().__init__()
        self.proj_cn = nn.Linear(stgcn_dim, hidden)
        self.proj_at = (nn.Identity() if stgat_dim == hidden
                        else nn.Linear(stgat_dim, hidden))
        self.extended = extended_gate
        gate_in = hidden if not extended_gate else hidden * 2 + node_emb + 1
        self.W1 = nn.Linear(gate_in, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.W3 = nn.Linear(hidden, hidden)
        self.node_emb = nn.Embedding(n_vertex, node_emb) if extended_gate else None
        self.head = (nn.Linear(hidden, n_pred) if not head_hidden else
                     nn.Sequential(nn.Linear(hidden, head_hidden), nn.ReLU(),
                                   nn.Linear(head_hidden, n_pred)))

    def forward(self, s, t, tod=None):
        """s [B,N,Ds], t [B,N,Dt] -> [B, N, n_pred] in per-section z-score."""
        s = self.proj_cn(s)
        t = self.proj_at(t)
        if self.extended:
            b, n, _ = s.shape
            emb = self.node_emb.weight.unsqueeze(0).expand(b, -1, -1)
            tod = (torch.zeros(b, n, 1, device=s.device) if tod is None
                   else tod.view(b, 1, 1).expand(b, n, 1))
            g_in = torch.cat([s + t, s - t, emb, tod], dim=-1)
        else:
            g_in = s + t
        gate = torch.sigmoid(self.W1(g_in))
        h = torch.tanh(self.W2(t * gate) + self.W3(s * (1.0 - gate)))
        return self.head(h)

    def gate_value(self, s, t, tod=None):
        """Mean gate opening per (batch, section). For the anomaly analysis.

        A gate that never moves means the model settled on a fixed blend and the whole
        exercise reduces to the constant weight 實驗記錄 §13.7 already measured.
        """
        with torch.no_grad():
            s2, t2 = self.proj_cn(s), self.proj_at(t)
            if self.extended:
                b, n, _ = s2.shape
                emb = self.node_emb.weight.unsqueeze(0).expand(b, -1, -1)
                tod = (torch.zeros(b, n, 1, device=s2.device) if tod is None
                       else tod.view(b, 1, 1).expand(b, n, 1))
                g_in = torch.cat([s2 + t2, s2 - t2, emb, tod], dim=-1)
            else:
                g_in = s2 + t2
            return torch.sigmoid(self.W1(g_in)).mean(-1)


class DualPathModel(nn.Module):
    """Both backbones plus the gate, as one module.

    `freeze` decides which experiment this is:
        "both"   -- the paths are fixed and only the gate/head learn. 實驗記錄 §13.7's
                    0.934 error correlation is then a hard ceiling: two frozen backbones
                    are wrong in the same places and no gate can invent independent
                    information. Cheap, and it measures the gate in isolation.
        "stgat"  -- STGAT fixed (it is the stronger model and the expensive one to
                    train), STGCN retrained. The correlation can still fall, because one
                    side is free to cover what the other misses.
        "stgcn"  -- the mirror image; mostly useful as a control.
        "none"   -- 計劃書 §4.3's architecture with nothing held back. Both paths can
                    specialise, which is what the gate is supposed to induce.

    Note that ALL of them emit 15/30/60 min from one forward: the horizon lives in the
    fusion head, which is new in every case, not in the backbones. What a frozen STGCN
    costs is different -- its features were optimised under a SINGLE-horizon objective
    (the shipped checkpoints are p3/p6/p12), so the other two horizons are served by
    features never asked to support them.
    """

    FREEZE = ("none", "both", "stgcn", "stgat")

    def __init__(self, gso, adj, n_vertex, n_pred=12, hidden=64, freeze="none",
                 extended_gate=False, head_hidden=0, cuda=False, n_his=12,
                 stgcn_channels=1):
        super().__init__()
        if freeze not in self.FREEZE:
            raise ValueError(f"freeze must be one of {self.FREEZE}, got {freeze!r}")
        self.stgcn = STGCNPath(gso, n_vertex, n_his=n_his, in_channels=stgcn_channels)
        self.stgat = STGATPath(adj, n_vertex, n_his=n_his, cuda=cuda)
        self.fusion = GatedFusion(self.stgcn.out_dim, self.stgat.out_dim, n_vertex,
                                  hidden=hidden, n_pred=n_pred,
                                  extended_gate=extended_gate, head_hidden=head_hidden)
        self.freeze = freeze
        for name in self.frozen_paths():
            for p in getattr(self, name).parameters():
                p.requires_grad_(False)

    def frozen_paths(self):
        return {"none": (), "both": ("stgcn", "stgat"),
                "stgcn": ("stgcn",), "stgat": ("stgat",)}[self.freeze]

    def train(self, mode=True):
        """Keep frozen backbones in eval mode.

        Both paths contain BatchNorm. Left in train mode they would keep updating their
        running statistics from our batches even with requires_grad=False -- the weights
        would be frozen while the normalisation quietly drifted, and nothing would raise.
        """
        super().train(mode)
        for name in self.frozen_paths():
            getattr(self, name).eval()
        return self

    def paths(self, x_cn, x_at):
        frozen = self.frozen_paths()

        def run(name, mod, x):
            if name in frozen:
                with torch.no_grad():
                    return mod.features(x)
            return mod.features(x)

        return run("stgcn", self.stgcn, x_cn), run("stgat", self.stgat, x_at)

    def forward(self, x_cn, x_at, tod=None):
        s, t = self.paths(x_cn, x_at)
        return self.fusion(s, t, tod)                # [B, N, n_pred], per-section z

    def load_pretrained(self, stgcn_ckpt=None, stgat_ckpt=None, device="cpu"):
        """Warm-start the backbones from the separately trained models.

        Optional, and off by default for joint training: STGCN's checkpoints are
        HORIZON-SPECIFIC (one each for p3/p6/p12) because its output block emits a single
        step, so warm-starting a twelve-step model from any one of them imports a bias
        toward that horizon. Useful for the frozen experiment, questionable for the
        joint one -- which is why the caller has to ask for it explicitly.
        """
        out = {}
        if stgcn_ckpt:
            out["stgcn_unused_keys"] = self.stgcn.load_pretrained(stgcn_ckpt, device)
        if stgat_ckpt:
            out["stgat_unused_keys"] = self.stgat.load_pretrained(stgat_ckpt, device)
        return out
