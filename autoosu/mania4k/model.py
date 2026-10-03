"""Note model: which beat-grid ticks carry notes, how many, and which start long notes.

Audio side: a dilated temporal conv stack over the log-mel frames, gathered at each tick, plus the
2.9 ms onset-envelope taps around the tick (the precise "is there an attack exactly here" signal).
Sequence side: a non-causal dilated residual TCN over the ticks (receptive field +-128 ticks, about
ten beats each way), so repetition and phrase structure inform each decision. Every block is
conditioned on the target star rating (FiLM).
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import DIV_CLASSES, ENV_TAPS, N_MELS, POSITIONS

ENV_DIM = 4 * len(ENV_TAPS)
LN_BINS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)     # LN length classes, beats


def ln_class(beats: torch.Tensor) -> torch.Tensor:
    edges = torch.tensor([(a + b) / 2 for a, b in zip(LN_BINS[:-1], LN_BINS[1:])], device=beats.device)
    return torch.bucketize(beats, edges)


class StarFourier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("freq", torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0]))
        self.proj = nn.Sequential(nn.Linear(11, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, stars: torch.Tensor) -> torch.Tensor:          # (B,)
        s = stars[:, None] / 4.0
        ang = 2 * math.pi * s * self.freq
        return self.proj(torch.cat([s, torch.sin(ang), torch.cos(ang)], dim=1))


class Block(nn.Module):
    def __init__(self, ch: int, dilation: int, cond_dim: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, 1)
        self.film = nn.Linear(cond_dim, 2 * ch)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g, b = self.film(cond).chunk(2, dim=-1)
        h = F.gelu(self.c1(x)) * (1 + g[:, :, None]) + b[:, :, None]
        return x + self.drop(self.c2(h))


class NoteNet(nn.Module):
    def __init__(self, hidden: int = 128, audio_dim: int = 96, dilations=(1, 2, 4, 8, 16, 32, 64, 1)):
        super().__init__()
        c = audio_dim
        self.audio = nn.Sequential(
            nn.Conv1d(N_MELS, c, 5, padding=2), nn.GELU(),
            nn.Conv1d(c, c, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(c, c, 3, padding=4, dilation=4), nn.GELU(),
        )
        self.env = nn.Sequential(nn.Linear(ENV_DIM, 96), nn.GELU(), nn.Linear(96, 64))
        self.pos = nn.Embedding(len(POSITIONS), 16)
        self.div = nn.Embedding(len(DIV_CLASSES), 8)
        self.bim = nn.Embedding(8, 8)
        d_in = c + 64 + 16 + 8 + 8 + 2
        self.inp = nn.Linear(d_in, hidden)
        self.cond = StarFourier(64)
        self.blocks = nn.ModuleList([Block(hidden, d, 64) for d in dilations])
        self.head = nn.Sequential(nn.Linear(hidden + 64, hidden), nn.GELU())
        self.count = nn.Linear(hidden, 5)
        self.lnhead = nn.Linear(hidden, 1)
        self.lnlen = nn.Linear(hidden, len(LN_BINS))
        self.config = {"hidden": hidden, "audio_dim": audio_dim, "dilations": tuple(dilations)}

    def forward(self, mel: torch.Tensor, frame_idx: torch.Tensor, env: torch.Tensor, pos: torch.Tensor,
                div: torch.Tensor, bim: torch.Tensor, loud: torch.Tensor, beat_ms: torch.Tensor,
                stars: torch.Tensor) -> Dict[str, torch.Tensor]:
        """mel (B, F, N_MELS) float 0..1; frame_idx (B, T) into F; per-tick tensors (B, T, ...)."""
        a = self.audio(mel.transpose(1, 2))                               # (B, C, F)
        idx = frame_idx.clamp(0, a.shape[2] - 1)
        a = torch.gather(a, 2, idx[:, None, :].expand(-1, a.shape[1], -1)).transpose(1, 2)  # (B, T, C)
        x = torch.cat([a, self.env(env), self.pos(pos), self.div(div), self.bim(bim.clamp(0, 7)),
                       loud[..., None], (beat_ms[..., None] - 400.0) / 200.0], dim=-1)
        cond = self.cond(stars)
        h = self.inp(x).transpose(1, 2)                                   # (B, H, T)
        for blk in self.blocks:
            h = blk(h, cond)
        h = h.transpose(1, 2)
        h = self.head(torch.cat([h, cond[:, None, :].expand(-1, h.shape[1], -1)], dim=-1))
        return {"count": self.count(h), "lnhead": self.lnhead(h).squeeze(-1), "lnlen": self.lnlen(h)}

    @staticmethod
    def from_checkpoint(path: str, device: str = "cpu") -> "NoteNet":
        ck = torch.load(path, map_location=device, weights_only=False)
        net = NoteNet(**ck.get("config", {}))
        net.load_state_dict(ck["model"])
        return net.to(device).eval()
