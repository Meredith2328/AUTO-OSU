"""Small audio-conditioned four-lane model; unrelated to the standard checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .mania_data import LANES, N_CLASSES, PATCH
from .prepare_data import N_MELS


@dataclass(frozen=True)
class ManiaConfig:
    width: int = 96
    blocks: int = 3
    dropout: float = 0.1


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2 * dilation,
                                   dilation=dilation, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h = self.pointwise(self.activation(self.depthwise(h))).transpose(1, 2)
        return x + self.dropout(h)


class ManiaNet(nn.Module):
    """Predict empty/tap/LN head/body/tail per lane at every eighth-beat tick."""

    def __init__(self, cfg: ManiaConfig = ManiaConfig()):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.audio = nn.Sequential(nn.Flatten(-2), nn.Linear(PATCH * N_MELS, w), nn.GELU())
        self.phase = nn.Embedding(128, 16)
        self.fuse = nn.Sequential(nn.Linear(w + 16 + 1 + 2, w), nn.GELU())
        self.blocks = nn.Sequential(*(ResidualBlock(w, 2 ** i, cfg.dropout) for i in range(cfg.blocks)))
        self.head = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, LANES * N_CLASSES))
        self.chord_head = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 1 << LANES))
        self.duration_head = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, LANES * 33))

    def forward(self, patches: torch.Tensor, phase: torch.Tensor, beat_ms: torch.Tensor,
                condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio = self.audio(patches)
        p = self.phase(phase.clamp(0, 127))
        tempo = torch.log2(beat_ms.clamp_min(50) / 500).unsqueeze(-1)
        cond = condition[:, None, :].expand(-1, phase.shape[1], -1)
        h = self.fuse(torch.cat((audio, p, tempo, cond), dim=-1))
        h = self.blocks(h)
        return (self.head(h).reshape(*phase.shape, LANES, N_CLASSES), self.chord_head(h),
                self.duration_head(h).reshape(*phase.shape, LANES, 33))

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> tuple["ManiaNet", dict]:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("kind") != "autoosu-mania4k-v1":
            raise ValueError("Checkpoint is not a dedicated AUTO-OSU mania 4K model")
        model = cls(ManiaConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device).eval(), ckpt

    def checkpoint(self, **metadata) -> dict:
        return {"kind": "autoosu-mania4k-v1", "config": asdict(self.cfg),
                "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}, **metadata}
