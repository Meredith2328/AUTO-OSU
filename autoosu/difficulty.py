"""Difficulty presets. All rhythm/placement knobs live here so tuning is one file."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DifficultyPreset:
    name: str
    hp: float
    cs: float
    od: float
    ar: float
    slider_multiplier: float
    spacing: float              # distance-snap multiplier
    max_divisor: int            # finest grid allowed: 1 = 1/1, 2 = 1/2, 4 = 1/4
    strength_threshold: float   # onsets weaker than this (0..1) are dropped
    max_nps: float              # note density cap (objects per second, 2 s window)
    min_gap_beats: float        # minimum gap between consecutive object starts
    min_gap_ms: float           # ...and in milliseconds, so fast songs use a coarser grid
    slider_bias: float          # how eagerly to turn notes into sliders (0..1)
    min_slider_beats: float     # shortest slider allowed
    stream_max_len: int         # max consecutive 1/4 notes (0 = no streams)
    jump_prob: float            # chance a strong beat becomes a sharp-angle jump
    min_spinner_ms: int
    combo_min: int = 3          # do not start a new combo before this many objects
    kiai_sv: float = 1.0        # slider velocity multiplier inside kiai sections
    star: float = 3.5           # star rating the ML models are conditioned on for this difficulty


PRESETS: Dict[str, DifficultyPreset] = {
    "Easy": DifficultyPreset(
        name="Easy", hp=2, cs=2.5, od=2, ar=3,
        slider_multiplier=0.8, spacing=0.85, max_divisor=1,
        strength_threshold=0.50, max_nps=1.6, min_gap_beats=1.0, min_gap_ms=280,
        slider_bias=0.6, min_slider_beats=1.0, stream_max_len=0, jump_prob=0.0,
        min_spinner_ms=3000, combo_min=4, kiai_sv=1.0, star=2.0,
    ),
    "Normal": DifficultyPreset(
        name="Normal", hp=4, cs=3.5, od=4, ar=5,
        slider_multiplier=1.0, spacing=1.0, max_divisor=2,
        strength_threshold=0.38, max_nps=2.6, min_gap_beats=0.5, min_gap_ms=230,
        slider_bias=0.5, min_slider_beats=0.5, stream_max_len=0, jump_prob=0.05,
        min_spinner_ms=2000, kiai_sv=1.1, star=3.2,
    ),
    "Hard": DifficultyPreset(
        name="Hard", hp=5, cs=4, od=6, ar=8,
        slider_multiplier=1.4, spacing=1.15, max_divisor=4,
        strength_threshold=0.26, max_nps=4.5, min_gap_beats=0.25, min_gap_ms=130,
        slider_bias=0.4, min_slider_beats=0.5, stream_max_len=6, jump_prob=0.25,
        min_spinner_ms=1500, kiai_sv=1.15, star=4.5,
    ),
    "Insane": DifficultyPreset(
        name="Insane", hp=6, cs=4, od=8, ar=9.3,
        slider_multiplier=1.7, spacing=1.3, max_divisor=4,
        strength_threshold=0.14, max_nps=7.0, min_gap_beats=0.25, min_gap_ms=70,
        slider_bias=0.35, min_slider_beats=0.5, stream_max_len=16, jump_prob=0.45,
        min_spinner_ms=1000, kiai_sv=1.2, star=5.5,
    ),
}


def get_preset(name: str) -> DifficultyPreset:
    key = name.strip().lower()
    for k, v in PRESETS.items():
        if k.lower() == key:
            return v
    raise KeyError(f"unknown difficulty {name!r}, choose from {', '.join(PRESETS)}")
