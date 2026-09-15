"""Minimal .osu parser for training data: timing points, hit objects (with end times) and settings.

Handles file formats v3..v14 well enough for ranked/loved osu!standard maps. Anything odd is
skipped rather than guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

OBJ_CIRCLE, OBJ_SLIDER, OBJ_SPINNER = 0, 1, 2


@dataclass
class RedLine:
    time: float
    beat_length: float
    meter: int


@dataclass
class ParsedBeatmap:
    mode: int
    slider_multiplier: float
    red_lines: List[RedLine]
    sv_points: List[Tuple[float, float]]          # (time, sv multiplier), sorted
    objects: np.ndarray                           # (N, 4) float: time, type, end_time, new_combo
    settings: Dict[str, float] = field(default_factory=dict)

    @property
    def n_objects(self) -> int:
        return int(len(self.objects))


def _sections(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1]
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return out


def _kv(lines: List[str]) -> Dict[str, str]:
    d = {}
    for l in lines:
        if ":" in l:
            k, v = l.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def parse_osu(text: str) -> ParsedBeatmap:
    sec = _sections(text)
    general = _kv(sec.get("General", []))
    diff = _kv(sec.get("Difficulty", []))
    mode = int(float(general.get("Mode", 0)))
    sm = float(diff.get("SliderMultiplier", 1.4) or 1.4)
    settings = {k: float(diff[k]) for k in ("HPDrainRate", "CircleSize", "OverallDifficulty", "ApproachRate") if k in diff}

    red: List[RedLine] = []
    sv: List[Tuple[float, float]] = []
    for l in sec.get("TimingPoints", []):
        p = l.split(",")
        if len(p) < 2:
            continue
        try:
            t, bl = float(p[0]), float(p[1])
        except ValueError:
            continue
        meter = int(float(p[2])) if len(p) > 2 and p[2].strip() else 4
        uninherited = (p[6].strip() == "1") if len(p) > 6 else bl > 0
        if uninherited and bl > 0:
            red.append(RedLine(t, bl, max(1, meter)))
        elif bl < 0:
            sv.append((t, max(0.05, min(10.0, -100.0 / bl))))
    red.sort(key=lambda r: r.time)
    sv.sort(key=lambda s: s[0])

    def beat_length_at(t: float) -> float:
        cur = red[0].beat_length if red else 500.0
        for r in red:
            if r.time <= t + 1e-6:
                cur = r.beat_length
            else:
                break
        return cur

    def sv_at(t: float) -> float:
        # the latest inherited point at or before t, but reset to 1.0 by a later red line
        last_red = -1e18
        for r in red:
            if r.time <= t + 1e-6:
                last_red = r.time
        cur = 1.0
        for st, mult in sv:
            if st <= t + 1e-6:
                if st >= last_red - 1e-6:
                    cur = mult
            else:
                break
        return cur

    objs: List[Tuple[float, int, float, int]] = []
    for l in sec.get("HitObjects", []):
        p = l.split(",")
        if len(p) < 4:
            continue
        try:
            t, typ = float(p[2]), int(float(p[3]))
        except ValueError:
            continue
        nc = 1 if typ & 4 else 0
        if typ & 8:                                  # spinner
            end = float(p[5]) if len(p) > 5 and re.match(r"^-?\d+(\.\d+)?$", p[5].strip()) else t
            objs.append((t, OBJ_SPINNER, max(end, t), nc))
        elif typ & 2:                                # slider
            if len(p) < 8:
                continue
            try:
                slides, length = int(float(p[6])), float(p[7])
            except ValueError:
                continue
            bl = beat_length_at(t)
            dur = length / (sm * 100.0 * sv_at(t)) * bl * max(1, slides)
            objs.append((t, OBJ_SLIDER, t + dur, nc))
        elif typ & 1:                                # circle
            objs.append((t, OBJ_CIRCLE, t, nc))
        # mania holds (128) are ignored: std only
    arr = np.array(objs, dtype=np.float64).reshape(-1, 4)
    if len(arr):
        arr = arr[np.argsort(arr[:, 0], kind="stable")]
    return ParsedBeatmap(mode=mode, slider_multiplier=sm, red_lines=red, sv_points=sv, objects=arr, settings=settings)
