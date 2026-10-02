"""osu!mania 4K chart model: parse ranked .osu files, describe timing, serialise generated charts."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

KEYS = 4
LANE_X = (64, 192, 320, 448)


@dataclass(frozen=True)
class RedLine:
    time: float          # ms
    beat_ms: float       # ms per beat
    meter: int = 4

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_ms


@dataclass(frozen=True)
class Note:
    time: float          # ms
    lane: int            # 0..3
    end: float = 0.0     # ms; 0 for a tap

    @property
    def is_hold(self) -> bool:
        return self.end > self.time


@dataclass
class Chart:
    notes: List[Note]
    red_lines: List[RedLine]
    od: float = 8.0
    hp: float = 8.0
    version: str = ""
    title: str = ""
    artist: str = ""
    audio: str = ""
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def heads(self) -> List[float]:
        return sorted({n.time for n in self.notes})

    def duration_ms(self) -> float:
        if not self.notes:
            return 0.0
        return max(max(n.end, n.time) for n in self.notes) - self.notes[0].time


def _sections(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    cur: Optional[str] = None
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


def _kv(lines: Sequence[str]) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for line in lines:
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def parse_osu(text: str) -> Optional[Chart]:
    """Parse a mode-3, 4-key chart; anything else returns None."""
    sec = _sections(text.lstrip("﻿"))
    general, diff, meta = _kv(sec.get("General", [])), _kv(sec.get("Difficulty", [])), _kv(sec.get("Metadata", []))
    try:
        if int(float(general.get("Mode", "0"))) != 3 or abs(float(diff.get("CircleSize", "0")) - KEYS) > 0.01:
            return None
        od = float(diff.get("OverallDifficulty", "8"))
        hp = float(diff.get("HPDrainRate", "8"))
    except ValueError:
        return None
    reds: List[RedLine] = []
    for line in sec.get("TimingPoints", []):
        p = line.split(",")
        try:
            t, beat = float(p[0]), float(p[1])
            meter = int(float(p[2])) if len(p) > 2 else 4
            uninherited = len(p) <= 6 or p[6].strip() != "0"
        except (ValueError, IndexError):
            continue
        if uninherited and beat > 0 and math.isfinite(t) and math.isfinite(beat):
            reds.append(RedLine(t, beat, max(1, meter)))
    reds.sort(key=lambda r: r.time)
    notes: List[Note] = []
    for line in sec.get("HitObjects", []):
        p = line.split(",")
        try:
            x, t, typ = int(float(p[0])), float(p[2]), int(p[3])
        except (ValueError, IndexError):
            continue
        lane = min(KEYS - 1, max(0, int(x * KEYS // 512)))
        end = 0.0
        if typ & 128:
            try:
                end = float(p[5].split(":", 1)[0])
            except (ValueError, IndexError):
                end = 0.0
            if end <= t:
                end = 0.0
        notes.append(Note(t, lane, end))
    notes.sort(key=lambda n: (n.time, n.lane))
    if not reds or not notes:
        return None
    return Chart(notes, reds, od, hp, meta.get("Version", ""), meta.get("Title", ""), meta.get("Artist", ""),
                 general.get("AudioFilename", "").strip(), meta)


def load_osu(path: str | Path) -> Optional[Chart]:
    return parse_osu(Path(path).read_text(encoding="utf-8-sig", errors="replace"))


# --------------------------------------------------------------------------- timing helpers

def red_line_at(reds: Sequence[RedLine], t: float) -> RedLine:
    cur = reds[0]
    for r in reds:
        if r.time <= t + 1e-6:
            cur = r
        else:
            break
    return cur


def snap_of(reds: Sequence[RedLine], t: float, divisors: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16),
            tol_ms: float = 4.0) -> Optional[int]:
    """Smallest divisor d so that t lies on the 1/d grid of its red line, else None."""
    r = red_line_at(reds, t)
    beats = (t - r.time) / r.beat_ms
    for d in divisors:
        if abs(beats * d - round(beats * d)) * r.beat_ms / d <= tol_ms:
            return d
    return None


def main_bpm(reds: Sequence[RedLine], end_ms: float) -> float:
    """BPM that covers most of the song (what osu! shows as the map's BPM)."""
    spans: Dict[float, float] = {}
    for i, r in enumerate(reds):
        nxt = reds[i + 1].time if i + 1 < len(reds) else max(end_ms, r.time)
        spans[round(r.bpm, 3)] = spans.get(round(r.bpm, 3), 0.0) + max(0.0, nxt - r.time)
    return max(spans.items(), key=lambda kv: kv[1])[0]
