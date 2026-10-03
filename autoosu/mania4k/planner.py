"""Section planner: what each part of the song should feel like, learned from ranked charts.

Human 4K charts first commit to an overall style (archetype: 切 stream-, 叠 jack-, LN-dominant or
hybrid), then give each musical section one main pattern type and a density that follows the
section's energy (rest sections ~45-60 % sparser than the chart, climaxes slightly denser and with
the harder subtypes: jumpstream over stream, more chordjack, more LN). The type usually changes
when the music changes section. All of these statistics are fitted per archetype by
``scripts/mania4k_fit_plan.py`` into ``weights/plan.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .chart import RedLine
from .features import MEL_HOP
from .onsets import SR, attack_times
from .sections import energy_levels, section_audio_features
from .structure import ARCHETYPES, TYPES, measure_windows, measure_features, novelty, section_bounds

PLAN_FILE = Path(__file__).resolve().parent / "weights" / "plan.json"
# share of long sections that mix in a second type; human in-section purity is 0.81 for LN charts
# and 0.62-0.67 for the other archetypes
SECONDARY_RATE = {"切": 0.5, "叠": 0.5, "LN": 0.15, "hybrid": 0.55}
STYLE_NAMES = {"auto": None, "stream": "切", "jack": "叠", "ln": "LN", "hybrid": "hybrid",
               "切": "切", "叠": "叠", "LN": "LN"}


@lru_cache(maxsize=1)
def load_plan() -> dict:
    return json.loads(PLAN_FILE.read_text(encoding="utf-8"))


@dataclass
class Section:
    start_ms: float
    end_ms: float
    measures: int
    energy: float
    level: int                 # 0 rest, 1 low, 2 mid, 3 climax
    type: int = 1              # index into structure.TYPES
    density: float = 0.0       # log density relative to the chart


@dataclass
class Plan:
    archetype: int
    sections: List[Section]

    @property
    def archetype_name(self) -> str:
        return ARCHETYPES[self.archetype]

    def section_of(self, times: np.ndarray) -> np.ndarray:
        starts = np.array([s.start_ms for s in self.sections])
        return np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(self.sections) - 1)

    def summary(self) -> str:
        return " | ".join(f"{s.start_ms / 1000:.0f}s {TYPES[s.type]}{'*' * s.level}" for s in self.sections)


def music_sections(mel: np.ndarray, env, reds: Sequence[RedLine], start_ms: float, end_ms: float) -> List[Section]:
    wins = measure_windows(None, list(reds), start_ms, end_ms)
    if len(wins) < 4:
        return [Section(start_ms, end_ms, max(1, len(wins)), 0.0, 2)]
    bounds = section_bounds(novelty(measure_features(mel, SR / MEL_HOP, wins)))
    feats = section_audio_features(mel, env, attack_times(env), wins, bounds)
    energy = np.nan_to_num(feats[:, 0])
    levels = energy_levels(energy)
    out = []
    for k, (s, e) in enumerate(zip(bounds, list(bounds[1:]) + [len(wins)])):
        out.append(Section(wins[s][0], wins[e - 1][1], e - s, float(energy[k]), int(levels[k])))
    return out


def choose_archetype(stars: float, style: str = "auto", ln_propensity: float = 0.0) -> int:
    """Requested style, or the most likely archetype for the star rating, nudged towards LN when the
    note model hears many sustained sounds. Below 2 stars jack-heavy charts are not used."""
    name = STYLE_NAMES.get(style, None)
    if name:
        a = ARCHETYPES.index(name)
    else:
        plan = load_plan()
        bands = plan["star_bands"]
        b = min(int(np.searchsorted(bands, stars, side="right") - 1), len(plan["arch_prior"]) - 1)
        p = np.array(plan["arch_prior"][b], float)
        p[ARCHETYPES.index("LN")] *= 0.5 + 2.0 * ln_propensity
        a = int(np.argmax(p))
    if stars < 2.0 and ARCHETYPES[a] == "叠":
        a = ARCHETYPES.index("切")
    return a


def make_plan(sections: List[Section], archetype: int, rng: np.random.Generator, temperature: float = 1.0) -> Plan:
    """Sample one pattern type per section from P(type | archetype, energy level, previous type)."""
    plan = load_plan()
    first = np.array(plan["first"])
    trans = np.array(plan["trans"])
    dens = np.array(plan["density"])
    light = TYPES.index("light")
    prev: Optional[int] = None
    out = []
    for s in sections:
        p = first[archetype, s.level] if prev is None else trans[archetype, s.level, prev]
        p = np.array(p, float)
        if s.level >= 2:
            p[light] = 0.0                       # loud sections are never left sparse
        p = p ** (1.0 / temperature)
        p /= p.sum()
        t = int(rng.choice(len(TYPES), p=p))
        d = float(dens[archetype, s.level])
        # humans mix at most two types in a section: long sections get a secondary type on some
        # 4-measure phrases (alternating), drawn from the same distribution without the primary
        phrases = _phrases(s)
        if len(phrases) >= 2 and t != light and rng.random() < SECONDARY_RATE[ARCHETYPES[archetype]]:
            q = p.copy()
            q[t] = 0.0
            q[light] = 0.0
            if q.sum() > 0:
                t2 = int(rng.choice(len(TYPES), p=q / q.sum()))
                for n, (a, b) in enumerate(phrases):
                    out.append(Section(a, b, 4, s.energy, s.level, t if n % 2 == 0 else t2, d))
                prev = t
                continue
        out.append(Section(s.start_ms, s.end_ms, s.measures, s.energy, s.level, t,
                           d + (-0.7 if t == light else 0.0)))
        prev = t
    return Plan(archetype, out)


def _phrases(s: Section, size: int = 4) -> List[tuple]:
    """Split a section of >= 8 measures into 4-measure phrases (last one absorbs the remainder)."""
    if s.measures < 2 * size:
        return []
    m = (s.end_ms - s.start_ms) / s.measures
    n = s.measures // size
    edges = [s.start_ms + i * size * m for i in range(n)] + [s.end_ms]
    return list(zip(edges[:-1], edges[1:]))


def type_targets(type_index: int) -> dict:
    return load_plan()["targets"][TYPES[type_index]]
