"""Deterministic rule-based osu!mania 4K note and lane generation.

The bundled AUTO-OSU checkpoints were trained only on osu!standard maps.  This
module deliberately uses audio features and rules instead of presenting those
checkpoints as mania models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .audio import AudioAnalysis
from .beatmap import Circle, HitObject, Hold
from .difficulty import DifficultyPreset
from .rhythm import RhythmEvent, Sections, Tick, effective_grid, select_ticks, tick_features
from .timing import Timing

LANE_X: Tuple[int, int, int, int] = (64, 192, 320, 448)
MANIA_Y = 192


@dataclass(frozen=True)
class ManiaStyle:
    chord_chance: float
    triple_chance: float
    hold_chance: float
    max_hold_beats: float


STYLES = {
    "Easy": ManiaStyle(0.02, 0.00, 0.20, 2.0),
    "Normal": ManiaStyle(0.06, 0.00, 0.17, 2.0),
    "Hard": ManiaStyle(0.13, 0.01, 0.14, 1.5),
    "Insane": ManiaStyle(0.21, 0.035, 0.11, 1.0),
}


def _lane_order(rng: np.random.Generator, previous: int | None, occupied: Sequence[int]) -> List[int]:
    """Prefer alternating hands and avoid jacks, while retaining seeded variation."""
    candidates = [lane for lane in range(4) if lane not in occupied]
    rng.shuffle(candidates)
    if previous is None:
        return candidates
    previous_hand = previous // 2
    random_rank = {lane: rank for rank, lane in enumerate(candidates)}
    return sorted(candidates, key=lambda lane: (lane == previous, lane // 2 == previous_hand, random_rank[lane]))


def _hold_end(tick: Tick, timing: Timing, style: ManiaStyle, rng: np.random.Generator,
              max_time_ms: int | None) -> int:
    """Return a quantised LN end, or zero when this tick should stay a tap."""
    min_beats = 1.0 if style.max_hold_beats >= 2.0 else 0.5
    if tick.sustain_beats < min_beats * 0.8 or rng.random() >= style.hold_chance:
        return 0
    quantum = 0.5
    beats = min(style.max_hold_beats, tick.sustain_beats)
    if max_time_ms is not None:
        beats = min(beats, (max_time_ms - tick.time) / timing.beat_length)
    beats = np.floor(beats / quantum) * quantum
    if beats < min_beats:
        return 0
    end = timing.ms_at(tick.beat + beats)
    return end if end - tick.time >= 100 and (max_time_ms is None or end <= max_time_ms) else 0


def build_mania_objects(analysis: AudioAnalysis, timing: Timing, preset: DifficultyPreset,
                        rng: np.random.Generator, sections: Sections | None = None, *,
                        min_time_ms: int = 0, max_time_ms: int | None = None
                        ) -> tuple[List[RhythmEvent], List[HitObject]]:
    """Build playable 4K taps/chords/LNs with no overlap inside a lane."""
    sections = sections or Sections(np.ones(1), [])
    style = STYLES[preset.name]
    grid = effective_grid(analysis, timing, preset)
    selected = select_ticks(tick_features(analysis, timing, grid), preset, grid, timing, sections)
    events: List[RhythmEvent] = []
    objects: List[HitObject] = []
    lane_free = [-10**9] * 4
    previous_lane: int | None = None

    for tick in selected:
        if tick.time < min_time_ms or (max_time_ms is not None and tick.time > max_time_ms):
            continue
        available = [lane for lane in range(4) if lane_free[lane] < tick.time]
        if not available:
            continue
        order = _lane_order(rng, previous_lane, ())
        primary = next(lane for lane in order if lane in available)
        lanes = [primary]

        # Chords are reserved for salient strong beats and never repeat a lane at the same timestamp.
        strong = tick.score >= max(0.45, preset.strength_threshold + 0.12)
        chord_probability = style.chord_chance * (0.55 + 0.65 * sections.at(tick.beat))
        if strong and rng.random() < chord_probability:
            partners = _lane_order(rng, primary, lanes)
            partner = next((lane for lane in partners if lane in available and lane // 2 != primary // 2), None)
            if partner is not None:
                lanes.append(partner)
                if style.triple_chance and rng.random() < style.triple_chance:
                    third = next((lane for lane in partners if lane in available and lane not in lanes), None)
                    if third is not None:
                        lanes.append(third)

        hold_end = _hold_end(tick, timing, style, rng, max_time_ms)
        hold_lane = primary if hold_end else None
        for lane in sorted(lanes):
            end = hold_end if lane == hold_lane else 0
            event = RhythmEvent(
                time=tick.time, beat=tick.beat, kind="hold" if end else "circle",
                strength=tick.score, percussive=tick.percussive, kick=tick.kick, snare=tick.snare,
                intensity=sections.at(tick.beat), end_time=end,
                end_beat=timing.beat_at(end) if end else 0.0, lane=lane,
            )
            events.append(event)
            if end:
                objects.append(Hold(LANE_X[lane], MANIA_Y, tick.time, end=end))
                lane_free[lane] = end
            else:
                objects.append(Circle(LANE_X[lane], MANIA_Y, tick.time))
                lane_free[lane] = tick.time
        previous_lane = primary

    return events, objects
