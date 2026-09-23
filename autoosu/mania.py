"""osu!mania 7K generation: song -> rows of notes -> columns -> .osu (Mode 3, 7 keys).

The rhythm comes from the same analysis as the osu!standard path (drum / melody features per grid
tick, section loudness). On top of it this module decides, per difficulty:

* which ticks become rows (density follows the loudness of the section),
* how many keys each row presses (strong drums and loud sections get the chords),
* which rows become long notes (held melody notes),
* and which columns are used (hand balance, no unwanted jacks, flowing stairs / trills).

Presets were tuned against ranked-style community 7K sets (Easy ... Expert, about 1.3 ... 5.5 stars).
"""
from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio import AudioAnalysis
from .beatmap import Beatmap, HitObject, TimingPoint
from .difficulty import DifficultyPreset
from .rhythm import Grid, Sections, Tick, _is_on, _snap_fraction, select_ticks, tick_features
from .timing import Timing

KEYS = 7
LEFT, THUMB, RIGHT = (0, 1, 2), 3, (4, 5, 6)
TYPE_NOTE = 1
TYPE_HOLD = 128


# --------------------------------------------------------------------------- presets

@dataclass(frozen=True)
class ManiaPreset:
    name: str
    od: float
    hp: float
    rows_per_s: float            # target row density in a loud section
    max_divisor: int             # finest straight grid 1/2/4
    min_gap_ms: float            # shortest gap between two rows
    chords: Tuple[float, ...]    # share of rows pressing 1, 2, 3, ... keys
    fast_chord_max: int          # largest chord allowed when the previous row is < 1/2 beat away
    jack_ms: float               # same column twice closer than this is avoided
    ln_share: float              # target share of rows that start a long note
    ln_min_beats: float          # shortest long note
    ln_overlap: bool             # may other notes be pressed while a long note is held
    stream_max_len: int = 16
    floor: float = 0.12          # weakest tick score that may become a row
    star: float = 0.0            # rough target star rating (for reports and tests)


MANIA_PRESETS: Dict[str, ManiaPreset] = {
    "Easy": ManiaPreset(
        name="Easy", od=7, hp=7, rows_per_s=3.0, max_divisor=1, min_gap_ms=300,
        chords=(0.78, 0.22), fast_chord_max=1, jack_ms=900, ln_share=0.15, ln_min_beats=1.0,
        ln_overlap=False, stream_max_len=0, star=1.3, floor=0.15,
    ),
    "Normal": ManiaPreset(
        name="Normal", od=7, hp=7.5, rows_per_s=4.9, max_divisor=2, min_gap_ms=170,
        chords=(0.68, 0.28, 0.04), fast_chord_max=1, jack_ms=600, ln_share=0.14, ln_min_beats=1.0,
        ln_overlap=False, stream_max_len=0, star=2.1, floor=0.14,
    ),
    "Hard": ManiaPreset(
        name="Hard", od=7.5, hp=8, rows_per_s=7.2, max_divisor=4, min_gap_ms=95,
        chords=(0.56, 0.35, 0.09), fast_chord_max=2, jack_ms=420, ln_share=0.13, ln_min_beats=0.5,
        ln_overlap=True, stream_max_len=8, star=3.1, floor=0.1,
    ),
    "Insane": ManiaPreset(
        name="Insane", od=8, hp=8, rows_per_s=8.4, max_divisor=4, min_gap_ms=90,
        chords=(0.44, 0.35, 0.17, 0.04), fast_chord_max=3, jack_ms=300, ln_share=0.12,
        ln_min_beats=0.5, ln_overlap=True, stream_max_len=16, star=4.2, floor=0.09,
    ),
    "Expert": ManiaPreset(
        name="Expert", od=8, hp=8.5, rows_per_s=11.5, max_divisor=4, min_gap_ms=75,
        chords=(0.34, 0.33, 0.22, 0.09, 0.02), fast_chord_max=3, jack_ms=260, ln_share=0.12,
        ln_min_beats=0.5, ln_overlap=True, stream_max_len=32, star=5.4, floor=0.05,
    ),
}


def get_mania_preset(name: str) -> ManiaPreset:
    key = name.strip().lower()
    for k, v in MANIA_PRESETS.items():
        if k.lower() == key:
            return v
    raise KeyError(f"unknown 7K difficulty {name!r}, choose from {', '.join(MANIA_PRESETS)}")


# --------------------------------------------------------------------------- hit objects

def column_x(col: int) -> int:
    return int((col + 0.5) * 512 / KEYS)


@dataclass
class ManiaNote(HitObject):
    column: int = 0
    end: int = 0                 # > time: hold note

    @property
    def is_hold(self) -> bool:
        return self.end > self.time

    @property
    def end_time(self) -> int:
        return self.end if self.is_hold else self.time

    def to_line(self) -> str:
        x = column_x(self.column)
        if self.is_hold:
            return f"{x},192,{self.time},{TYPE_HOLD},{self.hitsound},{self.end}:0:0:0:0:"
        return f"{x},192,{self.time},{TYPE_NOTE},{self.hitsound},0:0:0:0:"


# --------------------------------------------------------------------------- rows

@dataclass
class Row:
    tick: Tick
    size: int = 1
    ln_beats: float = 0.0        # > 0: one note of this row is held this long
    ln_all: bool = False         # every note of the chord is held
    cols: Tuple[int, ...] = ()
    intensity: float = 0.5

    @property
    def beat(self) -> float:
        return self.tick.beat


def mania_grid(analysis: AudioAnalysis, timing: Timing, preset: ManiaPreset) -> Grid:
    base = preset.max_divisor
    while base > 1 and timing.beat_length / base < preset.min_gap_ms:
        base //= 2
    triplet = base > 1 and _snap_fraction(analysis, timing, 3) > _snap_fraction(analysis, timing, 4) + 0.05
    div = {1: 1, 2: 3, 4: 6}[base] if triplet else base
    step = 1.0 / div
    min_gap = step if step * timing.beat_length >= preset.min_gap_ms else 1.0 / base
    return Grid(div=div, base=base, min_gap=min_gap, min_slider=preset.ln_min_beats, quant=0.5 if base > 1 else 1.0)


def select_rows(ticks: List[Tick], preset: ManiaPreset, grid: Grid, timing: Timing,
                sections: Sections) -> List[Row]:
    shim = DifficultyPreset(
        name=preset.name, hp=preset.hp, cs=KEYS, od=preset.od, ar=8, slider_multiplier=1, spacing=1,
        max_divisor=preset.max_divisor, strength_threshold=0, max_nps=preset.rows_per_s,
        min_gap_beats=grid.min_gap, min_gap_ms=preset.min_gap_ms, slider_bias=0, min_slider_beats=1,
        stream_max_len=preset.stream_max_len, jump_prob=0, min_spinner_ms=10 ** 9,
    )
    return [Row(t, intensity=sections.at(t.beat)) for t in select_ticks(ticks, shim, grid, timing, sections, preset.floor)]


def _row_weight(row: Row) -> float:
    t = row.tick
    slot = t.beat % 4
    metric = 0.25 if abs(slot) < 1e-6 else 0.12 if _is_on(t.beat, 1) and round(slot) == 2 else \
        0.06 if _is_on(t.beat, 1) else 0.0
    return 0.45 * t.drum + 0.15 * t.melody + 0.25 * row.intensity + metric + 0.1 * t.loud


def assign_chords(rows: List[Row], preset: ManiaPreset, timing: Timing, rng: np.random.Generator) -> None:
    """Give the preset's chord mix to the rows, the heaviest rows (downbeats, loud drums) get the
    biggest chords. Fast passages stay light so streams remain readable."""
    if not rows:
        return
    w = np.array([_row_weight(r) for r in rows]) + rng.normal(0, 0.04, len(rows))
    order = np.argsort(-w)
    shares = np.array(preset.chords, dtype=float)
    shares = shares / shares.sum()
    counts = np.floor(shares * len(rows)).astype(int)
    counts[0] += len(rows) - counts.sum()
    sizes: List[int] = []
    for k in range(len(counts) - 1, -1, -1):
        sizes.extend([k + 1] * counts[k])
    for rank, i in enumerate(order):
        rows[i].size = sizes[rank]
    for i, r in enumerate(rows):
        gap_prev = r.beat - rows[i - 1].beat if i else 4.0
        gap_next = rows[i + 1].beat - r.beat if i + 1 < len(rows) else 4.0
        if min(gap_prev, gap_next) < 0.5 - 1e-6:
            r.size = min(r.size, preset.fast_chord_max)
        if r.intensity < 0.25 and r.size > 2:
            r.size -= 1


def assign_long_notes(rows: List[Row], preset: ManiaPreset, grid: Grid, rng: np.random.Generator) -> None:
    """Held melody notes become long notes; the longest sustains win until the preset share is met."""
    if not rows:
        return
    quant = 0.5 if grid.base >= 2 else 1.0
    cands = []
    for i, r in enumerate(rows):
        sus = r.tick.sustain_beats
        if sus < preset.ln_min_beats * 0.9 or r.tick.melody < 0.2:
            continue
        nxt = rows[i + 1].beat if i + 1 < len(rows) else r.beat + 8
        limit = min(sus, 4.0)
        if not preset.ln_overlap:
            limit = min(limit, nxt - r.beat - (1.0 if grid.base == 1 else 0.5))
        length = math.floor(limit / quant + 1e-6) * quant
        if length >= preset.ln_min_beats - 1e-6:
            cands.append((sus * (0.5 + r.tick.melody) + rng.normal(0, 0.05), i, length))
    cands.sort(reverse=True)
    target = int(round(preset.ln_share * len(rows)))
    for _, i, length in cands[:target]:
        rows[i].ln_beats = length
        rows[i].ln_all = rows[i].size >= 2 and preset.ln_overlap and rng.random() < 0.25


# --------------------------------------------------------------------------- columns

@dataclass
class _ColumnState:
    last_hit: List[float] = field(default_factory=lambda: [-1e9] * KEYS)     # ms
    held_until: List[float] = field(default_factory=lambda: [-1e9] * KEYS)   # ms incl. release gap
    load: np.ndarray = field(default_factory=lambda: np.zeros(KEYS))
    usage: np.ndarray = field(default_factory=lambda: np.zeros(KEYS))
    prev: Tuple[int, ...] = ()
    prev2: Tuple[int, ...] = ()
    last_time: float = -1e9
    direction: int = 1


_SUBSETS = {k: list(itertools.combinations(range(KEYS), k)) for k in range(1, KEYS + 1)}


def _hand(c: int) -> int:
    return 0 if c < THUMB else 2 if c > THUMB else 1


def _chord_cost(cols: Tuple[int, ...], preset: ManiaPreset) -> float:
    """Shape cost of pressing these columns together (one hand spanning three neighbours is hard)."""
    cost = 0.0
    left = [c for c in cols if c < THUMB]
    right = [c for c in cols if c > THUMB]
    for hand in (left, right):
        if len(hand) >= 3:
            cost += 1.5 if preset.max_divisor < 4 else 0.6
        elif len(hand) == 2 and abs(hand[0] - hand[1]) == 1 and preset.max_divisor < 2:
            cost += 0.4
    if len(cols) >= 2 and not left and THUMB not in cols:
        cost += 0.5
    if len(cols) >= 2 and not right and THUMB not in cols:
        cost += 0.5
    return cost


def pick_columns(row: Row, time_ms: float, beat_ms: float, st: _ColumnState, preset: ManiaPreset,
                 rng: np.random.Generator) -> Tuple[int, ...]:
    free = [c for c in range(KEYS) if st.held_until[c] <= time_ms]
    if not free:
        return ()
    k = max(1, min(row.size, len(free)))
    gap = time_ms - st.last_time
    decay = math.exp(-gap / 900.0)
    load = st.load * decay
    usage_dev = st.usage - st.usage.mean()
    best: List[Tuple[float, Tuple[int, ...]]] = []
    for cols in _SUBSETS[k]:
        if any(c not in free for c in cols):
            continue
        cost = 0.0
        for c in cols:
            since = time_ms - st.last_hit[c]
            if since < preset.jack_ms:
                cost += 4.0 * (1.0 - since / preset.jack_ms) + (4.0 if since < 0.6 * beat_ms else 0.0)
            cost += 0.9 * load[c] + 0.04 * usage_dev[c]
        if k >= 2:
            cost += _chord_cost(cols, preset)
            if set(cols) == set(st.prev):
                cost += 2.5
        held = [c for c in range(KEYS) if st.held_until[c] > time_ms]
        for c in cols:
            if any(_hand(c) == _hand(h) and _hand(c) != 1 for h in held):
                cost += 0.35
        if k == 1:
            c = cols[0]
            if st.prev and len(st.prev) == 1:
                step = c - st.prev[0]
                if step * st.direction in (1, 2):
                    cost -= 0.45                       # flowing stairs / rolls
                if st.prev2 and len(st.prev2) == 1 and c == st.prev2[0] and gap < 0.6 * beat_ms:
                    cost -= 0.25                       # short trills are idiomatic
            if c == THUMB:
                cost += 0.15
        # hand balance over the recent past
        lh = load[:THUMB].sum() + sum(1 for c in cols if c < THUMB)
        rh = load[THUMB + 1:].sum() + sum(1 for c in cols if c > THUMB)
        cost += 0.12 * abs(lh - rh)
        cost += rng.normal(0, 0.25)
        best.append((cost, cols))
    best.sort(key=lambda b: b[0])
    cols = best[0][1] if best else tuple(free[:k])
    if rng.random() < 0.12:
        st.direction *= -1
    if k == 1 and st.prev and len(st.prev) == 1:
        if cols[0] in (0, KEYS - 1):
            st.direction = 1 if cols[0] == 0 else -1
    return tuple(sorted(cols))


def assign_columns(rows: List[Row], timing: Timing, preset: ManiaPreset, rng: np.random.Generator) -> List[ManiaNote]:
    st = _ColumnState()
    notes: List[ManiaNote] = []
    bl = timing.beat_length
    release = max(0.5 * bl, 120.0) if not preset.ln_overlap else max(0.25 * bl, 90.0)
    for row in rows:
        t = float(row.tick.time)
        row.cols = pick_columns(row, t, bl, st, preset, rng)
        decay = math.exp(-(t - st.last_time) / 900.0)
        st.load *= decay
        hold_cols: Tuple[int, ...] = ()
        holding = sum(1 for c in range(KEYS) if st.held_until[c] > t)
        if holding + (len(row.cols) if row.ln_all else 1) > 3:
            row.ln_beats = 0
        if row.ln_beats > 0:
            if row.ln_all:
                hold_cols = row.cols
            else:
                hold_cols = (max(row.cols, key=lambda c: -st.usage[c]),)
        end_ms = timing.ms_at(row.beat + row.ln_beats) if row.ln_beats > 0 else 0
        for c in row.cols:
            held = c in hold_cols
            notes.append(ManiaNote(x=column_x(c), y=192, time=int(t), column=c, end=end_ms if held else 0))
            st.last_hit[c] = end_ms if held else t
            st.held_until[c] = (end_ms + release) if held else t + 1
            st.load[c] += 1.0
            st.usage[c] += 1.0
        st.prev2, st.prev, st.last_time = st.prev, row.cols, t
    return notes


# --------------------------------------------------------------------------- beatmap

@dataclass
class ManiaDiff:
    preset: ManiaPreset
    rows: List[Row]
    beatmap: Beatmap

    def summary(self) -> Dict[str, float]:
        objs = self.beatmap.hit_objects
        n = len(objs)
        holds = sum(1 for o in objs if isinstance(o, ManiaNote) and o.is_hold)
        span = (max(o.end_time for o in objs) - objs[0].time) / 1000.0 if n > 1 else 1.0
        return {"objects": n, "rows": len(self.rows), "holds": holds,
                "nps": round(n / max(span, 1e-6), 2), "rows_per_s": round(len(self.rows) / max(span, 1e-6), 2),
                "length_s": round(span, 1)}


def build_mania_beatmap(preset: ManiaPreset, timing: Timing, notes: List[ManiaNote], kiai: Sequence[tuple],
                        audio_filename: str, title: str, artist: str, creator: str, shift_ms: int) -> Beatmap:
    for n in notes:
        n.time -= shift_ms
        if n.end:
            n.end -= shift_ms
    tps = [TimingPoint(int(round(timing.offset_ms)) - shift_ms, timing.beat_length, uninherited=True)]
    for start, end in kiai:
        tps.append(TimingPoint(start - shift_ms, -100.0, uninherited=False, kiai=True))
        tps.append(TimingPoint(end - shift_ms, -100.0, uninherited=False, kiai=False))
    notes.sort(key=lambda n: (n.time, n.column))
    preview = (kiai[0][0] - shift_ms) if kiai else (notes[len(notes) * 2 // 5].time if notes else -1)
    return Beatmap(
        audio_filename=audio_filename, title=title, artist=artist, version=f"7K {preset.name}", creator=creator,
        tags="autoosu ai-generated mania 7k", hp=preset.hp, cs=KEYS, od=preset.od, ar=5,
        preview_time=preview, timing_points=tps, hit_objects=list(notes), mode=3,
    )


def generate_diff(analysis: AudioAnalysis, timing: Timing, sections: Sections, preset: ManiaPreset,
                  rng: np.random.Generator, audio_filename: str, title: str, artist: str, creator: str,
                  shift_ms: int) -> ManiaDiff:
    grid = mania_grid(analysis, timing, preset)
    ticks = tick_features(analysis, timing, grid)
    rows = select_rows(ticks, preset, grid, timing, sections)
    assign_chords(rows, preset, timing, rng)
    assign_long_notes(rows, preset, grid, rng)
    notes = assign_columns(rows, timing, preset, rng)
    bm = build_mania_beatmap(preset, timing, notes, sections.kiai, audio_filename, title, artist, creator, shift_ms)
    return ManiaDiff(preset, rows, bm)


def star_rating(beatmap: Beatmap) -> Optional[float]:
    """Star rating as computed by osu! (rosu-pp), when the optional package is installed."""
    try:
        import rosu_pp_py as rosu
    except ImportError:
        return None
    bm = rosu.Beatmap(content=beatmap.to_osu())
    return float(rosu.Difficulty().calculate(bm).stars)


# --------------------------------------------------------------------------- pipeline

@dataclass
class ManiaResult:
    osz: Path
    audio_file: Path
    timing: Timing
    analysis: AudioAnalysis
    diffs: List[ManiaDiff] = field(default_factory=list)
    elapsed_s: float = 0.0
    osu_shift_ms: int = 26
    device: str = "cpu"


def generate_mania(audio_path, difficulties: List[str], out_dir="out", seed: int = 0,
                   bpm: Optional[float] = None, offset_ms: Optional[float] = None,
                   title: Optional[str] = None, artist: Optional[str] = None, creator: str = "AUTO-OSU",
                   osu_shift_ms: int = 26, log: Callable[[str], None] = print,
                   progress: Optional[Callable[[float, str], None]] = None, **_ignored) -> ManiaResult:
    """Analyse a song and write one .osz with the requested 7K difficulties."""
    import time as _time

    from .audio import analyze, load_audio
    from .audio_io import VIDEO_EXTS
    from .package import extract_cover, prepare_audio, prepare_background, read_metadata, write_osz
    from .rhythm import analyse_sections
    from .timing import estimate_timing

    t0 = _time.perf_counter()
    audio_path, out_dir = Path(audio_path), Path(out_dir)
    presets = [get_mania_preset(d) for d in difficulties]
    if not presets:
        raise ValueError("Choose at least one difficulty")
    report = progress or (lambda f, m: None)
    report(0.0, "load")
    log(f"[1/4] loading {audio_path.name}")
    y, sr = load_audio(audio_path)
    report(0.05, "analyse")
    log(f"[2/4] analysing audio ({len(y) / sr:.1f} s)")
    analysis = analyze(y, sr)
    report(0.3, "timing")
    log("[3/4] estimating timing")
    timing = estimate_timing(analysis, bpm_override=bpm, offset_override_ms=offset_ms)
    log(f"      BPM {timing.bpm:g}  offset {timing.offset_ms:.0f} ms")

    meta_title, meta_artist = read_metadata(audio_path)
    title, artist = title or meta_title, artist or meta_artist
    workdir = out_dir / ".work"
    audio_file = prepare_audio(audio_path, workdir)
    background = None
    cover = extract_cover(audio_path)
    if cover:
        background = prepare_background(cover[0], cover[1], workdir)
    elif audio_path.suffix.lower() in VIDEO_EXTS:
        from .audio_io import extract_video_frame

        background = extract_video_frame(audio_path, workdir / "bg.jpg")
    sections = analyse_sections(analysis, timing)

    log("[4/4] generating 7K difficulties")
    diffs: List[ManiaDiff] = []
    for i, preset in enumerate(presets):
        report(0.4 + 0.55 * i / len(presets), f"7K {preset.name}")
        rng = np.random.default_rng(seed * 1000 + 700 + i)
        d = generate_diff(analysis, timing, sections, preset, rng, audio_file.name, title, artist, creator,
                          osu_shift_ms)
        if background:
            d.beatmap.background = background.name
        diffs.append(d)
        s = d.summary()
        sr_ = star_rating(d.beatmap)
        log(f"      7K {preset.name:<7} {s['objects']:5d} notes ({s['holds']} LN) "
            f"{s['nps']:.2f} notes/s" + (f"  {sr_:.2f}*" if sr_ is not None else ""))

    report(0.97, "package")
    osz = write_osz([d.beatmap for d in diffs], audio_file, out_dir, extra_files=[background] if background else ())
    report(1.0, "done")
    return ManiaResult(osz=osz, audio_file=audio_file, timing=timing, analysis=analysis, diffs=diffs,
                       elapsed_s=_time.perf_counter() - t0, osu_shift_ms=osu_shift_ms)
