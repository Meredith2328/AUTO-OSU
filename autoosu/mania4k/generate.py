"""Song -> osu!mania 4K charts: exact timing, learned note placement, learned lane patterns.

    analysis = analyse_song(path)                       # audio features, timing (once per song)
    chart, report = generate_chart(analysis, "Hard")    # one difficulty, calibrated to its star target

Guarantees enforced here (and re-checked by :func:`autoosu.mania4k.verify.verify_chart`):
* every note head and long-note tail lies exactly on the beat grid of the red lines;
* every note head sits on a distinct audible attack (a prominent onset peak within +-8 ms of the
  tick) - stricter than human mappers, ~14 % of whose notes have no such peak;
* lanes never overlap, same-lane repeats respect the difficulty's minimum interval, chord sizes
  and snap divisors respect the difficulty;
* the chart's star rating (computed with rosu-pp, the same algorithm as osu!) lands on the target.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .chart import KEYS, Chart, Note
from .features import DIV_CLASSES, POS_DIV, Grid, SongFeatures, build_grid, env_matrix, local_loudness, mel_index, song_features
from .model import LN_BINS, NoteNet
from .onsets import OnsetEnvelopes, attack_times, near_attack
from .patterns import PatternNet, RowState, advance, allowed_masks, lanes_of, row_features
from .timing import TimingResult, estimate_timing, tracker_activations

WEIGHTS = Path(__file__).resolve().parent / "weights"

# Ranked charts are timed this much earlier than the attack in the decoded audio (median over the
# ranked corpus, measured with the same onset envelope the timing engine locks onto).
OSU_SHIFT_MS = 23.0

DIFFICULTIES: Dict[str, float] = {"Easy": 1.6, "Normal": 2.35, "Hard": 3.2, "Insane": 4.3, "Expert": 5.4}


@dataclass(frozen=True)
class Rules:
    """Hard limits per difficulty, from the 1st percentiles of ranked 4K charts of that star range."""
    min_jack_ms: float
    min_row_gap_ms: float
    max_chord: int
    divisors: Tuple[int, ...]
    od: float
    hp: float
    max_ln_share: float


def rules_for(stars: float) -> Rules:
    if stars < 1.9:
        return Rules(300, 150, 2, (1, 2), 6.5, 6.5, 0.25)
    if stars < 2.7:
        return Rules(170, 90, 2, (1, 2, 4, 3), 7.0, 7.0, 0.25)
    if stars < 3.6:
        return Rules(130, 62, 3, (1, 2, 4, 3, 6), 7.5, 7.5, 0.25)
    if stars < 4.6:
        return Rules(100, 45, 3, (1, 2, 4, 8, 3, 6), 8.0, 8.0, 0.25)
    return Rules(85, 34, 4, (1, 2, 4, 8, 3, 6), 8.0, 8.0, 0.25)


@dataclass
class SongAnalysis:
    path: Path
    duration_ms: float
    features: SongFeatures
    timing: TimingResult
    grid: Grid
    support: np.ndarray                 # per tick: True when an attack sits on the tick
    probs: Dict[float, Dict[str, np.ndarray]] = field(default_factory=dict)   # by star target


_MODELS: Dict[str, object] = {}


def load_models(device: str = "cpu") -> Tuple[NoteNet, PatternNet]:
    if "notes" not in _MODELS:
        _MODELS["notes"] = NoteNet.from_checkpoint(str(WEIGHTS / "notes.pt"), device)
        _MODELS["patterns"] = PatternNet.from_checkpoint(str(WEIGHTS / "patterns.pt"), device)
    return _MODELS["notes"], _MODELS["patterns"]          # type: ignore[return-value]


def onset_support(env: OnsetEnvelopes, times: np.ndarray) -> np.ndarray:
    """True where a distinct attack (prominent onset peak, any band) lies within +-8 ms of the tick."""
    return near_attack(times, attack_times(env))


def analyse_song(path: str | Path, y: Optional[np.ndarray] = None, sr: Optional[int] = None,
                 beat_logits: Optional[Tuple[np.ndarray, np.ndarray]] = None, bpm: Optional[float] = None,
                 offset_ms: Optional[float] = None) -> SongAnalysis:
    from ..audio import load_audio

    path = Path(path)
    if y is None:
        y, sr = load_audio(path, sr=22050)
    feat = song_features(y, sr)
    beat, down = beat_logits if beat_logits is not None else tracker_activations(y, sr)
    timing = estimate_timing(feat.env, beat, down, bpm=bpm, offset_ms=offset_ms)
    from .onsets import onset_peaks

    o, h = onset_peaks(feat.env, min_height=0.12)
    first = float(o[0]) if len(o) else 0.0
    last = float(o[-1]) if len(o) else feat.duration_ms
    grid = build_grid(timing.red_lines, max(0.0, first - 50.0), min(feat.duration_ms - 30.0, last + 50.0))
    return SongAnalysis(path, feat.duration_ms, feat, timing, grid, onset_support(feat.env, grid.times))


@torch.no_grad()
def note_probabilities(an: SongAnalysis, net: NoteNet, stars: float) -> Dict[str, np.ndarray]:
    key = round(stars, 3)
    if key in an.probs:
        return an.probs[key]
    g = an.grid
    mel = torch.from_numpy(an.features.mel.astype(np.float32) / 255.0)[None]
    t = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt)[None]   # noqa: E731
    out = net(mel, t(mel_index(g.times)), t(env_matrix(an.features.env, g.times), torch.float32),
              t(g.pos), t(g.div), t(g.beat_in_measure), t(local_loudness(an.features.mel, g.times), torch.float32),
              t(g.beat_ms, torch.float32), torch.tensor([stars], dtype=torch.float32))
    res = {"count": torch.softmax(out["count"][0], -1).numpy(), "ln": torch.sigmoid(out["lnhead"][0]).numpy(),
           "lnlen": torch.softmax(out["lnlen"][0], -1).numpy()}
    an.probs[key] = res
    return res


# --------------------------------------------------------------------------- selection

@dataclass
class Row:
    tick: int
    time: float
    k: int
    ln: bool
    ln_beats: float


def select_rows(an: SongAnalysis, probs: Dict[str, np.ndarray], rules: Rules, theta: float,
                chord_boost: float = 1.0) -> List[Row]:
    g = an.grid
    p_note = 1.0 - probs["count"][:, 0]
    allowed_div = np.isin(np.array([DIV_CLASSES[d] for d in g.div]), rules.divisors)
    cand = np.where((p_note >= theta) & an.support & allowed_div)[0]
    order = cand[np.argsort(-p_note[cand], kind="stable")]
    taken: List[float] = []
    chosen: List[int] = []
    for i in order:
        t = g.times[i]
        j = bisect.bisect_left(taken, t)
        if j > 0 and t - taken[j - 1] < rules.min_row_gap_ms:
            continue
        if j < len(taken) and taken[j] - t < rules.min_row_gap_ms:
            continue
        taken.insert(j, t)
        chosen.append(int(i))
    chosen.sort()
    if not chosen:
        return []
    pc = probs["count"][chosen]
    cond = pc[:, 1:] / np.maximum(pc[:, 1:].sum(1, keepdims=True), 1e-9)     # p(k | note)
    ge2, ge3, ge4 = cond[:, 1:].sum(1), cond[:, 2:].sum(1), cond[:, 3]
    # keep the model's expected share of chords, giving them to the most chord-like rows
    k = np.ones(len(chosen), int)
    for level, share_p in ((2, ge2), (3, ge3), (4, ge4)):
        if level > rules.max_chord:
            break
        n = min(len(chosen), int(round(share_p.sum() * chord_boost)))
        if n <= 0:
            continue
        top = np.argsort(-share_p, kind="stable")[:n]
        k[top[k[top] >= level - 1]] = level
    # long notes where the model is confident (sustained sounds), at most the difficulty's share
    p_ln = probs["ln"][chosen]
    confident = np.where(p_ln >= 0.5)[0]
    n_ln = min(len(confident), int(rules.max_ln_share * len(chosen)))
    ln_rows = set(confident[np.argsort(-p_ln[confident], kind="stable")[:n_ln]].tolist())
    rows = []
    for r, i in enumerate(chosen):
        beats = LN_BINS[int(np.argmax(probs["lnlen"][i]))]
        rows.append(Row(i, float(g.times[i]), int(k[r]), r in ln_rows, beats))
    return rows


# --------------------------------------------------------------------------- patterns

def _snap_tail(an: SongAnalysis, head_tick: int, beats: float, rules: Rules) -> Optional[float]:
    g = an.grid
    target = g.beat[head_tick] + beats
    j = int(np.searchsorted(g.beat, target - 1e-6))
    best = None
    for jj in (j - 1, j, j + 1):
        if 0 <= jj < len(g) and jj > head_tick and DIV_CLASSES[g.div[jj]] in rules.divisors and POS_DIV[g.pos[jj]] in (1, 2, 4):
            if best is None or abs(g.beat[jj] - target) < abs(g.beat[best] - target):
                best = jj
    if best is None:
        return None
    return float(g.times[best]) if g.times[best] - g.times[head_tick] >= 80.0 else None


@torch.no_grad()
def assign_lanes(an: SongAnalysis, rows: List[Row], rules: Rules, pnet: PatternNet, stars: float,
                 rng: np.random.Generator, temperature: float = 0.75) -> List[Note]:
    st = RowState.fresh()
    notes: List[Note] = []
    release_gap = lambda bm: max(60.0, bm / 4.0)         # noqa: E731 - free time after an LN tail
    for row in rows:
        bm = float(an.grid.beat_ms[row.tick])
        # a lane is busy until its LN tail plus a short release gap
        busy = st.held_until + release_gap(bm)
        saved = st.held_until.copy()
        st.held_until = np.where(busy > row.time - 1.0, np.maximum(st.held_until, row.time), st.held_until)
        k = row.k
        ok = allowed_masks(st, row.time, k, rules.min_jack_ms)
        while not ok.any() and k > 1:
            k -= 1
            ok = allowed_masks(st, row.time, k, rules.min_jack_ms)
        st.held_until = saved
        if not ok.any():
            continue
        x = torch.from_numpy(row_features(st, row.time, k, row.ln, bm, stars))[None]
        logits = pnet(x)[0].numpy().astype(np.float64)
        logits[~ok] = -np.inf
        p = np.exp((logits - logits[ok].max()) / temperature)
        p /= p.sum()
        mask = int(rng.choice(16, p=p))
        ends = [0.0] * KEYS
        if row.ln:
            lanes = lanes_of(mask)
            ln_lanes = lanes if (len(lanes) == 1 or rng.random() < 0.35) else [lanes[int(rng.integers(len(lanes)))]]
            tail = _snap_tail(an, row.tick, row.ln_beats, rules)
            if tail is not None:
                for l in ln_lanes:
                    ends[l] = tail
        for l in lanes_of(mask):
            notes.append(Note(row.time, l, ends[l]))
        advance(st, row.time, mask, ends)
    return notes


# --------------------------------------------------------------------------- star rating

def star_rating(chart_text: str) -> float:
    import rosu_pp_py as rosu

    return float(rosu.Difficulty().calculate(rosu.Beatmap(content=chart_text)).stars)


def chart_to_osu_text(chart: Chart, shift_ms: float = OSU_SHIFT_MS, title: str = "x", artist: str = "x",
                      creator: str = "AUTO-OSU", audio: str = "audio.mp3", background: str = "",
                      kiai: Sequence[Tuple[float, float]] = (), preview_ms: int = -1, tags: str = "") -> str:
    from ..beatmap import Beatmap, Circle, Hold, TimingPoint

    tps = [TimingPoint(int(round(r.time - shift_ms)), r.beat_ms, meter=r.meter) for r in chart.red_lines]
    for a, b in kiai:
        tps.append(TimingPoint(int(round(a - shift_ms)), -100.0, uninherited=False, kiai=True))
        tps.append(TimingPoint(int(round(b - shift_ms)), -100.0, uninherited=False, kiai=False))
    objs = []
    for n in sorted(chart.notes, key=lambda n: (n.time, n.lane)):
        x = (64, 192, 320, 448)[n.lane]
        t = int(round(n.time - shift_ms))
        objs.append(Hold(x, 192, t, end=int(round(n.end - shift_ms))) if n.is_hold else Circle(x, 192, t))
    bm = Beatmap(audio_filename=audio, title=title, artist=artist, version=chart.version, creator=creator,
                 hp=chart.hp, cs=4, od=chart.od, ar=5, preview_time=preview_ms, background=background,
                 mode=3, timing_points=tps, hit_objects=objs,
                 tags=tags or "autoosu mania 4k ranked-calibrated")
    return bm.to_osu()


# --------------------------------------------------------------------------- main entry

@dataclass
class ChartReport:
    name: str
    target_stars: float
    stars: float
    theta: float
    rows: int
    notes: int
    holds: int


def generate_chart(an: SongAnalysis, name: str, target_stars: Optional[float] = None, seed: int = 0,
                   models: Optional[Tuple[NoteNet, PatternNet]] = None, tolerance: float = 0.08
                   ) -> Tuple[Chart, ChartReport]:
    target = float(target_stars if target_stars is not None else DIFFICULTIES[name])
    rules = rules_for(target)
    nnet, pnet = models or load_models()
    probs = note_probabilities(an, nnet, target)

    def build(theta: float, boost: float) -> Tuple[Chart, float]:
        rows = select_rows(an, probs, rules, theta, boost)
        notes = assign_lanes(an, rows, rules, pnet, target, np.random.default_rng(seed))
        chart = Chart(notes, list(an.timing.red_lines), rules.od, rules.hp, name)
        return chart, (star_rating(chart_to_osu_text(chart)) if notes else 0.0)

    best: Optional[Tuple[float, Chart, float]] = None
    for boost in (1.0, 1.3, 1.6):           # more chords only when the song is too sparse otherwise
        lo, hi = 0.02, 0.98                  # higher theta -> fewer notes -> lower stars
        theta = 0.5
        for _ in range(12):
            chart, sr = build(theta, boost)
            if best is None or abs(sr - target) < abs(best[2] - target):
                best = (theta, chart, sr)
            if abs(sr - target) <= tolerance:
                break
            if sr > target:
                lo = theta
            else:
                hi = theta
            theta = 0.5 * (lo + hi)
        if abs(best[2] - target) <= tolerance or best[2] > target:
            break
    theta, chart, sr = best
    holds = sum(n.is_hold for n in chart.notes)
    return chart, ChartReport(name, target, sr, theta, len({n.time for n in chart.notes}), len(chart.notes), holds)


def kiai_sections(an: SongAnalysis, min_measures: int = 8) -> List[Tuple[float, float]]:
    """Loudest stretches of the song (>= min_measures long) for kiai time."""
    reds = an.timing.red_lines
    r = reds[0]
    measure = r.beat_ms * r.meter
    starts = np.arange(r.time, an.duration_ms - measure, measure)
    if len(starts) < 2 * min_measures:
        return []
    loud = local_loudness(an.features.mel, starts + measure / 2, window_s=measure / 1000.0)
    hot = loud >= max(0.7, float(np.percentile(loud, 70)))
    out, i = [], 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j < len(hot) and hot[j]:
                j += 1
            if j - i >= min_measures:
                out.append((float(starts[i]), float(starts[j - 1] + measure)))
            i = j
        else:
            i += 1
    return out


def build_beatmap(an: SongAnalysis, chart: Chart, audio_filename: str, title: str, artist: str,
                  creator: str = "AUTO-OSU", shift_ms: float = OSU_SHIFT_MS, background: str = "",
                  kiai: Optional[Sequence[Tuple[float, float]]] = None, stars: Optional[float] = None):
    """A ready-to-write osu!mania Beatmap (osu! time = audio time - shift_ms)."""
    from ..beatmap import Break, Circle, Hold, TimingPoint, Beatmap

    kiai = kiai_sections(an) if kiai is None else kiai
    tps = [TimingPoint(int(round(r.time - shift_ms)), r.beat_ms, meter=r.meter) for r in chart.red_lines]
    for a, b in kiai:
        tps.append(TimingPoint(int(round(a - shift_ms)), -100.0, uninherited=False, kiai=True))
        tps.append(TimingPoint(int(round(b - shift_ms)), -100.0, uninherited=False, kiai=False))
    objs = []
    for n in sorted(chart.notes, key=lambda n: (n.time, n.lane)):
        x = (64, 192, 320, 448)[n.lane]
        t = int(round(n.time - shift_ms))
        objs.append(Hold(x, 192, t, end=int(round(n.end - shift_ms))) if n.is_hold else Circle(x, 192, t))
    breaks = []
    ends = sorted((o.time, o.end_time) for o in objs)
    for (a0, a1), (b0, _) in zip(ends, ends[1:]):
        if b0 - a1 >= 6000:
            breaks.append(Break(a1 + 500, b0 - 1500))
    preview = int(round(kiai[0][0] - shift_ms)) if kiai else (objs[len(objs) * 2 // 5].time if objs else -1)
    tag_sr = f" sr{stars:.2f}" if stars is not None else ""
    return Beatmap(audio_filename=audio_filename, title=title, artist=artist, version=chart.version,
                   creator=creator, hp=chart.hp, cs=4, od=chart.od, ar=5, preview_time=preview,
                   background=background, mode=3, timing_points=tps, breaks=breaks, hit_objects=objs,
                   tags="autoosu mania 4k ranked-calibrated kanzei" + tag_sr)
