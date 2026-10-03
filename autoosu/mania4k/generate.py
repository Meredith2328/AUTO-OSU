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

from .chart import KEYS, Chart, Note, RedLine
from .structure import TYPES
from .features import DIV_CLASSES, POS_DIV, Grid, SongFeatures, build_grid, env_matrix, local_loudness, mel_index, song_features
from .model import LN_BINS, NoteNet
from .onsets import OnsetEnvelopes, attack_times, near_attack
from .patterns import PatternNet, RowState, advance, allowed_masks, lanes_of, row_features
from .timing import TimingResult, estimate_timing, tracker_activations

WEIGHTS = Path(__file__).resolve().parent / "weights"

# Ranked charts are timed this much earlier than the attack in the decoded audio (median over the
# ranked corpus, measured with the same onset envelope the timing engine locks onto).
OSU_SHIFT_MS = 24.0

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
    max_ln_share: float          # long notes on at most this share of rows (most sustained sounds)
    max_chord_share: float       # ranked 75th percentile of chord rows for the star range


def rules_for(stars: float) -> Rules:
    if stars < 1.9:
        return Rules(300, 150, 2, (1, 2), 6.5, 6.5, 0.12, 0.27)
    if stars < 2.7:
        return Rules(170, 90, 2, (1, 2, 4, 3), 7.0, 7.0, 0.12, 0.44)
    if stars < 3.6:
        return Rules(130, 62, 3, (1, 2, 4, 3, 6), 7.5, 7.5, 0.12, 0.51)
    if stars < 4.6:
        return Rules(100, 45, 3, (1, 2, 4, 8, 3, 6), 8.0, 8.0, 0.12, 0.51)
    return Rules(85, 34, 4, (1, 2, 4, 8, 3, 6), 8.0, 8.0, 0.12, 0.47)


@dataclass
class SongAnalysis:
    path: Path
    duration_ms: float
    features: SongFeatures
    timing: TimingResult
    grid: Grid
    support: np.ndarray                 # per tick: True when an attack sits on the tick
    sections: list = field(default_factory=list)    # planner.Section: music sections + energy
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
    from .planner import music_sections

    secs = music_sections(feat.mel, feat.env, timing.red_lines, float(grid.times[0]), float(grid.times[-1]) + 1)
    return SongAnalysis(path, feat.duration_ms, feat, timing, grid, onset_support(feat.env, grid.times), secs)


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
    style: int = 1             # planned pattern type of the row's section (structure.TYPES)


def select_rows(an: SongAnalysis, probs: Dict[str, np.ndarray], rules: Rules, theta: float,
                chord_boost: float = 1.0, plan=None, contrast: float = 0.8) -> List[Row]:
    """Rows above theta, with each section's density scaled by its planned relative density, and
    chords / long notes given per section in the proportions human charts use for its pattern type."""
    from .planner import type_targets

    g = an.grid
    p_note = 1.0 - probs["count"][:, 0]
    sec = plan.section_of(g.times) if plan is not None else np.zeros(len(g), int)
    weight = np.exp(contrast * np.array([s.density for s in plan.sections]))[sec] if plan is not None else 1.0
    score = p_note * weight
    divs = np.array([DIV_CLASSES[d] for d in g.div])
    # 1/8 only where it is a playable rhythm rather than a flam (>= 55 ms apart)
    allowed_div = np.isin(divs, rules.divisors) & ((divs != 8) | (g.beat_ms / 8.0 >= 55.0))
    cand = np.where((score >= theta) & an.support & allowed_div)[0]
    order = cand[np.argsort(-score[cand], kind="stable")]
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
    chosen = consistent_snaps(g, sorted(chosen), p_note)
    if not chosen:
        return []
    chosen_arr = np.array(chosen)
    pc = probs["count"][chosen_arr]
    cond = pc[:, 1:] / np.maximum(pc[:, 1:].sum(1, keepdims=True), 1e-9)     # p(k | note)
    ge2, ge3, ge4 = cond[:, 1:].sum(1), cond[:, 2:].sum(1), cond[:, 3]
    p_ln = probs["ln"][chosen_arr]
    k = np.ones(len(chosen), int)
    ln = np.zeros(len(chosen), bool)
    groups = [np.arange(len(chosen))] if plan is None else \
        [np.where(sec[chosen_arr] == j)[0] for j in range(len(plan.sections))]
    for j, idx in enumerate(groups):
        if len(idx) == 0:
            continue
        if plan is None:
            shares = (ge2[idx].sum() / len(idx), ge3[idx].sum() / len(idx), ge4[idx].sum() / len(idx))
            ln_share, ln_floor = rules.max_ln_share, 0.25
            cap = rules.max_chord_share
        else:
            tg = type_targets(plan.sections[j].type)
            multi = max(0.0, tg["notes_per_row"] - 1.0 - tg["triple"])          # rows with >= 2 notes
            shares = (multi, tg["triple"], tg["triple"] * 0.15)
            heavy = TYPES[plan.sections[j].type] in ("chordjack", "handstream")
            cap = 0.95 if heavy else rules.max_chord_share * 1.25
            ln_share = tg["ln_rows"] if TYPES[plan.sections[j].type] == "ln" else min(tg["ln_rows"], rules.max_ln_share)
            ln_floor = 0.0 if TYPES[plan.sections[j].type] == "ln" else 0.25
        for level, share, rank in ((2, min(shares[0] * chord_boost, cap), ge2), (3, shares[1] * chord_boost, ge3),
                                   (4, shares[2], ge4)):
            if level > rules.max_chord:
                break
            n = int(round(share * len(idx)))
            if n <= 0:
                continue
            top = idx[np.argsort(-rank[idx], kind="stable")[:n]]
            k[top[k[top] >= level - 1]] = level
        conf = idx[p_ln[idx] >= ln_floor]
        n_ln = min(len(conf), int(round(ln_share * len(idx))))
        if n_ln > 0:
            ln[conf[np.argsort(-p_ln[conf], kind="stable")[:n_ln]]] = True
    rows = []
    for r, i in enumerate(chosen):
        beats = LN_BINS[int(np.argmax(probs["lnlen"][i]))]
        style = plan.sections[int(sec[i])].type if plan is not None else 1
        rows.append(Row(i, float(g.times[i]), int(k[r]), bool(ln[r]), beats, style))
    return rows


def consistent_snaps(g: Grid, chosen: List[int], p_note: np.ndarray) -> List[int]:
    """Mappers do not mix straight and triplet rhythms inside a beat, nor drop a lone triplet into
    straight music: keep one family per beat (the one the model supports more) and only keep
    triplet beats that belong to a triplet passage (another triplet beat within two beats)."""
    if not chosen:
        return chosen
    fam = {}                                         # beat -> {"s": [...], "t": [...]}
    for i in chosen:
        d = POS_DIV[g.pos[i]]
        b = int(np.floor(g.beat[i] + 1e-6))
        kind = "t" if d in (3, 6) else ("s" if d != 1 else "b")
        fam.setdefault(b, {"s": [], "t": [], "b": []})[kind].append(i)
    triplet_beats = set()
    keep: List[int] = []
    for b, f in fam.items():
        if f["t"] and f["s"]:
            if p_note[f["t"]].sum() > p_note[f["s"]].sum():
                f["s"] = []
            else:
                f["t"] = []
        if f["t"]:
            triplet_beats.add(b)
    for b, f in fam.items():
        keep += f["b"] + f["s"]
        if f["t"] and any(nb in triplet_beats for nb in (b - 2, b - 1, b + 1, b + 2)):
            keep += f["t"]
    return sorted(keep)


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
                 rng: np.random.Generator, temperature: float = 0.75, archetype: int = 0) -> List[Note]:
    st = RowState.fresh()
    notes: List[Note] = []
    release_gap = lambda bm: max(60.0, bm / 4.0)         # noqa: E731 - free time after an LN tail
    ln_type = TYPES.index("ln")
    for ri, row in enumerate(rows):
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
        x = torch.from_numpy(row_features(st, row.time, k, row.ln, bm, stars, row.style, archetype))[None]
        logits = pnet(x)[0].numpy().astype(np.float64) + style_bias(st, row.style, row.time, bm)
        logits[~ok] = -np.inf
        p = np.exp((logits - logits[ok].max()) / temperature)
        p /= p.sum()
        mask = int(rng.choice(16, p=p))
        ends = [0.0] * KEYS
        if row.ln:
            lanes = lanes_of(mask)
            ln_lanes = lanes if (len(lanes) == 1 or rng.random() < 0.35) else [lanes[int(rng.integers(len(lanes)))]]
            if row.style == ln_type:
                # LN sections: holds run through the flow, released on the row after next
                nxt = rows[ri + 2].time if ri + 2 < len(rows) else row.time + 2 * bm
                beats = max(0.5, min(4.0, (nxt - row.time) / bm))
            else:
                beats = max(0.5, row.ln_beats)
            tail = _snap_tail(an, row.tick, beats, rules)
            if tail is not None:
                for l in ln_lanes:
                    ends[l] = tail
        for l in lanes_of(mask):
            notes.append(Note(row.time, l, ends[l]))
        advance(st, row.time, mask, ends)
    return notes


_STYLE_IDX = {name: i for i, name in enumerate(TYPES)}


def style_bias(st: RowState, style: int, t: float, beat_ms: float, strength: float = 1.5) -> np.ndarray:
    """Logit bonus that makes the planned pattern type clearly recognisable: staircases for rolls,
    ABAB for trill sections, shared lanes for jacks/chordjacks, no shared lanes for streams."""
    b = np.zeros(16)
    hist = st.history
    if not hist:
        return b
    prev = hist[-1][0]
    name = TYPES[style]
    close = (t - st.last_time) <= beat_ms * 0.75 if st.last_time is not None else False
    for m in range(1, 16):
        overlap = (m & prev) != 0
        if name in ("jack", "chordjack"):
            b[m] += strength * (1.0 if overlap else -0.5)
        elif name in ("stream", "jumpstream", "handstream", "roll", "trill") and close:
            b[m] -= strength * (1.0 if overlap else 0.0)
    if name == "trill" and len(hist) >= 2:
        b[hist[-2][0]] += strength
    if name == "roll" and len(hist) >= 2 and bin(prev).count("1") == bin(hist[-2][0]).count("1") == 1:
        a, c = hist[-2][0].bit_length(), prev.bit_length()
        nxt = c + (c - a)
        if abs(c - a) == 1:
            if 1 <= nxt <= 4:
                b[1 << (nxt - 1)] += strength
            else:                                    # turn around at the edge: 1234321...
                b[1 << (c - (c - a) - 1)] += strength
    return b


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
    archetype: str = ""
    plan: str = ""


def generate_chart(an: SongAnalysis, name: str, target_stars: Optional[float] = None, seed: int = 0,
                   models: Optional[Tuple[NoteNet, PatternNet]] = None, tolerance: float = 0.08,
                   style: str = "auto", archetype: Optional[int] = None) -> Tuple[Chart, ChartReport]:
    """One difficulty. ``style`` picks the chart archetype (auto | stream | speed | jack | ln | hybrid);
    the section plan (one pattern type and density per music section) follows it."""
    from .planner import choose_archetype, make_plan
    from .style import analysis_descriptors

    target = float(target_stars if target_stars is not None else DIFFICULTIES[name])
    rules = rules_for(target)
    nnet, pnet = models or load_models()
    probs = note_probabilities(an, nnet, target)
    if archetype is None:
        archetype = choose_archetype(target, style, descriptors=analysis_descriptors(an, target))
    plan = make_plan(an.sections, archetype, np.random.default_rng(seed + 7919)) if an.sections else None

    def build(theta: float, boost: float) -> Tuple[Chart, float]:
        rows = select_rows(an, probs, rules, theta, boost, plan)
        notes = assign_lanes(an, rows, rules, pnet, target, np.random.default_rng(seed),
                             archetype=archetype)
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
    return chart, ChartReport(name, target, sr, theta, len({n.time for n in chart.notes}), len(chart.notes), holds,
                              plan.archetype_name if plan else "", plan.summary() if plan else "")


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


def normalized_sv(reds: Sequence[RedLine], end_ms: float) -> List[float]:
    """Slider velocity per red line that cancels osu!mania's BPM-dependent scroll speed: speed is
    relative to the beat length covering most of the map (as osu! computes it), so a red line
    with beat length L gets SV L / L_main. 1.0 everywhere for a single BPM."""
    from .chart import main_bpm

    main = 60000.0 / main_bpm(reds, end_ms)
    return [float(np.clip(r.beat_ms / main, 0.1, 10.0)) if abs(r.beat_ms - main) > 1e-3 else 1.0 for r in reds]


def build_beatmap(an: SongAnalysis, chart: Chart, audio_filename: str, title: str, artist: str,
                  creator: str = "AUTO-OSU", shift_ms: float = OSU_SHIFT_MS, background: str = "",
                  kiai: Optional[Sequence[Tuple[float, float]]] = None, stars: Optional[float] = None):
    """A ready-to-write osu!mania Beatmap (osu! time = audio time - shift_ms)."""
    from ..beatmap import Break, Circle, Hold, TimingPoint, Beatmap

    kiai = kiai_sections(an) if kiai is None else kiai
    reds = sorted(chart.red_lines, key=lambda r: r.time)
    tps = [TimingPoint(int(round(r.time - shift_ms)), r.beat_ms, meter=r.meter) for r in reds]
    sv = normalized_sv(reds, max((n.end or n.time) for n in chart.notes) if chart.notes else reds[-1].time)

    def sv_at(t: float) -> float:
        return sv[max([i for i, r in enumerate(reds) if r.time <= t + 1] or [0])]

    def kiai_at(t: float) -> bool:
        return any(a <= t < b for a, b in kiai)

    if len(set(sv)) > 1:                        # several BPMs: green lines keep the scroll speed constant
        for r, v in zip(reds, sv):
            tps.append(TimingPoint(int(round(r.time - shift_ms)), -100.0 / v, uninherited=False,
                                   kiai=kiai_at(r.time)))
    for a, b in kiai:
        tps.append(TimingPoint(int(round(a - shift_ms)), -100.0 / sv_at(a), uninherited=False, kiai=True))
        tps.append(TimingPoint(int(round(b - shift_ms)), -100.0 / sv_at(b), uninherited=False, kiai=False))
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
