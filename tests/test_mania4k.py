"""The ranked-calibrated 4K engine: timing exactness, grid, constraints, verifier, file round trip.

The neural beat tracker is replaced by synthetic activations here, so the tests need no download.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from autoosu.mania4k.chart import Chart, Note, RedLine, parse_osu, snap_of
from autoosu.mania4k.features import POSITIONS, build_grid
from autoosu.mania4k.generate import chart_to_osu_text, rules_for
from autoosu.mania4k.onsets import onset_envelopes
from autoosu.mania4k.patterns import RowState, advance, allowed_masks
from autoosu.mania4k.timing import TRACKER_FPS, estimate_timing, snap_bpm
from autoosu.mania4k.verify import verify_chart
from synth import make_song


def song(tmp_path, bpm, bars=24, offset_s=0.5, name="s.wav"):
    y, sr = sf.read(make_song(tmp_path / name, bpm=bpm, bars=bars, offset_s=offset_s))
    return y.astype(np.float32), sr


def logits_for(beats_ms, downs_ms, duration_ms):
    """Tracker-like logits: peaks of +6 at the given times, -8 elsewhere."""
    n = int(duration_ms / 1000 * TRACKER_FPS) + 1
    beat, down = np.full(n, -8.0, np.float32), np.full(n, -8.0, np.float32)
    for arr, times in ((beat, beats_ms), (down, downs_ms)):
        for t in times:
            i = int(round(t / 1000 * TRACKER_FPS))
            if 0 < i < n - 1:
                arr[i] = 6.0
                arr[i - 1] = arr[i + 1] = 0.0
    return beat, down


@pytest.mark.parametrize("bpm", [128.0, 174.0])
def test_constant_tempo_is_exact(tmp_path, bpm):
    y, sr = song(tmp_path, bpm)
    period = 60000.0 / bpm
    beats = [500.0 + k * period for k in range(24 * 4)]
    # a tracker that drifts into double time for a while must not matter
    beats += [500.0 + (k + 0.5) * period for k in range(30, 38)]
    beat, down = logits_for(sorted(beats), beats[0:96:4], len(y) / sr * 1000)
    res = estimate_timing(onset_envelopes(y, sr), beat, down)
    assert res.kind == "constant"
    assert len(res.red_lines) == 1
    r = res.red_lines[0]
    assert r.bpm == pytest.approx(bpm, abs=1e-6)
    # red line on a downbeat, phase within 2 ms of the synthetic drums
    k = (500.0 - r.time) / (4 * period)
    assert abs(k - round(k)) * 4 * period < 2.0


def test_tempo_change_gets_a_second_red_line(tmp_path):
    y1, sr = song(tmp_path, 140.0, bars=24, offset_s=0.5, name="a.wav")
    y2, _ = song(tmp_path, 165.0, bars=24, offset_s=0.0, name="b.wav")
    cut = int((0.5 + 24 * 4 * 60 / 140) * sr)
    y = np.concatenate([y1[:cut], y2])
    p1, p2 = 60000 / 140, 60000 / 165
    change = cut / sr * 1000
    beats = [500 + k * p1 for k in range(96)] + [change + k * p2 for k in range(96)]
    beat, down = logits_for(beats, beats[::4], len(y) / sr * 1000)
    res = estimate_timing(onset_envelopes(y, sr), beat, down)
    assert res.kind == "changes"
    bpms = [round(r.bpm, 2) for r in res.red_lines]
    assert bpms[0] == 140.0 and bpms[-1] == 165.0
    assert abs(res.red_lines[-1].time - change) <= p2 + 1


def test_bpm_snapping_prefers_integers_only_when_safe():
    assert snap_bpm(60000 / 174.0004, 400) == 174.0
    assert snap_bpm(60000 / 174.3, 400) != 174.0


def test_grid_contains_quarter_and_third_positions():
    g = build_grid([RedLine(1000.0, 500.0)], 1000.0, 3000.0)
    beats = g.beat - g.beat[0]
    assert len(POSITIONS) == 12
    for frac in (0.25, 1 / 3, 0.5, 2 / 3, 0.75, 0.125):
        assert np.any(np.abs(beats - (1 + frac)) < 1e-6)
    assert np.all(np.diff(g.times) > 0)


def test_lane_constraints():
    st = RowState.fresh()
    advance(st, 1000.0, 0b0001, [1600.0, 0, 0, 0])     # lane 0 holds until 1600
    advance(st, 1100.0, 0b0010)
    ok = allowed_masks(st, 1200.0, 1, min_jack_ms=150.0)
    assert not ok[0b0001]                               # held
    assert not ok[0b0010]                               # 100 ms jack < 150 ms
    assert ok[0b0100] and ok[0b1000]
    assert not allowed_masks(st, 1200.0, 2, 150.0)[0b0011]


def test_verifier_flags_every_problem_class():
    reds = [RedLine(1000.0, 500.0)]
    good = Chart([Note(1000.0, 0), Note(1250.0, 1), Note(1500.0, 2, 2000.0), Note(1750.0, 3)], reds)
    assert verify_chart(good, None, rules_for(3.0)).ok
    bad = Chart([Note(1000.0, 0), Note(1037.0, 1),            # off grid
                 Note(1500.0, 2, 2000.0), Note(1750.0, 2),    # inside the LN
                 Note(2000.0, 0), Note(2062.5, 0)], reds)     # 62 ms jack
    chk = verify_chart(bad, None, rules_for(3.0))
    assert chk.off_grid == 1 and chk.overlaps == 1 and chk.fast_jacks == 1
    assert not chk.ok


def test_osu_text_round_trip_applies_the_offset_convention():
    reds = [RedLine(1000.0, 60000 / 150), RedLine(9000.0, 60000 / 180)]
    notes = [Note(1000.0, 0), Note(1200.0, 3, 1600.0), Note(9000.0, 1), Note(9000.0, 2)]
    text = chart_to_osu_text(Chart(notes, reds, 8.0, 7.5, "Hard"), shift_ms=23.0)
    back = parse_osu(text)
    assert back is not None and back.od == 8.0 and back.hp == 7.5
    assert [(n.time, n.lane, n.end) for n in back.notes] == [(977.0, 0, 0.0), (1177.0, 3, 1577.0),
                                                             (8977.0, 1, 0.0), (8977.0, 2, 0.0)]
    assert [round(r.bpm, 6) for r in back.red_lines] == [150.0, 180.0]
    assert snap_of(back.red_lines, 1177.0) == 2


def test_ranked_engine_end_to_end(tmp_path, monkeypatch):
    """generate() with the shipped weights; the tracker is replaced by synthetic activations."""
    import zipfile

    from autoosu.generate import generate
    from autoosu.mania4k import generate as g4

    wav = make_song(tmp_path / "Artist - Song.wav", bpm=150, bars=32, offset_s=0.5)
    period = 400.0

    def fake_tracker(y, sr, device="cpu"):
        beats = [500.0 + k * period for k in range(32 * 4)]
        return logits_for(beats, beats[::4], len(y) / sr * 1000)

    monkeypatch.setattr(g4, "tracker_activations", fake_tracker)
    seen = []
    res = generate(wav, ["Normal", "Hard"], tmp_path / "out", mode="mania4k", log=seen.append)
    assert res.timing.bpm == 150.0
    with zipfile.ZipFile(res.osz) as z:
        charts = [parse_osu(z.read(n).decode("utf-8")) for n in z.namelist() if n.endswith(".osu")]
    assert len(charts) == 2 and all(c is not None and len(c.notes) > 50 for c in charts)
    for c in charts:
        assert len(c.red_lines) == 1 and c.red_lines[0].bpm == pytest.approx(150.0)
        # every note on the 1/4 or 1/3 grid of the red line
        assert all(snap_of(c.red_lines, n.time, (1, 2, 4, 3, 6), 1.0) is not None for n in c.notes)
    assert any("verified" in line for line in seen)


def _rows_to_notes(rows, beat_ms=250.0, t0=1000.0):
    """rows: list of lane masks, one per 1/4 beat; returns (time, lane, end) notes."""
    out = []
    for i, m in enumerate(rows):
        for l in range(4):
            if m >> l & 1:
                out.append((t0 + i * beat_ms / 4 * 2, l, 0.0))
    return out


def test_pattern_types_are_recognised():
    from autoosu.mania4k.structure import chart_profile

    reds = [RedLine(1000.0, 500.0)]
    cases = {
        "roll": [1, 2, 4, 8, 4, 2, 1, 2, 4, 8, 4, 2, 1, 2, 4, 8] * 4,
        "trill": [1, 2] * 32,
        "chordjack": [3, 7, 3, 14, 6, 7, 3, 11] * 8,
        "jumpstream": [5, 2, 8, 1, 10, 4, 1, 8] * 8,
    }
    for want, masks in cases.items():
        p = chart_profile(_rows_to_notes(masks), reds)
        active = [t for t in p["types"] if t != "light"]
        assert max(set(active), key=active.count) == want, (want, active)


def test_archetype_and_plan_follow_the_tables():
    import numpy as np

    from autoosu.mania4k.planner import Section, choose_archetype, make_plan
    from autoosu.mania4k.structure import ARCHETYPES, TYPES, chart_archetype

    assert chart_archetype(["ln"] * 6 + ["stream"] * 2) == "LN"
    assert chart_archetype(["chordjack"] * 4 + ["jumpstream"] * 6) == "叠"
    assert chart_archetype(["stream"] * 5 + ["jumpstream"] * 4 + ["mixed"]) == "切"
    assert ARCHETYPES[choose_archetype(1.5, "jack")] == "切"       # no jack charts below 2 stars
    assert ARCHETYPES[choose_archetype(4.0, "ln")] == "LN"
    secs = [Section(i * 8000.0, (i + 1) * 8000.0, 8, e, lv) for i, (e, lv) in
            enumerate([(-1.5, 0), (0.0, 2), (1.5, 3), (-0.5, 1)] * 3)]
    plan = make_plan(secs, ARCHETYPES.index("LN"), np.random.default_rng(0))
    types = [TYPES[s.type] for s in plan.sections]
    assert types.count("ln") >= len(types) // 2                  # an LN chart is mostly LN
    rest = [s.density for s in plan.sections if s.level == 0]
    peak = [s.density for s in plan.sections if s.level == 3]
    assert max(rest) < min(peak)                                  # quiet sections sparser than climaxes
    assert all(TYPES[s.type] != "light" for s in plan.sections if s.level >= 2)
