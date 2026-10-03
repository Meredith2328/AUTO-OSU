"""Synthetic final-output contracts. No checkpoint, audio file, or neural inference."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from autoosu.mania4k import generate as gen
from autoosu.mania4k import timing
from autoosu.mania4k.chart import Chart, Note, RedLine, parse_osu
from autoosu.mania4k.features import build_grid
from autoosu.mania4k.verify import verify_chart


class FixedLogits:
    def __call__(self, x):
        return torch.zeros((len(x), 16))


@pytest.fixture(autouse=True)
def forbid_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real model / audio access is forbidden in this suite")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(gen, "load_models", forbidden)
    monkeypatch.setattr(gen, "tracker_activations", forbidden)


def analysis_and_probs(times, counts=None, ln_time=None, reds=None):
    reds = reds or [RedLine(1000.0, 500.0)]
    grid = build_grid(reds, min(times), max(times) + 1)
    support = np.zeros(len(grid), bool)
    pc = np.zeros((len(grid), 5))
    pc[:, 0] = 1.0
    ln = np.zeros(len(grid))
    for j, t in enumerate(times):
        i = int(np.argmin(abs(grid.times - t)))
        assert abs(grid.times[i] - t) < 1e-4
        support[i] = True
        pc[i] = 0
        pc[i, 0] = 0.1
        pc[i, counts[j] if counts else 1] = 0.9
        if t == ln_time:
            ln[i] = 1.0
    length = np.zeros((len(grid), 8))
    length[:, 5] = 1.0  # two beats
    probs = {"count": pc, "ln": ln, "lnlen": length}
    an = gen.SongAnalysis(Path("synthetic-unused"), max(times) + 2000, None,
                          timing.TimingResult(reds, [], 1.0, "synthetic"), grid, support)
    return an, probs


def rhythm_oracle(chart):
    """Independent test oracle: enumerate local beat positions, tolerate export rounding."""
    groups = {}
    for t in chart.heads:
        ri = max([i for i, r in enumerate(chart.red_lines) if r.time <= t + 1.0] or [0])
        r = chart.red_lines[ri]
        u = (t - r.time) / r.beat_ms
        nearest = round(u)
        if abs(u - nearest) * r.beat_ms <= 1.0:
            continue
        beat = int(np.floor(u))
        frac = u - beat
        straight = min(abs(frac - k / 8) for k in range(9)) * r.beat_ms <= 1.0
        triplet = min(abs(frac - k / 6) for k in (1, 2, 4, 5)) * r.beat_ms <= 1.0
        if straight or triplet:
            groups.setdefault((ri, beat), set()).add("s" if straight else "t")
    mixed = {k for k, v in groups.items() if len(v) > 1}
    triplets = {k for k, v in groups.items() if "t" in v}
    isolated = {k for k in triplets if not any((k[0], k[1] + d) in triplets for d in (-2, -1, 1, 2))}
    return mixed, isolated


def decoded_fixture(monkeypatch):
    times = [1000 + 500 / 3, 1500.0, 1500 + 500 / 3] + list(np.arange(3000., 7501., 500.))
    an, probs = analysis_and_probs(times, [1, 4, 1] + [1] * 10, 1500.0)
    monkeypatch.setattr(gen, "note_probabilities", lambda *a: probs)
    scored = []
    def score(text):
        scored.append(parse_osu(text))
        return 5.4
    monkeypatch.setattr(gen, "star_rating", score)
    chart, report = gen.generate_chart(an, "Expert", models=(None, FixedLogits()), archetype=0)
    return an, probs, chart, report, scored


def test_final_chart_does_not_keep_triplet_after_its_partner_is_blocked(monkeypatch):
    an, probs, chart, report, scored = decoded_fixture(monkeypatch)
    rows = gen.select_rows(an, probs, gen.rules_for(5.4), 0.5)
    assert len(rows) == 13  # both triplet beats really reach the decoder
    assert any(n.time == 1500 and n.is_hold for n in chart.notes)
    assert not rhythm_oracle(chart)[1], chart.heads
    assert report.holds == 4  # retain real LN occupancy, do not cure by disabling LN
    assert all(not any(rhythm_oracle(c)) for c in scored)
    for text in (gen.chart_to_osu_text(chart), gen.build_beatmap(an, chart, "unused", "x", "x", kiai=[]).to_osu()):
        back = parse_osu(text)
        assert back is not None and not any(rhythm_oracle(back))
        assert verify_chart(back, rules=gen.rules_for(5.4)).ok


@pytest.mark.parametrize("times", [[1000 + 500 / 3], [1125, 1000 + 500 / 3, 1500 + 500 / 3]])
def test_verifier_rejects_invalid_final_rhythm(times):
    chart = Chart([Note(t, i % 4) for i, t in enumerate(times)], [RedLine(1000, 500)])
    assert any(rhythm_oracle(chart))
    assert not verify_chart(chart).ok
    back = parse_osu(gen.chart_to_osu_text(chart))
    assert not verify_chart(back).ok


@pytest.mark.parametrize("bpm,beats", [(120.012, 400), (179.9, 96), (100.00049, 2000)])
def test_snap_bpm_obeys_whole_segment_budget(bpm, beats):
    period = 60000 / bpm
    selected = timing.snap_bpm(period, beats)
    drift = abs(60000 / selected - period) * beats
    budget = 6.0 if selected == round(selected) else 2.0
    assert drift <= budget, (selected, drift, budget)


def test_final_timing_phase_redline_and_serialization_budget():
    period, phase, beats = 60000 / 120.012, 1000.37, 400
    o = phase + np.arange(beats + 1) * period
    act = np.ones(int(o[-1] / 20) + 10)
    finished = timing._finish(timing.Segment(o[0], o[-1], period, phase), o, np.ones(len(o)), act, act, 4)
    reds = timing.to_red_lines([finished], 4, phase)
    chart = Chart([Note(finished.phase + k * finished.period, k % 4) for k in range(beats + 1)], reds)
    back = parse_osu(gen.chart_to_osu_text(chart))
    assert back is not None
    selected = finished.bpm
    budget = 6.0 if selected == round(selected) else 2.0
    unquantized_drift = abs(finished.period - period) * beats
    serialized_drift = abs(back.red_lines[0].beat_ms - period) * beats
    start_error = back.notes[0].time + gen.OSU_SHIFT_MS - o[0]
    end_error = back.notes[-1].time + gen.OSU_SHIFT_MS - o[-1]
    print({"bpm": selected, "span_drift_ms": unquantized_drift, "serialized_span_drift_ms": serialized_drift,
           "start_error_ms": start_error, "end_error_ms": end_error, "phase": finished.phase,
           "red_time": reds[0].time, "serialized_red_time": back.red_lines[0].time})
    assert unquantized_drift <= budget
    assert serialized_drift <= budget  # six-decimal beat length is part of the output contract
    assert abs(end_error - start_error) <= budget + 1.0  # separately account for integer note quantization
    assert verify_chart(back).ok


def test_lower_threshold_preserves_selected_supported_rows():
    times = [1125., 1000 + 500 / 3, 1000 + 1000 / 3, 1500 + 500 / 3]
    an, probs = analysis_and_probs(times)
    for t, p in zip(times, [.8, .45, .45, .8]):
        i = np.argmin(abs(an.grid.times - t))
        probs["count"][i] = [1 - p, p, 0, 0, 0]
    high = {r.time for r in gen.select_rows(an, probs, gen.rules_for(5.4), .7)}
    low = {r.time for r in gen.select_rows(an, probs, gen.rules_for(5.4), .4)}
    assert high <= low, (high, low)


def test_zero_probability_support_does_not_create_rows():
    an, probs = analysis_and_probs([1000., 1500., 2000.])
    probs["count"][:] = [1, 0, 0, 0, 0]
    assert not gen.select_rows(an, probs, gen.rules_for(5.4), 0.0)


@pytest.mark.parametrize("bpm,beats", [(123.456789, 200000), (100.12349, 2000), (174.0004, 400),
                                      (120.00359, 400.99), (150.25, 900), (119.999, 500)])
def test_timing_fallback_fractional_span_and_valid_simplifications(bpm, beats):
    p = 60000 / bpm
    chosen = timing.snap_bpm(p, beats)
    budget = 6 if chosen == round(chosen) else 2
    assert max(abs(60000 / chosen - p), abs(round(60000 / chosen, 6) - p)) * beats <= budget
    if bpm == 174.0004:
        assert chosen == 174  # retain safe integer preference


def test_unrepresentable_export_precision_fails_closed():
    # Deliberately beyond ordinary song duration: silently over-budget is never allowed.
    with pytest.raises(ValueError, match="precision"):
        timing.snap_bpm(499.9999995, 20_000_000)


def test_peaked_activation_phase_and_downbeat_control():
    period, phase = 60000 / 120.012, 1000.37
    o = phase + np.arange(401) * period
    act = np.zeros(int(o[-1] / 20) + 10)
    down = act.copy()
    act[np.rint(o / 20).astype(int)] = 1
    down[np.rint(o[::4] / 20).astype(int)] = 1
    s = timing._finish(timing.Segment(o[0], o[-1], period, phase), o, np.ones(len(o)), act, down, 4)
    assert s.period == pytest.approx(period, abs=1e-9)
    assert s.phase == pytest.approx(phase, abs=1e-8)
    reds = timing.to_red_lines([s], 4, phase)
    assert abs((reds[0].time - s.downbeat) / (4 * period) - round((reds[0].time - s.downbeat) / (4 * period))) < 1e-9


def test_accepted_phase_change_survives_timing_finalization(monkeypatch):
    # Inject only the already-accepted segmentation/search result; finish/redline/export are real.
    first = np.arange(1000., 20000., 500.)
    second = np.arange(20009., 39009., 500.)
    o = np.r_[first, second]
    act = np.full(2100, -8.)
    down = act.copy()
    act[np.rint(o / 20).astype(int)] = 6
    down[np.rint(o[::4] / 20).astype(int)] = 6
    monkeypatch.setattr(timing, "pick_beats", lambda a: o)
    monkeypatch.setattr(timing, "onset_peaks", lambda env: (o, np.ones(len(o))))
    monkeypatch.setattr(timing, "search_period", lambda *a: (500., 1000., 1.))
    monkeypatch.setattr(timing, "find_segments", lambda *a: [timing.Segment(1000., 20000., 500., 1000.),
                                                            timing.Segment(20000., 39009., 500., 1009.)])
    res = timing.estimate_timing(None, act, down)
    assert res.kind == "changes" and len(res.red_lines) == 2
    assert res.red_lines[0].beat_ms == res.red_lines[1].beat_ms == 500.
    assert (res.red_lines[1].time - res.red_lines[0].time) % 500 == pytest.approx(9.)
    notes = [Note(1500., 0), Note(22009., 1)]
    chart = Chart(notes, res.red_lines)
    for text in (gen.chart_to_osu_text(chart), gen.build_beatmap(SimpleNamespace(), chart, "unused", "x", "x", kiai=[]).to_osu()):
        back = parse_osu(text)
        assert back is not None and len(back.red_lines) == 2
        assert (back.red_lines[1].time - back.red_lines[0].time) % 500 == 9.
        assert verify_chart(back).ok
    assert timing._near_global(timing.Segment(0, 10000, 500, 1125), timing.Segment(0, 10000, 500, 1000))


@pytest.mark.parametrize("distance,valid", [(1, True), (2, True), (3, False)])
def test_triplet_passage_distance_and_export(distance, valid):
    reds = [RedLine(1000.37, 60000 / 179.9)]
    p = reds[0].beat_ms
    chart = Chart([Note(reds[0].time + p / 3, 0), Note(reds[0].time + (distance + 1 / 3) * p, 1)], reds)
    assert verify_chart(chart).ok == valid
    assert verify_chart(parse_osu(gen.chart_to_osu_text(chart))).ok == valid


@pytest.mark.parametrize("new_period", [500., 400.])
def test_beat_aligned_redline_preserves_triplet_passage(new_period):
    reds = [RedLine(1000., 500.), RedLine(2000., new_period)]
    times = [1500 + 500 / 3, 2000 + new_period / 3]
    an, probs = analysis_and_probs(times, reds=reds)
    rows = gen.select_rows(an, probs, gen.rules_for(5.4), .5)
    assert len(rows) == 2
    chart = Chart([Note(t, i) for i, t in enumerate(times)], reds)
    assert verify_chart(chart).ok
    assert verify_chart(parse_osu(gen.chart_to_osu_text(chart))).ok


def test_phase_discontinuity_cannot_supply_a_triplet_partner():
    reds = [RedLine(1000., 500.), RedLine(2009., 500.)]
    times = [1500 + 500 / 3, 2009 + 500 / 3]
    an, probs = analysis_and_probs(times, reds=reds)
    assert not gen.select_rows(an, probs, gen.rules_for(5.4), .5)
    assert not verify_chart(Chart([Note(t, i) for i, t in enumerate(times)], reds)).ok


def test_random_threshold_monotonicity_with_section_plan():
    from autoosu.mania4k.planner import Plan, Section
    from autoosu.mania4k.structure import TYPES
    reds = [RedLine(1000., 500.), RedLine(5000., 400.)]
    grid = build_grid(reds, 1000, 10000)
    an = gen.SongAnalysis(Path("unused"), 11000, None, timing.TimingResult(reds, [], 1, "synthetic"),
                          grid, np.ones(len(grid), bool))
    plan = Plan(3, [Section(1000, 5000, 2, 0, 1, TYPES.index("ln"), -.6),
                    Section(5000, 10000, 3, 1, 3, TYPES.index("chordjack"), .5)])
    rng = np.random.default_rng(1003)
    for _ in range(20):
        p = rng.uniform(0, 1, len(grid))
        p[rng.random(len(grid)) < .2] = 0
        count = np.zeros((len(grid), 5))
        count[:, 0], count[:, 1] = 1 - p, p
        probs = {"count": count, "ln": rng.random(len(grid)), "lnlen": np.ones((len(grid), 8))}
        an.support = rng.random(len(grid)) < .5
        prev = set()
        for theta in (.9, .7, .5, .3, .1, 0.):
            rows = gen.select_rows(an, probs, gen.rules_for(5.4), theta, plan=plan)
            current = {r.tick for r in rows}
            assert prev <= current
            assert all(an.support[i] and p[i] > 0 for i in current)
            assert all(r.style == plan.sections[int(plan.section_of([r.time])[0])].type for r in rows)
            prev = current


def test_export_preserves_ln_and_sv_with_tempo_change():
    reds = [RedLine(1000.37, 500.), RedLine(61000.37, 400.), RedLine(71000.37, 500.)]
    chart = Chart([Note(1000.37, 0, 2000.37), Note(61000.37, 1, 61800.37), Note(111000.37, 2)], reds)
    beatmap = gen.build_beatmap(SimpleNamespace(), chart, "unused", "x", "x", kiai=[(61400.37, 62200.37)])
    back = parse_osu(beatmap.to_osu())
    assert back is not None and verify_chart(back).ok
    assert sum(n.is_hold for n in back.notes) == 2
    assert [n.end - n.time for n in back.notes if n.is_hold] == [1000., 800.]
    greens = [tp for tp in beatmap.timing_points if not tp.uninherited]
    assert all(tp.beat_length == -125 for tp in greens if 60976 <= tp.time < 70976)
    assert gen.normalized_sv(reds, 111000.37) == [1., .8, 1.]
