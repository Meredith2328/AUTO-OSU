"""Pinned before/after pure-function regressions; all inputs are synthetic.

Search responses are controlled units, NOT measured osu! star ratings.
"""
import numpy as np
import pytest

from mania4k_source import namespace, search_response, selection_input


def finish_fixture(ns):
    period = 60000/179.9
    onsets = 1000 + np.arange(97)*period
    act = np.ones(int((onsets[-1]+2000)/1000*ns["TRACKER_FPS"])+1)
    seg = ns["Segment"](1000., onsets[-1]+period/2, period, 1000., 1.)
    finished = ns["_finish"](seg, onsets, np.ones(97), act, act, 4)
    signed = (onsets-finished.phase)/finished.period
    errors = (signed-np.round(signed))*finished.period
    return finished, float(np.max(np.abs(errors)))


def test_noninteger_tempo_rounding_preserves_the_measured_grid():
    before, old_error = finish_fixture(namespace(original=True))
    after, new_error = finish_fixture(namespace())
    assert before.bpm == 180. and old_error > 12.
    assert after.bpm == pytest.approx(179.9, abs=1e-6)
    assert new_error < 1e-6


@pytest.mark.parametrize("bpm,beats", [(128.004,64),(174.03,200),(179.9,96),(210.1,400)])
def test_bpm_simplification_stays_within_the_declared_half_segment_budget(bpm, beats):
    ns = namespace()
    period = 60000/bpm
    snapped = ns["snap_bpm"](period, beats)
    assert abs(60000/snapped-period)*beats/2 <= 2.+1e-6


def fold_fixture(ns, phase=9.):
    onsets = np.concatenate([1000+np.arange(40)*500, 21000+phase+np.arange(100)*500])
    glob = ns["Segment"](1000., onsets[-1]+500, 500., 1000.)
    segs = [ns["Segment"](1000.,21000.,500.,1000.),
            ns["Segment"](21000.,onsets[-1]+500,500.,1000.+phase)]
    # Actual post-finish folding helpers on already-identified segments.
    folded = [ns["_copy_global"](s,glob) if ns["_near_global"](s,glob) else s for s in segs]
    folded = ns["_merge_exact"](folded)
    return folded, ns["_total_rate"](onsets,np.ones(140),folded)


def test_accepted_phase_change_is_retained():
    old, old_rate = fold_fixture(namespace(original=True))
    new, new_rate = fold_fixture(namespace())
    assert len(old)==1 and old_rate==pytest.approx(40/140)
    assert len(new)==2 and new_rate==1.


def test_identical_segments_still_merge():
    segs, rate = fold_fixture(namespace(), phase=0.)
    assert len(segs)==1 and rate==1.


def test_lower_threshold_keeps_existing_supported_straight_rows():
    results=[]
    for original in [True,False]:
        ns=namespace(original=original); an,probs=selection_input(ns)
        rules=ns["rules_for"](4.3)  # Actual unmodified Insane rule values.
        high={r.tick for r in ns["select_rows"](an,probs,rules,.8)}
        low={r.tick for r in ns["select_rows"](an,probs,rules,.5)}
        results.append((high,low))
    assert results[0]==({1,4},{0,2,3,5})
    assert results[1]==({1,4},{0,1,2,3,4,5})


def test_default_export_coordinate_transform_stays_at_24ms():
    ns=namespace(); chart=ns["Chart"]([ns["Note"](1000.,0),ns["Note"](1300.,1,1600.)],
                                      [ns["RedLine"](1000.,600.)])
    parsed=ns["parse_osu"](ns["chart_to_osu_text"](chart))
    assert [(n.time,n.end) for n in parsed.notes]==[(976.,0.),(1276.,1576.)]
    assert parsed.red_lines[0].time==976.


def test_retained_mixed_rows_satisfy_the_existing_hard_verifier():
    ns=namespace(); an,probs=selection_input(ns); rules=ns["rules_for"](4.3)
    rows=ns["select_rows"](an,probs,rules,.5)
    chart=ns["Chart"]([ns["Note"](r.time,i%4) for i,r in enumerate(rows)],
                       [ns["RedLine"](1000.,600.)])
    check=ns["verify_chart"](chart,None,rules)
    assert check.ok, check.problems
    assert len(chart.notes)==6


@pytest.mark.parametrize("name,response,target,tolerance,old_hit", [
    ("nonmonotonic", lambda t,b: 4. if .68<=t<=.82 else (3. if t<=.5 else 2.), 4.,.08,False),
    ("midpoint", lambda t,b: 8*(1-t), 4.,.08,True),
    ("off_midpoint", lambda t,b: 8*(1-t), 4.1,.08,True),
    ("fine_monotonic", lambda t,b: 8*(1-t), 4.12345,.001,True),
    ("later_boost", lambda t,b: 4.2 if b==1. else 4., 4.,.08,False),
])
def test_bounded_search_retains_refinement_and_explores_missed_targets(name,response,target,tolerance,old_hit):
    before,old_calls=search_response(namespace(original=True),response,target,tolerance)
    after,calls=search_response(namespace(),response,target,tolerance)
    assert (abs(before.stars-target)<=tolerance)==old_hit
    assert abs(after.stars-target)<=tolerance
    assert len(calls)<=48
    if old_hit:
        assert calls==old_calls  # Original successful refinement path remains intact.


def test_unreachable_target_returns_best_without_a_false_hit():
    report,calls=search_response(namespace(),lambda t,b:3.)
    assert report.stars==3. and abs(report.stars-report.target_stars)>.08
    assert len(calls)<=48
