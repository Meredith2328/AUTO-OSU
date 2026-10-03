"""Load actual pure functions for synthetic tests without model/audio imports.

AST selection keeps original function bodies. No checkpoint, tracker, torch module,
audio decoder or star engine is loaded. Explicit stubs exercise only search logic.
"""
from __future__ import annotations

import ast
import bisect
import dataclasses
import math
import sys
import types
import typing
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def execute(path, ns, names=None, constants=()):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and (names is None or node.name in names):
            nodes.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in constants for target in targets):
                nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), ns)


def namespace(*, original=False):
    module = types.ModuleType("mania4k_contract_" + str(len(sys.modules)))
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns["__package__"] = "autoosu.mania4k"  # Permit the standard-library beatmap exporter only.
    ns.update({key: value for key, value in vars(typing).items() if not key.startswith("__")})
    ns.update(np=np, math=math, bisect=bisect, dataclass=dataclasses.dataclass,
              field=dataclasses.field, Path=Path)
    execute(ROOT / "autoosu/mania4k/chart.py", ns, constants=("KEYS", "LANE_X"))
    execute(ROOT / "autoosu/mania4k/features.py", ns, {"Grid", "build_grid"},
            ("POSITIONS", "POS_DIV", "DIV_CLASSES"))
    execute(ROOT / "autoosu/mania4k/timing.py", ns, constants=("TRACKER_FPS", "SNAP_TOL_MS"))
    execute(ROOT / "autoosu/mania4k/generate.py", ns,
            {"Rules", "Row", "ChartReport", "rules_for", "select_rows", "consistent_snaps",
             "generate_chart", "chart_to_osu_text"}, ("OSU_SHIFT_MS", "DIFFICULTIES"))
    execute(ROOT / "autoosu/mania4k/verify.py", ns)
    # Fixtures use p(LN)=0; the sole length bin is never used for an actual LN.
    ns["LN_BINS"] = [1.0]
    if original:
        execute(ROOT / "tests/fixtures/mania4k_original_functions.py", ns)
    return ns


def selection_input(ns):
    times = np.array([1200., 1300., 1400., 1800., 1900., 2000.])
    fractions = [1/3, .5, 2/3, 1/3, .5, 2/3]
    pos = np.array([min(range(len(ns["POSITIONS"])), key=lambda i: abs(ns["POSITIONS"][i]-f)) for f in fractions])
    div = np.array([ns["DIV_CLASSES"].index(ns["POS_DIV"][i]) for i in pos])
    grid = types.SimpleNamespace(times=times, pos=pos, div=div, beat_ms=np.full(6, 600.),
                                 beat=np.array(fractions)+np.array([0, 0, 0, 1, 1, 1]))
    p = np.array([.6, .9, .6, .6, .9, .6])
    count = np.zeros((6, 5)); count[:, 0] = 1-p; count[:, 1] = p
    return (types.SimpleNamespace(grid=grid, support=np.ones(6, dtype=bool)),
            dict(count=count, ln=np.zeros(6), lnlen=np.ones((6, 1))))


def search_response(ns, response, target=4., tolerance=.08):
    """Synthetic response units only: no real stars or chart/model inference."""
    calls = []
    ns["note_probabilities"] = lambda *args: {}
    ns["select_rows"] = lambda an, probs, rules, theta, boost: [types.SimpleNamespace(theta=theta, boost=boost)]
    ns["assign_lanes"] = lambda an, rows, *args: [types.SimpleNamespace(
        time=1000., is_hold=False, audit_theta=rows[0].theta, audit_boost=rows[0].boost)]
    ns["chart_to_osu_text"] = lambda chart: chart

    def score(chart):
        note = chart.notes[0]
        value = float(response(note.audit_theta, note.audit_boost))
        calls.append((note.audit_theta, note.audit_boost, value))
        return value

    ns["star_rating"] = score
    an = types.SimpleNamespace(timing=types.SimpleNamespace(red_lines=[]))
    _, report = ns["generate_chart"](an, "Insane", target, models=(object(), object()), tolerance=tolerance)
    return report, calls
