import numpy as np
import pytest

from autoosu.beatmap import PLAYFIELD_H, PLAYFIELD_W
from autoosu.sliderpath import fit_slider, path_length, path_points, position_at


def inside(pts, margin=0.0):
    return (pts[:, 0].min() >= margin and pts[:, 0].max() <= PLAYFIELD_W - margin
            and pts[:, 1].min() >= margin and pts[:, 1].max() <= PLAYFIELD_H - margin)


def test_lengths():
    assert path_length("L", [(0, 0), (100, 0)]) == pytest.approx(100)
    # quarter circle of radius 100 through (100,100)
    arc = path_length("P", [(0, 0), (100, 100), (200, 0)])
    assert arc == pytest.approx(np.pi * 100, rel=1e-3)
    # straight bezier degenerates to the chord
    assert path_length("B", [(0, 0), (50, 0), (100, 0)]) == pytest.approx(100, rel=1e-3)


def test_position_at_extends_past_end():
    p = position_at("L", [(0, 0), (100, 0)], 150)
    assert p[0] == pytest.approx(150) and p[1] == pytest.approx(0)


@pytest.mark.parametrize("seed", range(40))
def test_fit_keeps_path_inside(seed):
    rng = np.random.default_rng(seed)
    head = rng.uniform([0, 0], [PLAYFIELD_W, PLAYFIELD_H])
    kind = ["L", "B", "P"][seed % 3]
    n_mid = {"L": 0, "B": 1 + seed % 2, "P": 1}[kind]
    anchors = [head] + [head + rng.normal(0, 25, 2) for _ in range(n_mid + 1)]
    required = float(rng.uniform(40, 360))
    f = fit_slider(kind, anchors, required, margin=2.0)
    pts = path_points(kind, f.anchors)
    assert inside(pts[1:], 1.0), (seed, f)      # the head is given and may sit on the edge
    assert f.anchors[0] == (int(round(head[0])), int(round(head[1])))
    if f.sv_scale == 1.0:
        assert f.length == pytest.approx(required)
        assert path_length(kind, f.anchors) == pytest.approx(required, rel=0.05, abs=6)
    else:
        assert 0 < f.sv_scale < 1
        assert f.length == pytest.approx(required * f.sv_scale, rel=1e-3)


def test_fit_shortens_when_no_room():
    # head in the exact corner with a straight slider longer than the diagonal: must shrink + lower SV
    f = fit_slider("L", [(0, 0), (10, 0)], 2000, margin=0.0)
    assert f.sv_scale < 1 and f.length <= 2000
    assert inside(path_points("L", f.anchors))
