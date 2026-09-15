"""osu! slider paths: sample the curve, measure it, and fit a slider shape into the playfield.

Curve types follow the .osu format: L linear, P perfect circle (3 points, else bezier), B bezier
(a repeated anchor = red anchor starts a new segment), C catmull. Lengths here are the natural
length of the drawn path; osu! trims or extends the path to the length written in the file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import PLAYFIELD_H, PLAYFIELD_W

Point = Tuple[float, float]


def _bezier(cps: np.ndarray) -> np.ndarray:
    k = len(cps)
    if k <= 2:
        return cps.copy()
    n = int(np.clip(np.sum(np.linalg.norm(np.diff(cps, axis=0), axis=1)) / 3, 16, 400))
    t = np.linspace(0.0, 1.0, n)
    pts = np.repeat(cps[None], n, axis=0)           # de Casteljau, vectorised over t
    for _ in range(1, k):
        pts = (1 - t)[:, None, None] * pts[:, :-1] + t[:, None, None] * pts[:, 1:]
    return pts[:, 0]


def _perfect(cps: np.ndarray) -> np.ndarray:
    (ax, ay), (bx, by), (cx, cy) = cps
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return _bezier(cps)
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    t0, t1, t2 = (math.atan2(py - uy, px - ux) for px, py in cps)

    def ccw(a: float, b: float) -> float:
        return (b - a) % (2 * math.pi)

    sweep = ccw(t0, t2) if ccw(t0, t1) < ccw(t0, t2) else -ccw(t2, t0)
    n = int(np.clip(abs(sweep) * r / 3, 16, 400))
    ang = t0 + np.linspace(0.0, sweep, n)
    return np.stack([ux + r * np.cos(ang), uy + r * np.sin(ang)], axis=1)


def _catmull(cps: np.ndarray) -> np.ndarray:
    out = []
    n = len(cps)
    for i in range(n - 1):
        p0 = cps[i - 1] if i > 0 else cps[i]
        p1, p2 = cps[i], cps[i + 1]
        p3 = cps[i + 2] if i + 2 < n else p2
        t = np.linspace(0.0, 1.0, 40)[:, None]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out) if out else cps.copy()


def path_points(curve_type: str, anchors: Sequence[Point]) -> np.ndarray:
    """Sample the whole path (head first) as an (n, 2) array."""
    cps = np.asarray(anchors, dtype=float)
    if len(cps) < 2:
        return cps
    if curve_type == "L":
        return cps
    if curve_type == "P" and len(cps) == 3:
        return _perfect(cps)
    if curve_type == "C":
        return _catmull(cps)
    # bezier: split at repeated anchors
    pieces, start = [], 0
    for i in range(1, len(cps)):
        if np.array_equal(cps[i], cps[i - 1]):
            pieces.append(_bezier(cps[start:i]))
            start = i
    pieces.append(_bezier(cps[start:]))
    return np.vstack(pieces)


def path_length(curve_type: str, anchors: Sequence[Point]) -> float:
    pts = path_points(curve_type, anchors)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def position_at(curve_type: str, anchors: Sequence[Point], distance: float) -> np.ndarray:
    """Point at the given distance along the path; extends the last segment like osu! does."""
    pts = path_points(curve_type, anchors)
    if len(pts) < 2:
        return pts[0] if len(pts) else np.zeros(2)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if distance >= cum[-1]:
        d = pts[-1] - pts[-2]
        nrm = float(np.linalg.norm(d))
        return pts[-1] + (d / nrm * (distance - cum[-1]) if nrm > 0 else 0.0)
    i = int(np.searchsorted(cum, distance, side="right")) - 1
    i = max(0, min(i, len(seg) - 1))
    w = (distance - cum[i]) / seg[i] if seg[i] > 0 else 0.0
    return pts[i] + (pts[i + 1] - pts[i]) * w


@dataclass
class FittedSlider:
    anchors: List[Tuple[int, int]]     # all control points including the head
    length: float                      # pixel length written to the file (one slide)
    sv_scale: float                    # 1.0 = as requested; < 1 when the slider had to be shortened
    end: Tuple[float, float]           # position at `length` along the path


def _inside(pts: np.ndarray, margin: float) -> bool:
    """Every path point except the head (which is given and may sit on the edge) is inside the field."""
    p = pts[1:] if len(pts) > 1 else pts
    return bool(np.all(p[:, 0] >= margin) and np.all(p[:, 0] <= PLAYFIELD_W - margin)
                and np.all(p[:, 1] >= margin) and np.all(p[:, 1] <= PLAYFIELD_H - margin))


def _max_scale(head: np.ndarray, pts: np.ndarray, margin: float) -> float:
    """Largest s such that head + s * (pts - head) stays inside the playfield."""
    v = pts - head
    lim = np.full(len(v), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis, size in ((0, PLAYFIELD_W), (1, PLAYFIELD_H)):
            pos = v[:, axis] > 1e-9
            neg = v[:, axis] < -1e-9
            lim[pos] = np.minimum(lim[pos], (size - margin - head[axis]) / v[pos, axis])
            lim[neg] = np.minimum(lim[neg], (margin - head[axis]) / v[neg, axis])
    s = float(lim.min())
    return max(0.0, s if np.isfinite(s) else 0.0)


def _rot(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


_ROTATIONS = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 150, -150, 180]


def fit_slider(curve_type: str, anchors: Sequence[Point], required_length: float, margin: float = 2.0,
               max_stretch: float = 1000.0) -> FittedSlider:
    """Scale the anchors about the head so the path is `required_length` long and stays on screen.

    Order of preference: plain scaling; mirroring the bend; rotating the shape about the head
    (small angles first); finally the largest in-bounds scale with a reduced slider velocity.
    """
    cps = np.asarray(anchors, dtype=float)
    head = cps[0].copy()
    if len(cps) < 2 or required_length <= 0:
        return FittedSlider([(int(round(x)), int(round(y))) for x, y in cps], required_length, 1.0,
                            (float(head[0]), float(head[1])))
    natural = path_length(curve_type, cps)
    if natural < 1e-6:
        cps = np.vstack([head, head + np.array([1.0, 0.0])])
        curve_type, natural = "L", 1.0
    scale = min(required_length / natural, max_stretch)

    rel = cps - head
    chord = rel[-1]
    ang0 = math.degrees(math.atan2(chord[1], chord[0])) if np.linalg.norm(chord) > 1e-6 else 0.0

    def variant(mirror: bool, deg: float) -> np.ndarray:
        r = rel.copy()
        if mirror:   # reflect across the head -> last anchor chord: keeps the end, flips the bend
            r = r @ _rot(-ang0).T
            r[:, 1] *= -1
            r = r @ _rot(ang0).T
        return r @ _rot(deg).T

    best: Optional[Tuple[float, np.ndarray, float]] = None    # cost, control points, scale used
    for deg in _ROTATIONS:
        for mirror in (False, True):
            r = variant(mirror, deg)
            full = head + r * scale
            pts = path_points(curve_type, full)
            cost = abs(deg) / 60.0 + (0.3 if mirror else 0.0)
            if _inside(pts, margin):
                if best is None or best[2] < scale or cost < best[0]:
                    best = (cost, full, scale)
                continue
            # not enough room at full size: remember the biggest in-bounds version as a fallback
            s = min(scale, _max_scale(head, path_points(curve_type, head + r), margin))
            short = 3.0 * (1.0 - s / scale) + cost
            if best is None or (best[2] < scale and short < best[0]):
                best = (short, head + r * s, s)
        if best is not None and best[2] >= scale and best[0] <= abs(deg) / 60.0:
            break     # nothing later can beat an in-bounds candidate at this rotation cost
    assert best is not None
    _, full, s_used = best
    r_best = (full - head) / s_used if s_used > 0 else rel
    rounded = np.round(full)
    if s_used >= scale * 0.999:
        # integer anchors change the drawn length a little (a lot for flat perfect-circle arcs):
        # rescale a few times so the drawn path really is the required length
        s = s_used
        for _ in range(4):
            drawn = path_length(curve_type, rounded)
            if drawn < 1e-6 or abs(drawn - required_length) <= 0.01 * required_length:
                break
            s2 = s * required_length / drawn
            cand = np.round(head + r_best * s2)
            if not _inside(path_points(curve_type, cand), margin):
                break
            s, rounded = s2, cand
        length, sv_scale = required_length, 1.0
    else:
        length = max(1.0, natural * s_used)
        sv_scale = length / required_length
    end = position_at(curve_type, rounded, length)
    return FittedSlider([(int(x), int(y)) for x, y in rounded], length, sv_scale, (float(end[0]), float(end[1])))
