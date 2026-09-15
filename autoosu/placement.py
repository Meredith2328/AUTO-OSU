"""Placement layer: give rhythm events x/y positions, slider shapes and flow.

Spacing follows the section intensity (verses tight, choruses wide), strong drum hits in loud
sections become jumps, and every combo picks one movement style (curve / zigzag / square /
back-and-forth) so patterns look designed instead of random.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import PLAYFIELD_H, PLAYFIELD_W, Circle, HitObject, Slider, Spinner
from .difficulty import DifficultyPreset
from .rhythm import RhythmEvent
from .timing import Timing

CENTER = np.array([PLAYFIELD_W / 2.0, PLAYFIELD_H / 2.0])
DEVIATIONS = [0, 15, -15, 30, -30, 45, -45, 60, -60, 80, -80, 100, -100, 120, -120, 150, -150, 180]
SvSection = Tuple[int, int, float]   # start ms, end ms, slider velocity multiplier


def circle_radius(cs: float) -> float:
    return 54.4 - 4.48 * cs


def _unit(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)])


def _angle(v: np.ndarray) -> float:
    return math.atan2(float(v[1]), float(v[0]))


def _bezier_len(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 200) -> float:
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _circle_arc_len(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Length of the circular arc from p0 to p2 passing through p1 (osu "P" slider)."""
    ax, ay = map(float, p0)
    bx, by = map(float, p1)
    cx, cy = map(float, p2)
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6:
        return float(np.linalg.norm(p2 - p0))
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    center = np.array([ux, uy])
    r = float(np.linalg.norm(p0 - center))
    a0, a1, a2_ = (_angle(p - center) for p in (p0, p1, p2))
    ccw = lambda a, b: (b - a) % (2 * math.pi)
    sweep = ccw(a0, a2_) if ccw(a0, a1) < ccw(a0, a2_) else ccw(a2_, a0)
    return r * sweep


def _unit_bezier_arc(c: float) -> float:
    return _bezier_len(np.array([0.0, 0.0]), np.array([0.5, c]), np.array([1.0, 0.0]))


def _unit_circle_arc(c: float) -> Tuple[float, float]:
    r = (0.25 + c * c) / (2 * c)
    theta = 2 * math.asin(min(1.0, 0.5 / r))
    return r * theta, theta


@dataclass
class Shape:
    curve_type: str
    points: List[np.ndarray]     # anchors after the head
    end: np.ndarray
    end_dir: float
    length: float


def _make_shape(start: np.ndarray, ang: float, length: float, kind: str, c: float, side: float) -> Shape:
    u = _unit(ang)
    nrm = np.array([-u[1], u[0]]) * side
    if kind == "L":
        end = np.round(start + u * length)
        return Shape("L", [end], end, ang, float(np.linalg.norm(end - start)))
    if kind == "B":
        chord = length / _unit_bezier_arc(c)
        p1 = np.round(start + chord * (0.5 * u + c * nrm))
        p2 = np.round(start + chord * u)
        return Shape("B", [p1, p2], p2, _angle(p2 - p1), _bezier_len(start, p1, p2))
    arc, theta = _unit_circle_arc(c)
    chord = length / arc
    mid = np.round(start + chord * (0.5 * u + c * nrm))
    p2 = np.round(start + chord * u)
    return Shape("P", [mid, p2], p2, ang - side * theta / 2, _circle_arc_len(start, mid, p2))


class Placer:
    def __init__(self, preset: DifficultyPreset, timing: Timing, rng: np.random.Generator,
                 sv_sections: Sequence[SvSection] = ()):
        self.p, self.t, self.rng = preset, timing, rng
        self.sv_sections = list(sv_sections)
        self.r = circle_radius(preset.cs)
        self.diam = 2 * self.r
        self.margin = self.r + 4
        self.pos = np.round(np.array([rng.uniform(160, 352), rng.uniform(120, 264)]))
        self.dir = float(rng.uniform(-math.pi, math.pi))
        self.turn_sign = 1.0
        self.curve_sign = float(rng.choice([-1.0, 1.0]))
        self.stream_curve = self.curve_sign * float(rng.uniform(math.radians(8), math.radians(24)))
        self.style = "curve"
        self.style_mag = math.radians(30)
        self.style_sign = 1.0
        self.recent: List[Tuple[np.ndarray, int]] = []
        self.last_end_time: Optional[int] = None

    # -- helpers -------------------------------------------------------------
    def sv_at(self, time: int) -> float:
        for start, end, sv in self.sv_sections:
            if start <= time < end:
                return sv
        return 1.0

    def in_bounds(self, pt: np.ndarray) -> bool:
        return (self.margin <= pt[0] <= PLAYFIELD_W - self.margin
                and self.margin <= pt[1] <= PLAYFIELD_H - self.margin)

    def _dt(self, time: int) -> Optional[float]:
        if self.last_end_time is None:
            return None
        return max(0.05, (time - self.last_end_time) / self.t.beat_length)

    def _pick_style(self, ev: RhythmEvent) -> None:
        rng = self.rng
        if self.p.jump_prob <= 0.05:
            styles, w = ["curve", "zigzag"], [0.65, 0.35]
        elif ev.intensity < 0.5:
            styles, w = ["curve", "zigzag", "square"], [0.5, 0.35, 0.15]
        else:
            styles, w = ["zigzag", "square", "backforth", "curve"], [0.4, 0.25, 0.2, 0.15]
        self.style = str(rng.choice(styles, p=w))
        self.style_sign = float(rng.choice([-1.0, 1.0]))
        self.style_mag = {
            "curve": float(rng.uniform(math.radians(20), math.radians(40))),
            "zigzag": float(rng.uniform(math.radians(75), math.radians(120))),
            "square": math.radians(90),
            "backforth": math.pi,
        }[self.style]

    def _distance(self, dt: float, ev: RhythmEvent) -> Tuple[float, bool]:
        base = self.p.slider_multiplier * 100.0 * self.p.spacing * (0.7 + 0.6 * ev.intensity)
        if dt <= 0.3:
            dist = float(np.clip(base * dt, 0.35 * self.diam, 1.1 * self.diam))
        elif dt < 1.0:
            dist = base * dt
        else:
            dist = base * (1.0 + 0.6 * (min(dt, 3.0) - 1.0))
        jump = (dt >= 0.5 and ev.emphasis >= 0.7 and ev.intensity >= 0.55
                and self.rng.random() < self.p.jump_prob)
        if jump:
            dist *= 1.6
        return float(np.clip(dist, 0.3 * self.diam, 340.0)), jump

    def _turn(self, dt: float, jump: bool) -> float:
        rng = self.rng
        if dt <= 0.3:
            if rng.random() < 0.06:
                self.stream_curve = -self.stream_curve
            return self.stream_curve
        if jump:
            self.turn_sign = -self.turn_sign
            return self.turn_sign * float(rng.uniform(math.radians(140), math.radians(180)))
        if self.style == "curve":
            if rng.random() < 0.05:
                self.style_sign = -self.style_sign
            return self.style_sign * self.style_mag
        if self.style == "zigzag":
            self.turn_sign = -self.turn_sign
            return self.turn_sign * self.style_mag
        if self.style == "square":
            return self.style_sign * self.style_mag
        return math.pi   # backforth

    def _overlap_cost(self, cand: np.ndarray, time: int) -> float:
        cost = 0.0
        lim = 0.9 * self.diam
        recent = self.recent[:-1] if self.style == "backforth" else self.recent
        for pt, t in recent:
            d = float(np.linalg.norm(cand - pt))
            if d < lim and (time - t) / self.t.beat_length > 0.3:
                cost += 1.0 - d / lim
        return cost

    def _center_cost(self, cand: np.ndarray) -> float:
        return 0.25 * float(np.linalg.norm(cand - CENTER)) / 300.0

    def _choose_head(self, ev: RhythmEvent) -> Tuple[np.ndarray, float]:
        dt = self._dt(ev.time)
        if dt is None:
            return self.pos, self.dir
        dist, jump = self._distance(dt, ev)
        turn = self._turn(dt, jump)
        best = None
        for shrink in (1.0, 0.8, 0.6, 0.45):
            for dev in DEVIATIONS:
                ang = self.dir + turn + math.radians(dev)
                cand = np.round(self.pos + dist * shrink * _unit(ang))
                if not self.in_bounds(cand):
                    continue
                cost = (abs(dev) / 180.0 + 2.0 * (1.0 - shrink)
                        + self._overlap_cost(cand, ev.time) + self._center_cost(cand))
                if best is None or cost < best[0]:
                    best = (cost, cand, ang)
            if best is not None:
                break
        if best is None:
            ang = _angle(CENTER - self.pos)
            cand = np.round(self.pos + min(dist, float(np.linalg.norm(CENTER - self.pos))) * _unit(ang))
            best = (0.0, cand, ang)
        return best[1], best[2]

    def _commit(self, end: np.ndarray, end_dir: float, end_time: int, *marks: Tuple[np.ndarray, int]) -> None:
        self.pos, self.dir, self.last_end_time = end, end_dir, end_time
        self.recent.extend(marks)
        self.recent = self.recent[-6:]

    # -- objects -------------------------------------------------------------
    def place_circle(self, ev: RhythmEvent) -> Circle:
        pos, ang = self._choose_head(ev)
        self._commit(pos, ang, ev.time, (pos, ev.time))
        return Circle(int(pos[0]), int(pos[1]), ev.time, ev.new_combo, ev.hitsound)

    def place_spinner(self, ev: RhythmEvent) -> Spinner:
        self.recent = []
        self._commit(CENTER.copy(), float(self.rng.uniform(-math.pi, math.pi)), ev.end_time)
        return Spinner(int(CENTER[0]), int(CENTER[1]), ev.time, True, ev.hitsound, end=ev.end_time)

    def place_slider(self, ev: RhythmEvent) -> Slider:
        rng = self.rng
        head, head_dir = self._choose_head(ev)
        dur = ev.end_time - ev.time
        px_per_beat = self.p.slider_multiplier * 100.0 * self.sv_at(ev.time)
        length = max(1.0, px_per_beat * dur / self.t.beat_length / ev.repeats)

        if length < 1.3 * self.diam:
            kinds = ["L"]
        else:
            kinds = [str(rng.choice(["L", "B", "P"], p=[0.15, 0.5, 0.35]))]
        c = float(rng.uniform(0.12, 0.35)) * (0.8 if length > 250 else 1.0)
        flow = float(rng.uniform(-math.radians(30), math.radians(30)))

        best = None
        attempts = [(k, c) for k in kinds] + [("B", 0.5), ("B", 0.7), ("L", c)]
        for kind, cc in attempts:
            for dev in DEVIATIONS:
                for side in (self.curve_sign, -self.curve_sign):
                    ang = head_dir + flow + math.radians(dev)
                    shape = _make_shape(head, ang, length, kind, cc, side)
                    if not all(self.in_bounds(pt) for pt in [head, *shape.points]):
                        continue
                    cost = (abs(dev) / 180.0 + (0.3 if side != self.curve_sign else 0.0)
                            + self._overlap_cost(shape.end, ev.end_time) + self._center_cost(shape.end))
                    if best is None or cost < best[0]:
                        best = (cost, shape)
                    if kind == "L":
                        break
            if best is not None:
                break

        if best is None:
            # nothing fits: shorten the slider towards the centre (rare, changes its duration)
            ang = _angle(CENTER - head)
            room = max(20.0, float(np.linalg.norm(CENTER - head)) + min(CENTER) - self.margin - 10)
            shape = _make_shape(head, ang, min(length, room), "L", c, 1.0)
            length = shape.length
            dur = int(round(length / px_per_beat * self.t.beat_length * ev.repeats))
            ev.end_time = ev.time + dur
            ev.end_beat = self.t.beat_at(ev.end_time)
            best = (0.0, shape)
        shape = best[1]
        # the written length is the musical one; osu! trims/extends the drawn path to it, so the
        # end point may move a pixel or two from shape.end after anchor rounding (invisible in play)

        if ev.repeats % 2 == 0:
            end = head
            end_dir = shape.end_dir + math.pi if shape.curve_type == "L" else _angle(head - shape.points[0])
        else:
            end, end_dir = shape.end, shape.end_dir
        self.curve_sign = -self.curve_sign if rng.random() < 0.5 else self.curve_sign
        self._commit(end, end_dir, ev.end_time, (head, ev.time), (end, ev.end_time))
        return Slider(
            int(head[0]), int(head[1]), ev.time, ev.new_combo, ev.hitsound,
            curve_type=shape.curve_type,
            points=[(int(p[0]), int(p[1])) for p in shape.points],
            repeats=ev.repeats, length=length, duration=ev.end_time - ev.time,
            end_x=int(end[0]), end_y=int(end[1]),
        )

    def place(self, ev: RhythmEvent) -> HitObject:
        if ev.kind == "spinner":
            return self.place_spinner(ev)
        if ev.new_combo:
            self._pick_style(ev)
        if ev.kind == "slider":
            return self.place_slider(ev)
        return self.place_circle(ev)


def place(events: List[RhythmEvent], preset: DifficultyPreset, timing: Timing,
          rng: np.random.Generator, sv_sections: Sequence[SvSection] = ()) -> List[HitObject]:
    placer = Placer(preset, timing, rng, sv_sections)
    return [placer.place(ev) for ev in events]
