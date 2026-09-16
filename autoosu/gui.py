"""Desktop interface (customtkinter): pick a song, tick difficulties, press Generate.

Runs the pipeline in a worker thread and reports progress through a queue; all texts come from
`autoosu.i18n` so the window can switch between Chinese and English on the fly. Light and dark
looks share one palette taken from kanzei's OC: graphite black, amber-gold eyes, the electric-blue
cheek glow and warm back-light. A few small animations (gold pulse waves in the header, smooth progress,
a breathing Generate button) run on Tk timers and pause when the window is hidden.
"""
from __future__ import annotations

import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import __version__
from .difficulty import PRESETS
from .i18n import AUTHOR, language, memes, set_language, tr
from .models import MODELS, app_root, ensure_model, find_model

AUDIO_EXT = ("*.mp3", "*.ogg", "*.wav", "*.flac", "*.m4a", "*.aac", "*.wma", "*.opus")
QUALITY_STEPS = {"fast": 50, "normal": 100, "high": 200}
ASSETS = Path(__file__).resolve().parent / "assets"
AVATAR = ASSETS / "avatar.png"          # kanzei's OC; header avatar + window icon when present

Pair = Tuple[str, str]                  # (light, dark)
PALETTE: Dict[str, Pair] = {
    "bg": ("#f3f3f5", "#0f0f11"),        # neutral paper / graphite black
    "panel": ("#ffffff", "#18181b"),
    "inner": ("#e9e9ee", "#202024"),
    "border": ("#d6d6dc", "#2c2c32"),
    "text": ("#1c1c22", "#ecebe6"),
    "muted": ("#6f6f7a", "#8e8e98"),
    "gold": ("#d99a2b", "#e6a93c"),      # amber eyes
    "gold_hover": ("#bf8420", "#cf9530"),
    "cyan": ("#2b8fe6", "#4fb8ff"),      # cheek glow
    "warm": ("#e0a985", "#f2c9a0"),      # back-light on skin
    "warn": ("#b3541e", "#f0a060"),
}


def _hex(c: str) -> Tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def blend(a: str, b: str, t: float) -> str:
    """Colour between a (t=0) and b (t=1)."""
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * t)) for x, y in zip(_hex(a), _hex(b)))


def settings_path() -> Path:
    base = Path(os.environ.get("APPDATA") or Path.home())
    return base / "AUTO-OSU" / "settings.json"


def load_settings() -> Dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: Dict) -> None:
    try:
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def open_path(path: Path) -> None:
    """Open a file with its default app (an .osz opens in osu!) or a folder in Explorer."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass


def _avatar_image(size: int):
    """Round-cropped PIL image of assets/avatar.png, or None when the file is absent."""
    if not AVATAR.exists():
        return None
    try:
        from PIL import Image, ImageDraw

        img = Image.open(AVATAR).convert("RGBA")
        side = min(img.size)
        img = img.crop(((img.width - side) // 2, 0, (img.width - side) // 2 + side, side)).resize((size, size), Image.LANCZOS)
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).ellipse((0, 0, img.width - 1, img.height - 1), fill=255)
        img.putalpha(mask)
        return img
    except Exception:
        return None


def run_gui() -> int:
    import customtkinter as ctk

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
    except Exception:  # drag and drop is optional
        DND_FILES = TkinterDnD = None

    theme = ASSETS / "theme.json"
    ctk.set_default_color_theme(str(theme) if theme.exists() else "blue")
    ctk.set_appearance_mode("system")

    class Header:
        """Canvas banner with anti-aliased gold pulse waves, title, subtitle, avatar and the toggles.

        Rendering strategy (measured on a 240 Hz display):
        * idle: a seamless loop is pre-rendered once (PIL, 2x supersampled, in a background thread)
          and played back by swapping canvas images at the display refresh rate: ~0.1 ms per frame,
          memory capped by LOOP_BUDGET_MB (unique frames per second adapt to the budget);
        * generating: frames are rendered live (~1 ms each) so the crest can follow the progress,
          still paced to the refresh rate; a caption in osu! slang rides on the crest.
        Frames are paced by a background thread that sleeps precisely and posts a virtual event
        (Tk's own after() at 240 Hz burns ~25 % of a core just spinning); the main thread draws.
        Motion is time based, so a dropped frame never changes the speed."""

        N_LINES = 8
        N_POINTS = 64
        SS = 2                      # supersampling for anti-aliasing
        LOOP_SECONDS = 2.0          # the idle loop is exactly periodic over this
        LOOP_BUDGET_MB = 64         # Tk keeps 4 bytes per pixel per frame

        def __init__(self, app: "App", master) -> None:
            import tkinter.font as tkfont

            self.app = app
            try:   # DPI scaling customtkinter applies to its widgets (1.5 on a 150 % display)
                s = float(ctk.ScalingTracker.get_widget_scaling(app))
            except Exception:
                s = 1.0
            self.s = s
            self.h = int(124 * s)
            self.canvas = tk.Canvas(master, height=self.h, highlightthickness=0, bd=0, bg=app.c("bg"))
            self.canvas.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 0))
            # negative sizes = pixels, the same unit customtkinter uses for its own scaled fonts
            self.f_title = tkfont.Font(family=_ui_font(), size=-int(round(24 * s)), weight="bold")
            self.f_sub = tkfont.Font(family=_ui_font(), size=-int(round(13 * s)))
            self.refresh_hz = _display_refresh_hz()
            self.frame_dt = 1.0 / self.refresh_hz
            self.next_due = time.perf_counter()
            self.t0 = time.time()
            self.energy = 0.25          # 0.25 idle, 1 while generating, >1 for the finish flash
            self.energy_target = 0.25
            self.busy = False
            self.progress = 0.0
            self.crest = 0.68           # where the pulse sits (fraction of the width); follows progress when busy
            self.caption_at = 0.0
            self.caption_idx = -1
            self.celebrate_until = 0.0
            self._last_energy_fill = -1.0
            self.photo = None
            self.loop = None            # list of PhotoImage once the idle loop is ready
            self.loop_key = None
            self.loop_n = 0
            self._loop_job = None       # (key, thread, result holder)
            self._loop_pending = None
            self.stats = {"mode": "live", "frame_ms": 0.0, "fps": 0.0, "loop_mb": 0.0, "loop_fps": 0}
            self._fps_count = 0
            self._fps_t = time.perf_counter()
            self._last_tick = time.perf_counter()
            omega = 2 * math.pi / self.LOOP_SECONDS
            rng = random.Random(3)
            self.lines = []
            n = self.N_LINES
            for k in range(n):
                u = k / (n - 1)
                self.lines.append(dict(
                    y0=(0.56 + (u - 0.5) * 0.52) * self.h,          # stacked around the middle
                    a1=(8 + 30 * u) * s, f1=1.1 + 0.35 * u + rng.uniform(-0.08, 0.08), p1=rng.uniform(0, 6.3),
                    a2=(3 + 9 * u) * s, f2=2.4 + 0.6 * u + rng.uniform(-0.1, 0.1), p2=rng.uniform(0, 6.3),
                    v1=omega * (1 + k % 2), v2=-omega * (1 + k % 3),   # integer multiples: the loop closes exactly
                    tone=u, bright=(k % 5 == 2), width=(3 if k % 4 == 1 else 2),
                    fill="#e6a93c", halo="#000000",
                ))
            self.omega = omega
            import numpy as np

            self.u_grid = np.linspace(0.0, 1.0, self.N_POINTS)
            self.img_id = self.canvas.create_image(0, 0, anchor="nw")
            self.loop_idx = -1
            self.loop_x = 0
            self.loop_y = 0
            x0 = 0
            self.avatar_img = None
            img = _avatar_image(int(72 * s))
            if img is not None:
                from PIL import ImageTk

                self.aura = [self.canvas.create_oval(0, 0, 1, 1, outline="") for _ in range(2)]
                self.avatar_img = ImageTk.PhotoImage(img)
                self.avatar_id = self.canvas.create_image(40 * s, 60 * s, image=self.avatar_img)
                x0 = 92 * s
            else:
                self.aura = []
            self.x0 = x0
            self.title_id = self.canvas.create_text(x0, 42 * s, text="", anchor="w", font=self.f_title)
            self.subtitle_id = self.canvas.create_text(x0, 82 * s, text="", anchor="w", font=self.f_sub)
            self.caption_id = self.canvas.create_text(0, 0, text="", anchor="s", font=self.f_sub)
            btn_kw = dict(width=92, height=30, font=app.font, fg_color="transparent", border_width=1,
                          border_color=PALETTE["border"], bg_color=PALETTE["bg"])
            self.theme_btn = ctk.CTkButton(self.canvas, text_color=PALETTE["cyan"], command=app.toggle_theme, **btn_kw)
            self.lang_btn = ctk.CTkButton(self.canvas, text_color=PALETTE["gold"], command=app.toggle_language, **btn_kw)
            self.theme_win = self.canvas.create_window(0, 20 * s, window=self.theme_btn, anchor="ne")
            self.lang_win = self.canvas.create_window(0, 20 * s, window=self.lang_btn, anchor="ne")
            self.canvas.bind("<Configure>", self._layout)
            self.retheme()
            self.visible = True
            self._stop = False
            self.frame_event = "<<AutoOsuFrame>>"
            app.bind(self.frame_event, lambda _e: self._tick())
            self._pacer = threading.Thread(target=self._pace, daemon=True)
            self._pacer.start()

        def _pace(self) -> None:
            """Background clock: one virtual event per display refresh (slow when hidden or static)."""
            nxt = time.perf_counter()
            while not self._stop:
                period = self.frame_dt if (self.visible and self.stats["mode"] != "static") else 0.2
                nxt += period
                d = nxt - time.perf_counter()
                if d > 0:
                    time.sleep(d)
                else:
                    nxt = time.perf_counter()
                try:
                    self.app.event_generate(self.frame_event, when="tail")
                except Exception:      # window gone
                    return

        def stop(self) -> None:
            self._stop = True

        # ------------------------------------------------------------ layout / colours
        def _layout(self, _event=None) -> None:
            w = self.canvas.winfo_width()
            s = self.s
            self.canvas.coords(self.theme_win, w - 2, 24 * s)
            self.canvas.coords(self.lang_win, w - 2 - (92 + 10) * s, 24 * s)

        def retheme(self) -> None:
            bg = self.app.c("bg")
            dark = self.app.mode == "dark"
            self.canvas.configure(bg=bg)
            self.canvas.itemconfig(self.title_id, fill=self.app.c("gold"))
            self.canvas.itemconfig(self.subtitle_id, fill=self.app.c("muted"))
            self.canvas.itemconfig(self.caption_id, fill=self.app.c("gold"))
            # amber -> pale gold (the eyes); a couple of lines almost white-gold for a halo feel
            lo, hi = ("#7a5414", "#f6d98a") if dark else ("#d9b25c", "#9a6208")
            for ln in self.lines:
                tone = blend(lo, hi, ln["tone"] ** 1.3)
                if ln["bright"]:
                    tone = blend(tone, "#fff2cc" if dark else "#b8780f", 0.6)
                ln["tone_color"] = tone
            self._last_energy_fill = -1.0
            self._recolor()
            self.loop = None            # colours changed: the idle loop must be rendered again
            self.loop_key = None
            s = self.s
            for k, oid in enumerate(self.aura):
                self.canvas.itemconfig(oid, fill=blend(self.app.c("cyan"), bg, 0.8 + 0.1 * k))
                r = (46 + 8 * k) * s
                self.canvas.coords(oid, 40 * s - r, 60 * s - r, 40 * s + r, 60 * s + r)

        def _colors_for(self, e: float) -> list:
            """(fill, halo) per line for a given energy."""
            bg = self.app.c("bg")
            out = []
            for ln in self.lines:
                tone = ln.get("tone_color", self.app.c("gold"))
                dim = 0.02 + 0.3 * (1 - ln["tone"]) + 0.3 * (1 - min(1.0, e)) - 0.3 * max(0.0, e - 1)
                glow = 0.9 - 0.12 * min(1.0, e) - 0.35 * max(0.0, e - 1)
                out.append((blend(tone, bg, max(0.0, min(0.7, dim))), blend(tone, bg, max(0.35, glow))))
            return out

        def _recolor(self) -> None:
            e = min(2.0, self.energy)
            if abs(e - self._last_energy_fill) < 0.02:
                return
            self._last_energy_fill = e
            for ln, (fill, halo) in zip(self.lines, self._colors_for(e)):
                ln["fill"], ln["halo"] = fill, halo

        # ------------------------------------------------------------ state from the app
        def set_state(self, busy: bool, progress: float) -> None:
            """Called by the app: the pulse crest follows the progress and the waves wake up."""
            self.busy = busy
            self.progress = max(0.0, min(1.0, progress))
            self.energy_target = 1.0 if busy else 0.25
            if busy and self.caption_idx < 0:
                self.caption_idx = 0
                self.caption_at = 0.0

        def celebrate(self) -> None:
            self.energy = 1.9
            self.energy_target = 0.25
            self.busy = False
            self.celebrate_until = time.time() + 3.0
            self.caption_idx = 0
            self.canvas.itemconfig(self.caption_id, text="SS!!")

        # ------------------------------------------------------------ drawing
        def _left(self, w: int) -> float:
            return min(0.45, (self.x0 + 330 * self.s) / w)   # waves grow out of one seam right of the title

        def _draw(self, t: float, w: int, energy: float, crest: float, colors: list):
            """One frame of the wave region as a PIL image; anti-aliased by drawing at 2x vertical
            resolution and box-reducing (the lines are near-horizontal, so that is where steps show)."""
            import numpy as np
            from PIL import Image, ImageDraw

            ss = self.SS
            left = self._left(w)
            breath = 0.82 + 0.18 * math.sin(t * self.omega)
            amp = 0.6 + 0.6 * min(1.6, energy)
            mid = 0.56 * self.h
            x_start = int(left * w) - 4
            rw = max(8, w - x_start)
            u = left + (1.0 - left) * self.u_grid
            spread = np.minimum(1.0, (u - left) / 0.18) ** 1.5           # lines fan out from the seam
            d1, d2 = (u - crest) / 0.2, (u - 0.93) / 0.08
            env = (np.exp(-d1 * d1) + 0.5 * np.exp(-d2 * d2)) * spread * (breath * amp)
            xs = (u * w - x_start)
            img = Image.new("RGB", (rw, self.h * ss), self.app.c("bg"))
            draw = ImageDraw.Draw(img)
            polylines = []
            for ln, (fill, halo) in zip(self.lines, colors):
                y = mid + (ln["y0"] - mid) * spread + env * (
                    ln["a1"] * np.sin(6.283 * ln["f1"] * u + ln["p1"] + t * ln["v1"])
                    + ln["a2"] * np.sin(6.283 * ln["f2"] * u + ln["p2"] + t * ln["v2"]))
                pts = np.column_stack((xs, y * ss)).ravel().tolist()
                polylines.append((ln, pts, fill, halo))
            if energy > 0.6:      # halos only once the waves are awake (they read as blur at rest)
                for ln, pts, fill, halo in polylines:
                    draw.line(pts, fill=halo, width=ln["width"] + 2, joint="curve")
            for ln, pts, fill, halo in polylines:
                draw.line(pts, fill=fill, width=ln["width"], joint="curve")
            return img.reduce((1, ss)), x_start

        def _show(self, pil, x_start: int) -> None:
            from PIL import ImageTk

            if self.photo is None or self.photo.width() != pil.width or self.photo.height() != pil.height:
                self.photo = ImageTk.PhotoImage(pil)
                self.canvas.itemconfig(self.img_id, image=self.photo)
            else:
                self.photo.paste(pil)
            self.canvas.coords(self.img_id, x_start, 0)
            self.loop_idx = -1

        def _show_loop_frame(self, idx: int) -> None:
            """Copy a pre-rendered frame into the displayed image (a C memcpy inside Tk, ~0.3 ms)."""
            from PIL import Image, ImageTk

            frame = self.loop[idx]
            if self.photo is None or self.photo.width() != frame.width() or self.photo.height() != frame.height():
                self.photo = ImageTk.PhotoImage(Image.new("RGB", (frame.width(), frame.height()), self.app.c("bg")))
                self.canvas.itemconfig(self.img_id, image=self.photo)
                self.canvas.coords(self.img_id, self.loop_x, self.loop_y)
            elif self.loop_idx < 0:
                self.canvas.coords(self.img_id, self.loop_x, self.loop_y)
            self.photo.tk.call(str(self.photo), "copy", str(frame), "-to", 0, 0)
            self.loop_idx = idx

        # ------------------------------------------------------------ idle loop (pre-rendered)
        def _ensure_loop(self, w: int) -> None:
            key = (w, self.app.mode, self.h)
            if self.loop is not None and self.loop_key == key:
                return
            if self._loop_job is not None:
                if self._loop_job[0] == key:
                    return
                self._loop_job = None      # width or theme changed while rendering: start over
            unique_fps = min(self.refresh_hz, 120)
            colors = self._colors_for(0.25)
            holder = {"frames": None, "x_start": 0, "y_start": 0, "n": 0}
            budget = self.LOOP_BUDGET_MB * 1024 * 1024

            def work() -> None:
                from PIL import Image, ImageChops

                # probe a few frames to find the rows the waves actually use, then size the loop to the budget
                bg = self.app.c("bg")
                top, bottom = self.h, 0
                for i in range(8):
                    pil, x_start = self._draw(i / 8 * self.LOOP_SECONDS, w, 0.25, 0.68, colors)
                    bbox = ImageChops.difference(pil, Image.new("RGB", pil.size, bg)).getbbox()
                    if bbox:
                        top, bottom = min(top, bbox[1]), max(bottom, bbox[3])
                top, bottom = max(0, top - 3), min(self.h, bottom + 3)
                if bottom <= top:
                    top, bottom = 0, self.h
                rw = w - (int(self._left(w) * w) - 4)
                per_frame = rw * (bottom - top) * 4
                n = max(24, min(int(round(self.LOOP_SECONDS * unique_fps)), budget // max(1, per_frame)))
                frames = []
                for i in range(n):
                    pil, x_start = self._draw(i / n * self.LOOP_SECONDS, w, 0.25, 0.68, colors)
                    frames.append(pil.crop((0, top, pil.width, bottom)))
                holder.update(frames=frames, x_start=x_start, y_start=top, n=n)

            th = threading.Thread(target=work, daemon=True)
            th.start()
            self._loop_job = (key, th, holder, 0)

        def _adopt_loop(self) -> None:
            """Convert the rendered PIL frames into Tk images, a few per tick, then switch to playback."""
            from PIL import ImageTk

            key, th, holder, n = self._loop_job
            if holder["frames"] is None:
                return
            pending = self._loop_pending
            if pending is None or pending[0] != key:
                pending = [key, [], holder["frames"], holder["x_start"], holder["y_start"]]
                self._loop_pending = pending
            done = pending[1]
            frames = pending[2]
            for _ in range(12):
                if len(done) >= len(frames):
                    break
                done.append(ImageTk.PhotoImage(frames[len(done)]))
            if len(done) >= len(frames):
                self.loop, self.loop_key, self.loop_n = done, key, len(done)
                self.loop_x, self.loop_y = pending[3], pending[4]
                self.loop_idx = -1
                self.stats["loop_mb"] = len(done) * frames[0].width * frames[0].height * 4 / 1048576
                self.stats["loop_fps"] = len(done) / self.LOOP_SECONDS
                self._loop_job = None
                self._loop_pending = None

        # ------------------------------------------------------------ main tick (paced to the display)
        def _tick(self) -> None:
            try:
                now = time.perf_counter()
                dt = now - self._last_tick
                if dt < 0.4 * self.frame_dt:       # events that piled up while the main thread was busy
                    return
                dt = min(0.1, dt)
                self._last_tick = now
                self.visible = bool(self.app.winfo_viewable()) and self.app.state() != "iconic"
                if not self.visible:
                    return
                tick0 = time.perf_counter()
                w = max(200, self.canvas.winfo_width())
                wall = time.time()
                t = wall - self.t0
                rate = 4.0 if self.energy < self.energy_target else 2.5
                self.energy += (self.energy_target - self.energy) * (1 - math.exp(-rate * dt))
                self._recolor()
                left = self._left(w)
                want = left + (1.0 - left) * (0.1 + 0.85 * self.progress) if self.busy else 0.68
                self.crest += (want - self.crest) * (1 - math.exp(-1.8 * dt))
                animate = bool(self.app.anim_var.get())
                idle = (not self.busy and abs(self.energy - 0.25) < 0.02 and wall > self.celebrate_until
                        and abs(self.crest - 0.68) < 0.01)
                if idle and animate:
                    self._ensure_loop(w)
                    if self._loop_job is not None:
                        self._adopt_loop()
                    if self.loop is not None and self.loop_key == (w, self.app.mode, self.h):
                        idx = int((t % self.LOOP_SECONDS) / self.LOOP_SECONDS * self.loop_n) % self.loop_n
                        if idx != self.loop_idx:          # only touch Tk when the picture changes
                            self._show_loop_frame(idx)
                        self.stats["mode"] = "loop"
                    else:
                        pil, xs = self._draw(t, w, self.energy, self.crest, [(ln["fill"], ln["halo"]) for ln in self.lines])
                        self._show(pil, xs)
                        self.stats["mode"] = "live"
                elif not animate and idle:
                    if self.stats["mode"] != "static":      # one frozen frame
                        pil, xs = self._draw(0.0, w, self.energy, self.crest, [(ln["fill"], ln["halo"]) for ln in self.lines])
                        self._show(pil, xs)
                        self.stats["mode"] = "static"
                else:
                    pil, xs = self._draw(t, w, self.energy, self.crest, [(ln["fill"], ln["halo"]) for ln in self.lines])
                    self._show(pil, xs)
                    self.stats["mode"] = "live"
                mid = 0.56 * self.h
                # a floating caption in osu! slang rides on the crest while generating
                if self.busy:
                    phrases = memes()
                    if wall - self.caption_at > 2.6:
                        self.caption_at = wall
                        self.caption_idx = (self.caption_idx + 1) % len(phrases)
                        self.canvas.itemconfig(self.caption_id, text=phrases[self.caption_idx])
                    cx = min(w - 250 * self.s, max(left * w + 70 * self.s, self.crest * w))
                    self.canvas.coords(self.caption_id, cx, mid - 34 * self.s + 3 * self.s * math.sin(t * 2.2))
                elif wall < self.celebrate_until:
                    cx = min(w - 250 * self.s, max(left * w + 70 * self.s, self.crest * w))
                    self.canvas.coords(self.caption_id, cx, mid - 34 * self.s)
                elif self.caption_idx >= 0:          # clear once; touching the item every frame forces redraws
                    self.canvas.itemconfig(self.caption_id, text="")
                    self.caption_idx = -1
                self.stats["frame_ms"] = (time.perf_counter() - tick0) * 1000
                self._fps_count += 1
                if now - self._fps_t >= 1.0:
                    self.stats["fps"] = self._fps_count / (now - self._fps_t)
                    self._fps_count, self._fps_t = 0, now
            except tk.TclError:
                return

        def set_texts(self, title: str, subtitle: str) -> None:
            self.canvas.itemconfig(self.title_id, text=title)
            self.canvas.itemconfig(self.subtitle_id, text=subtitle)

    class App(ctk.CTk):
        def __init__(self) -> None:
            super().__init__()
            self.dnd_ok = False
            if TkinterDnD is not None:
                try:
                    self.TkdndVersion = TkinterDnD._require(self)
                    self.dnd_ok = True
                except Exception:
                    self.dnd_ok = False
            self.settings = load_settings()
            set_language(self.settings.get("language", "zh" if _system_is_chinese() else "en"))
            mode = self.settings.get("appearance") or ctk.get_appearance_mode().lower()
            self.mode = mode if mode in ("light", "dark") else "dark"
            ctk.set_appearance_mode(self.mode)
            self.widgets: Dict[str, object] = {}
            self.q: "queue.Queue" = queue.Queue()
            self.busy = False
            self.last_osz: Optional[Path] = None
            self.advanced_open = bool(self.settings.get("advanced_open", False))
            self._progress_target = 0.0
            self._progress_shown = 0.0
            self._pulsing = False
            self.anim_var = ctk.BooleanVar(value=self.settings.get("animations", True))
            self.font = ctk.CTkFont(family=_ui_font(), size=13)
            self.font_bold = ctk.CTkFont(family=_ui_font(), size=15, weight="bold")
            self.font_title = ctk.CTkFont(family=_ui_font(), size=24, weight="bold")
            self.geometry(self.settings.get("geometry", "780x800"))
            self.minsize(660, 620)
            self.protocol("WM_DELETE_WINDOW", self.on_close)
            try:
                self.attributes("-alpha", 0.0)
            except tk.TclError:
                pass
            self.build()
            self.refresh_texts()
            self.refresh_models()
            self.after(100, self.poll)
            self.after(30, self._animate_progress)
            self.after(60, self._pulse)
            self.after(40, self._fade_in)

        def c(self, key: str) -> str:
            return PALETTE[key][0 if self.mode == "light" else 1]

        # ------------------------------------------------------------------ layout
        def build(self) -> None:
            w = self.widgets
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(6, weight=1)
            self.header = Header(self, self)
            self.after(400, self._set_window_icon)

            # song
            song = ctk.CTkFrame(self)
            song.grid(row=1, column=0, sticky="ew", padx=20, pady=6)
            song.grid_columnconfigure(0, weight=1)
            w["song.section"] = ctk.CTkLabel(song, font=self.font_bold, anchor="w")
            w["song.section"].grid(row=0, column=0, sticky="w", padx=12, pady=(8, 0))
            self.song_var = ctk.StringVar(value=self.settings.get("song", ""))
            w["song.entry"] = ctk.CTkEntry(song, textvariable=self.song_var, font=self.font, height=34)
            w["song.entry"].grid(row=1, column=0, sticky="ew", padx=(12, 6), pady=6)
            w["song.browse"] = ctk.CTkButton(song, width=110, font=self.font, command=self.browse_song)
            w["song.browse"].grid(row=1, column=1, padx=(0, 12), pady=6)
            w["song.hint"] = ctk.CTkLabel(song, font=self.font, anchor="w", text_color=PALETTE["muted"])
            w["song.hint"].grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 8))
            if self.dnd_ok:
                for target in (song, w["song.entry"], w["song.hint"]):
                    try:
                        target.drop_target_register(DND_FILES)
                        target.dnd_bind("<<Drop>>", self.on_drop)
                    except Exception:
                        pass

            # difficulties
            diff = ctk.CTkFrame(self)
            diff.grid(row=2, column=0, sticky="ew", padx=20, pady=6)
            w["diff.section"] = ctk.CTkLabel(diff, font=self.font_bold, anchor="w")
            w["diff.section"].grid(row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(8, 0))
            chosen = set(self.settings.get("difficulties", ["Normal", "Hard"]))
            self.diff_vars = {}
            for i, name in enumerate(PRESETS):
                var = ctk.BooleanVar(value=name in chosen)
                self.diff_vars[name] = var
                w[f"diff.{name}"] = ctk.CTkCheckBox(diff, variable=var, font=self.font)
                w[f"diff.{name}"].grid(row=1, column=i, sticky="w", padx=12, pady=6)
            w["diff.hint"] = ctk.CTkLabel(diff, font=self.font, anchor="w", text_color=PALETTE["muted"])
            w["diff.hint"].grid(row=2, column=0, columnspan=4, sticky="w", padx=12, pady=(0, 8))

            # output
            out = ctk.CTkFrame(self)
            out.grid(row=3, column=0, sticky="ew", padx=20, pady=6)
            out.grid_columnconfigure(1, weight=1)
            w["out.section"] = ctk.CTkLabel(out, font=self.font_bold, anchor="w")
            w["out.section"].grid(row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(8, 0))
            w["out.folder"] = ctk.CTkLabel(out, font=self.font, anchor="w")
            w["out.folder"].grid(row=1, column=0, sticky="w", padx=(12, 6), pady=6)
            self.out_var = ctk.StringVar(value=self.settings.get("out_dir", str(app_root() / "output")))
            w["out.entry"] = ctk.CTkEntry(out, textvariable=self.out_var, font=self.font, height=34)
            w["out.entry"].grid(row=1, column=1, sticky="ew", pady=6)
            w["out.browse"] = ctk.CTkButton(out, width=110, font=self.font, command=self.browse_out)
            w["out.browse"].grid(row=1, column=2, padx=(6, 12), pady=6)
            self.open_osu_var = ctk.BooleanVar(value=self.settings.get("open_osu", True))
            w["out.open_osu"] = ctk.CTkCheckBox(out, variable=self.open_osu_var, font=self.font)
            w["out.open_osu"].grid(row=2, column=0, columnspan=3, sticky="w", padx=12, pady=(0, 8))

            # advanced (collapsible)
            w["adv.toggle"] = ctk.CTkButton(self, fg_color="transparent", anchor="w", font=self.font,
                                            text_color=PALETTE["cyan"], hover=False, command=self.toggle_advanced)
            w["adv.toggle"].grid(row=4, column=0, sticky="w", padx=20, pady=(4, 0))
            adv = ctk.CTkFrame(self)
            self.adv_frame = adv
            adv.grid_columnconfigure((1, 3), weight=1)
            s = self.settings
            self.seed_var = ctk.StringVar(value=str(s.get("seed", 0)))
            self.bpm_var = ctk.StringVar(value=s.get("bpm", ""))
            self.offset_var = ctk.StringVar(value=s.get("offset", ""))
            self.creator_var = ctk.StringVar(value=s.get("creator", "AUTO-OSU"))
            self.star_var = ctk.StringVar(value=s.get("star", ""))
            self.quality_var = ctk.StringVar(value=s.get("quality", "normal"))
            self.device_var = ctk.StringVar(value=s.get("device", "auto"))
            self.engine_var = ctk.StringVar(value=s.get("engine", "ml"))
            self.preview_var = ctk.BooleanVar(value=s.get("preview", False))
            rows = [("adv.seed", self.seed_var), ("adv.bpm", self.bpm_var), ("adv.offset", self.offset_var),
                    ("adv.creator", self.creator_var), ("adv.star", self.star_var)]
            for i, (key, var) in enumerate(rows):
                r, c = divmod(i, 2)
                w[key] = ctk.CTkLabel(adv, font=self.font, anchor="w")
                w[key].grid(row=r, column=2 * c, sticky="w", padx=(12, 6), pady=4)
                ctk.CTkEntry(adv, textvariable=var, font=self.font, width=150).grid(
                    row=r, column=2 * c + 1, sticky="w", padx=(0, 12), pady=4)
            w["adv.quality"] = ctk.CTkLabel(adv, font=self.font, anchor="w")
            w["adv.quality"].grid(row=3, column=0, sticky="w", padx=(12, 6), pady=4)
            w["adv.quality.menu"] = ctk.CTkOptionMenu(adv, font=self.font, width=180, values=[],
                                                      command=lambda _v: None)
            w["adv.quality.menu"].grid(row=3, column=1, sticky="w", pady=4)
            w["adv.device"] = ctk.CTkLabel(adv, font=self.font, anchor="w")
            w["adv.device"].grid(row=3, column=2, sticky="w", padx=(12, 6), pady=4)
            w["adv.device.menu"] = ctk.CTkOptionMenu(adv, font=self.font, width=150, values=["auto", "cuda", "cpu"],
                                                     variable=self.device_var)
            w["adv.device.menu"].grid(row=3, column=3, sticky="w", pady=4)
            w["adv.engine"] = ctk.CTkLabel(adv, font=self.font, anchor="w")
            w["adv.engine"].grid(row=4, column=0, sticky="w", padx=(12, 6), pady=4)
            w["adv.engine.menu"] = ctk.CTkOptionMenu(adv, font=self.font, width=260, values=[], command=lambda _v: None)
            w["adv.engine.menu"].grid(row=4, column=1, columnspan=2, sticky="w", pady=4)
            w["adv.preview"] = ctk.CTkCheckBox(adv, variable=self.preview_var, font=self.font)
            w["adv.preview"].grid(row=5, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 10))
            w["adv.anim"] = ctk.CTkCheckBox(adv, variable=self.anim_var, font=self.font)
            w["adv.anim"].grid(row=5, column=2, columnspan=2, sticky="w", padx=12, pady=(4, 10))
            if self.advanced_open:
                adv.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 6))

            # models + run
            run = ctk.CTkFrame(self)
            run.grid(row=6, column=0, sticky="nsew", padx=20, pady=6)
            run.grid_columnconfigure(0, weight=1)
            run.grid_rowconfigure(4, weight=1)
            w["models.status"] = ctk.CTkLabel(run, font=self.font, anchor="w", wraplength=520, justify="left")
            w["models.status"].grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))
            w["models.download"] = ctk.CTkButton(run, width=130, font=self.font, command=self.download_models)
            w["models.download"].grid(row=0, column=1, padx=(6, 12), pady=(10, 4))
            self.progress = ctk.CTkProgressBar(run, height=14)
            self.progress.set(0)
            self.progress.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 2))
            w["status"] = ctk.CTkLabel(run, font=self.font, anchor="w", wraplength=700, justify="left")
            w["status"].grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 4))
            btns = ctk.CTkFrame(run, fg_color="transparent")
            btns.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
            w["run.generate"] = ctk.CTkButton(btns, height=44, width=210, font=self.font_bold, command=self.start)
            w["run.generate"].pack(side="left")
            secondary = dict(height=44, font=self.font, fg_color="transparent", border_width=1,
                             border_color=PALETTE["border"], text_color=PALETTE["gold"])
            w["run.open_osz"] = ctk.CTkButton(btns, command=lambda: self.last_osz and open_path(self.last_osz),
                                              state="disabled", **secondary)
            w["run.open_osz"].pack(side="left", padx=8)
            w["run.open_folder"] = ctk.CTkButton(btns, command=lambda: open_path(Path(self.out_var.get())), **secondary)
            w["run.open_folder"].pack(side="left")
            self.log_box = ctk.CTkTextbox(run, font=ctk.CTkFont(family="Consolas", size=12), height=120)
            self.log_box.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=12, pady=(4, 10))
            self.log_box.configure(state="disabled")

            w["about"] = ctk.CTkLabel(self, font=ctk.CTkFont(family=_ui_font(), size=11), text_color=PALETTE["muted"])
            w["about"].grid(row=7, column=0, sticky="e", padx=24, pady=(0, 8))

        def _set_window_icon(self) -> None:
            try:
                from PIL import Image, ImageTk

                if AVATAR.exists():
                    self._icon_img = ImageTk.PhotoImage(Image.open(AVATAR).convert("RGBA").resize((64, 64), Image.LANCZOS))
                    self.iconphoto(True, self._icon_img)
            except Exception:
                pass

        def refresh_texts(self) -> None:
            w = self.widgets
            self.title(f"AUTO-OSU {__version__} · {AUTHOR}")
            self.header.set_texts(tr("app.title"), tr("app.subtitle"))
            self.header.lang_btn.configure(text=tr("lang.toggle"))
            self.header.theme_btn.configure(text=tr("theme.light" if self.mode == "dark" else "theme.dark"))
            for key in ("song.section", "song.browse", "song.hint", "diff.section", "diff.hint", "out.section",
                        "out.folder", "out.browse", "out.open_osu", "adv.seed", "adv.bpm", "adv.offset",
                        "adv.creator", "adv.star", "adv.quality", "adv.device", "adv.engine", "adv.preview", "adv.anim",
                        "models.download", "run.open_osz", "run.open_folder", "about"):
                w[key].configure(text=tr(key))
            for name in PRESETS:
                w[f"diff.{name}"].configure(text=tr(f"diff.{name}"))
            w["adv.toggle"].configure(text=tr("adv.hide" if self.advanced_open else "adv.show"))
            self._quality_labels = {k: tr(f"adv.quality.{k}") for k in QUALITY_STEPS}
            w["adv.quality.menu"].configure(values=list(self._quality_labels.values()))
            w["adv.quality.menu"].set(self._quality_labels[self.quality_var.get()])
            self._engine_labels = {"ml": tr("adv.engine.ml"), "rules": tr("adv.engine.rules")}
            w["adv.engine.menu"].configure(values=list(self._engine_labels.values()))
            w["adv.engine.menu"].set(self._engine_labels[self.engine_var.get()])
            w["run.generate"].configure(text=tr("run.running" if self.busy else "run.generate"))
            if not self.busy:
                w["status"].configure(text=tr("status.idle"))
            self.refresh_models()

        # ------------------------------------------------------------------ animations
        def _fade_in(self) -> None:
            try:
                a = float(self.attributes("-alpha"))
                if a < 1.0:
                    self.attributes("-alpha", min(1.0, a + 0.12))
                    self.after(16, self._fade_in)
            except tk.TclError:
                pass

        def set_progress(self, value: float) -> None:
            self._progress_target = max(0.0, min(1.0, value))
            if value <= 0.0:
                self._progress_shown = 0.0
                self.progress.set(0.0)

        def _animate_progress(self) -> None:
            try:
                d = self._progress_target - self._progress_shown
                if abs(d) > 0.0015:
                    self._progress_shown += d * 0.22
                    self.progress.set(self._progress_shown)
                elif d:
                    self._progress_shown = self._progress_target
                    self.progress.set(self._progress_shown)
            except tk.TclError:
                return
            self.after(30, self._animate_progress)

        def _pulse(self) -> None:
            """The Generate button breathes between amber and pale gold while work is running."""
            try:
                btn = self.widgets["run.generate"]
                if self.busy:
                    k = (math.sin(time.time() * 3.0) + 1) / 2
                    btn.configure(fg_color=blend(self.c("gold"), "#f7dc8f", 0.6 * k))   # gold -> pale gold, never greenish
                    self._pulsing = True
                elif self._pulsing:
                    btn.configure(fg_color=PALETTE["gold"])
                    self._pulsing = False
            except tk.TclError:
                return
            self.after(60, self._pulse)

        def _flash_done(self, n: int = 6) -> None:
            if n <= 0:
                self.progress.configure(progress_color=PALETTE["cyan"])
                return
            self.progress.configure(progress_color=PALETTE["gold"] if n % 2 else PALETTE["cyan"])
            self.after(140, lambda: self._flash_done(n - 1))

        def _fade_frame(self, frame, step: int = 0) -> None:
            if step > 6:
                frame.configure(fg_color=PALETTE["panel"])
                return
            frame.configure(fg_color=blend(self.c("bg"), self.c("panel"), step / 6))
            self.after(22, lambda: self._fade_frame(frame, step + 1))

        # ------------------------------------------------------------------ actions
        def toggle_language(self) -> None:
            set_language("en" if language() == "zh" else "zh")
            self.refresh_texts()

        def toggle_theme(self) -> None:
            self.mode = "light" if self.mode == "dark" else "dark"
            ctk.set_appearance_mode(self.mode)
            self.header.retheme()
            self.refresh_texts()

        def toggle_advanced(self) -> None:
            self.advanced_open = not self.advanced_open
            if self.advanced_open:
                self.adv_frame.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 6))
                self._fade_frame(self.adv_frame)
            else:
                self.adv_frame.grid_forget()
            self.widgets["adv.toggle"].configure(text=tr("adv.hide" if self.advanced_open else "adv.show"))

        def browse_song(self) -> None:
            from tkinter import filedialog

            path = filedialog.askopenfilename(filetypes=[(tr("song.filetypes"), " ".join(AUDIO_EXT)), ("*", "*.*")])
            if path:
                self.song_var.set(path)

        def browse_out(self) -> None:
            from tkinter import filedialog

            path = filedialog.askdirectory(initialdir=self.out_var.get() or None)
            if path:
                self.out_var.set(path)

        def on_drop(self, event) -> None:
            data = event.data.strip()
            if data.startswith("{"):
                data = data[1:].split("}")[0]
            else:
                data = data.split(" ")[0]
            self.song_var.set(data)

        def models_present(self) -> Dict[str, Optional[Path]]:
            return {name: find_model(name) for name in MODELS}

        def refresh_models(self) -> None:
            w = self.widgets
            found = self.models_present()
            if all(found.values()):
                w["models.status"].configure(text=tr("models.ok", rhythm=found["rhythm"].name, coord=found["coord"].name),
                                             text_color=PALETTE["text"])
                w["models.download"].grid_remove()
            else:
                w["models.status"].configure(text=tr("models.missing"), text_color=PALETTE["warn"])
                w["models.download"].grid()

        def download_models(self) -> None:
            if self.busy:
                return
            self.set_busy(True)

            def work() -> None:
                try:
                    for name in MODELS:
                        if not find_model(name):
                            self.q.put(("progress", 0.0, tr("models.downloading", name=MODELS[name].filename)))
                            ensure_model(name, lambda f, m: self.q.put(("progress", f, m)))
                    self.q.put(("models", None))
                except Exception as e:  # noqa: BLE001
                    self.q.put(("error", f"{e}"))

            threading.Thread(target=work, daemon=True).start()

        def log(self, text: str) -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", text + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        def set_busy(self, busy: bool) -> None:
            self.busy = busy
            if not busy:
                self.header.set_state(False, 0.0)
            self.widgets["run.generate"].configure(state="disabled" if busy else "normal",
                                                   text=tr("run.running" if busy else "run.generate"))
            self.widgets["models.download"].configure(state="disabled" if busy else "normal")

        def collect_settings(self) -> Dict:
            quality = next((k for k, v in self._quality_labels.items() if v == self.widgets["adv.quality.menu"].get()), "normal")
            engine = next((k for k, v in self._engine_labels.items() if v == self.widgets["adv.engine.menu"].get()), "ml")
            self.quality_var.set(quality)
            self.engine_var.set(engine)
            return {
                "language": language(), "appearance": self.mode, "song": self.song_var.get(), "out_dir": self.out_var.get(),
                "difficulties": [n for n, v in self.diff_vars.items() if v.get()],
                "open_osu": self.open_osu_var.get(), "seed": self.seed_var.get(), "bpm": self.bpm_var.get(),
                "offset": self.offset_var.get(), "creator": self.creator_var.get(), "star": self.star_var.get(),
                "quality": quality, "device": self.device_var.get(), "engine": engine,
                "preview": self.preview_var.get(), "animations": self.anim_var.get(),
                "advanced_open": self.advanced_open, "geometry": self.geometry(),
            }

        def start(self) -> None:
            if self.busy:
                return
            s = self.collect_settings()
            save_settings(s)
            song = Path(s["song"].strip().strip('"'))
            if not s["song"].strip():
                return self.show_error(tr("err.no_song"))
            if not song.exists():
                return self.show_error(tr("err.file_missing", path=song))
            if not s["difficulties"]:
                return self.show_error(tr("err.no_diff"))
            found = self.models_present()
            if s["engine"] == "ml" and not all(found.values()):
                return self.show_error(tr("err.no_models"))

            def num(v: str):
                v = v.strip()
                return float(v) if v else None

            try:
                seed = int(s["seed"] or 0)
                bpm, offset, star = num(s["bpm"]), num(s["offset"]), num(s["star"])
            except ValueError as e:
                return self.show_error(str(e))
            self.log_box.configure(state="normal")
            self.log_box.delete("1.0", "end")
            self.log_box.configure(state="disabled")
            self.set_progress(0.0)
            self.widgets["run.open_osz"].configure(state="disabled")
            self.set_busy(True)
            kwargs = dict(
                seed=seed, bpm=bpm, offset_ms=offset, creator=s["creator"] or "AUTO-OSU", star_rating=star,
                coord_steps=QUALITY_STEPS[s["quality"]], device=s["device"],
                rhythm_model=str(found["rhythm"]) if s["engine"] == "ml" else None,
                coord_model=str(found["coord"]) if s["engine"] == "ml" else None,
            )
            threading.Thread(target=self.work, args=(song, s["difficulties"], Path(s["out_dir"]), kwargs, s["preview"]),
                             daemon=True).start()

        def work(self, song: Path, diffs, out_dir: Path, kwargs: Dict, preview: bool) -> None:
            from .generate import generate

            try:
                res = generate(song, diffs, out_dir, log=lambda t: self.q.put(("log", t)),
                               progress=lambda f, m: self.q.put(("progress", f, m)), **kwargs)
                if preview:
                    from .preview import render_preview

                    for d in res.diffs:
                        out = out_dir / f"{song.stem} [{d.preset.name}]_preview.mp3"
                        self.q.put(("log", f"preview: {render_preview(res.audio_file, d.beatmap, out, click_shift_ms=res.osu_shift_ms)}"))
                self.q.put(("done", res))
            except Exception as e:  # noqa: BLE001
                self.q.put(("log", traceback.format_exc()))
                self.q.put(("error", f"{type(e).__name__}: {e}"))

        def status_text(self, msg: str) -> str:
            if ": rhythm" in msg:
                return tr("status.rhythm", diff=msg.split(":")[0])
            if ": placing" in msg:
                diff, _, detail = msg.partition(": placing")
                return tr("status.placing", diff=diff, detail=detail.strip())
            return tr(f"status.{msg}") if f"status.{msg}" in _keys() else msg

        def poll(self) -> None:
            try:
                while True:
                    item = self.q.get_nowait()
                    kind = item[0]
                    if kind == "log":
                        self.log(item[1])
                    elif kind == "progress":
                        self.set_progress(item[1])
                        self.header.set_state(True, item[1])
                        self.widgets["status"].configure(text=self.status_text(item[2]))
                    elif kind == "models":
                        self.set_busy(False)
                        self.set_progress(0.0)
                        self.widgets["status"].configure(text=tr("status.idle"))
                        self.refresh_models()
                    elif kind == "done":
                        res = item[1]
                        self.set_busy(False)
                        self.set_progress(1.0)
                        self.last_osz = res.osz
                        self.widgets["run.open_osz"].configure(state="normal")
                        self.widgets["status"].configure(
                            text=tr("status.done", file=res.osz.name, secs=res.elapsed_s, device=res.device))
                        self.log(tr("result.summary", bpm=res.timing.bpm, n=len(res.diffs)))
                        for d in res.diffs:
                            s = d.summary()
                            self.log(tr("result.diff", name=d.preset.name, objects=s["objects"], sliders=s["sliders"], nps=s["nps"]))
                        self._flash_done()
                        self.header.celebrate()
                        if self.open_osu_var.get():
                            open_path(res.osz)
                    elif kind == "error":
                        self.set_busy(False)
                        self.widgets["status"].configure(text=tr("status.error", err=item[1]))
                        self.show_error(item[1])
            except queue.Empty:
                pass
            self.after(100, self.poll)

        def show_error(self, text: str) -> None:
            from tkinter import messagebox

            messagebox.showerror("AUTO-OSU", text)

        def on_close(self) -> None:
            try:
                self.header.stop()
                save_settings(self.collect_settings())
            except Exception:
                pass
            self.destroy()

    _high_res_timer(True)
    try:
        app = App()
        app.mainloop()
    finally:
        _high_res_timer(False)
    return 0


def _display_refresh_hz() -> int:
    """Refresh rate of the primary display (Windows), 60 elsewhere or on failure."""
    if sys.platform != "win32":
        return 60
    try:
        import ctypes
        import ctypes.wintypes as wt

        class DEVMODE(ctypes.Structure):
            _fields_ = [("dmDeviceName", wt.WCHAR * 32), ("dmSpecVersion", wt.WORD), ("dmDriverVersion", wt.WORD),
                        ("dmSize", wt.WORD), ("dmDriverExtra", wt.WORD), ("dmFields", wt.DWORD),
                        ("dmPositionX", ctypes.c_long), ("dmPositionY", ctypes.c_long),
                        ("dmDisplayOrientation", wt.DWORD), ("dmDisplayFixedOutput", wt.DWORD),
                        ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short), ("dmYResolution", ctypes.c_short),
                        ("dmTTOption", ctypes.c_short), ("dmCollate", ctypes.c_short), ("dmFormName", wt.WCHAR * 32),
                        ("dmLogPixels", wt.WORD), ("dmBitsPerPel", wt.DWORD), ("dmPelsWidth", wt.DWORD),
                        ("dmPelsHeight", wt.DWORD), ("dmDisplayFlags", wt.DWORD), ("dmDisplayFrequency", wt.DWORD)]

        dm = DEVMODE()
        dm.dmSize = ctypes.sizeof(DEVMODE)
        if ctypes.windll.user32.EnumDisplaySettingsW(None, -1, ctypes.byref(dm)):
            hz = int(dm.dmDisplayFrequency)
            if 30 <= hz <= 480:
                return hz
    except Exception:
        pass
    return 60


def _high_res_timer(enable: bool) -> None:
    """1 ms scheduler resolution on Windows so Tk timers can pace a 120-240 Hz display."""
    if sys.platform == "win32":
        try:
            import ctypes

            (ctypes.windll.winmm.timeBeginPeriod if enable else ctypes.windll.winmm.timeEndPeriod)(1)
        except Exception:
            pass


def _keys():
    from .i18n import STRINGS

    return STRINGS


def _system_is_chinese() -> bool:
    try:
        import locale

        lang = locale.getlocale()[0] or ""
        if not lang and sys.platform == "win32":
            import ctypes

            lang = locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage(), "")
        return lang.lower().startswith(("zh", "chinese"))
    except Exception:
        return False


def _ui_font() -> str:
    return "Microsoft YaHei UI" if sys.platform == "win32" else "Helvetica"


if __name__ == "__main__":
    sys.exit(run_gui())
