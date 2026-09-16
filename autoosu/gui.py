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
        """Canvas banner: layered pulse waves in silver / graphite, title, subtitle, avatar, toggles."""

        N_LINES = 8
        N_POINTS = 56

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
            rng = random.Random(3)
            self.lines = []
            n = self.N_LINES
            for k in range(n):
                u = k / (n - 1)
                self.lines.append(dict(
                    halo=self.canvas.create_line(0, 0, 1, 1, width=max(2, int(3 * s)), smooth=True, state="hidden"),
                    id=self.canvas.create_line(0, 0, 1, 1, width=2 if k % 4 == 1 else 1, smooth=True),
                    y0=(0.56 + (u - 0.5) * 0.52) * self.h,          # stacked around the middle
                    a1=(8 + 30 * u) * s, f1=1.1 + 0.35 * u + rng.uniform(-0.08, 0.08), p1=rng.uniform(0, 6.3),
                    a2=(3 + 9 * u) * s, f2=2.4 + 0.6 * u + rng.uniform(-0.1, 0.1), p2=rng.uniform(0, 6.3),
                    v1=0.55 + 0.35 * u, v2=-(0.8 + 0.4 * u),
                    tone=u, halo_bright=(k % 5 == 2),
                ))
            for ln in self.lines:
                self.canvas.tag_raise(ln["id"])
            self.xs = [i / (self.N_POINTS - 1) for i in range(self.N_POINTS)]
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
            self._tick()

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
            self._last_energy_fill = -1.0
            # amber -> pale gold (the eyes), every fifth line almost white-gold for a halo feel
            lo, hi = ("#7a5414", "#f6d98a") if dark else ("#d9b25c", "#9a6208")
            for ln in self.lines:
                tone = blend(lo, hi, ln["tone"] ** 1.3)
                if ln["halo_bright"]:
                    tone = blend(tone, "#fff2cc" if dark else "#b8780f", 0.6)
                ln["tone_color"] = tone
            self._recolor()
            s = self.s
            for k, oid in enumerate(self.aura):
                self.canvas.itemconfig(oid, fill=blend(self.app.c("cyan"), bg, 0.8 + 0.1 * k))
                r = (46 + 8 * k) * s
                self.canvas.coords(oid, 40 * s - r, 60 * s - r, 40 * s + r, 60 * s + r)

        def _recolor(self) -> None:
            """Line brightness follows the energy: dim at rest, glowing while generating."""
            e = min(2.0, self.energy)
            if abs(e - self._last_energy_fill) < 0.02:
                return
            self._last_energy_fill = e
            bg = self.app.c("bg")
            dark = self.app.mode == "dark"
            for ln in self.lines:
                tone = ln.get("tone_color", self.app.c("gold"))
                # crisp 1 px lines: little blending towards the background so they stay sharp
                dim = 0.02 + 0.3 * (1 - ln["tone"]) + 0.3 * (1 - min(1.0, e)) - 0.3 * max(0.0, e - 1)
                self.canvas.itemconfig(ln["id"], fill=blend(tone, bg, max(0.0, min(0.7, dim))))
                # the soft halo only appears for the finish flash
                if e > 1.15:
                    self.canvas.itemconfig(ln["halo"], state="normal", fill=blend(tone, bg, 0.55 + 0.3 * (2.0 - e)))
                else:
                    self.canvas.itemconfig(ln["halo"], state="hidden")

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
            self.canvas.itemconfig(self.caption_id, text="SS!!")

        def _tick(self) -> None:
            try:
                if self.app.winfo_viewable() and self.app.state() != "iconic":
                    w = max(200, self.canvas.winfo_width())
                    now = time.time()
                    t = now - self.t0
                    self.energy += (self.energy_target - self.energy) * (0.08 if self.energy > self.energy_target else 0.15)
                    e = self.energy
                    self._recolor()
                    left = min(0.45, (self.x0 + 330 * self.s) / w)   # waves grow out of one line right of the title
                    want = left + (1.0 - left) * (0.1 + 0.85 * self.progress) if self.busy else 0.68
                    self.crest += (want - self.crest) * 0.06
                    speed = 0.6 + 1.6 * min(1.0, e)
                    breath = 0.82 + 0.18 * math.sin(t * (0.9 + 1.5 * min(1.0, e)))
                    amp = 0.6 + 0.6 * min(1.6, e)
                    mid = 0.56 * self.h
                    for ln in self.lines:
                        pts = []
                        for u0 in self.xs:
                            u = left + (1.0 - left) * u0
                            spread = min(1.0, (u - left) / 0.18) ** 1.5       # lines fan out from the seam
                            d1, d2 = (u - self.crest) / 0.2, (u - 0.93) / 0.08
                            env = (math.exp(-d1 * d1) + 0.5 * math.exp(-d2 * d2)) * spread * breath * amp
                            y = mid + (ln["y0"] - mid) * spread + env * (
                                ln["a1"] * math.sin(6.283 * ln["f1"] * u + ln["p1"] + t * ln["v1"] * speed)
                                + ln["a2"] * math.sin(6.283 * ln["f2"] * u + ln["p2"] + t * ln["v2"] * speed))
                            pts.append(u * w)
                            pts.append(y)
                        self.canvas.coords(ln["id"], *pts)
                        self.canvas.coords(ln["halo"], *pts)
                    # a floating otaku caption rides on the crest while generating
                    if self.busy:
                        phrases = memes()
                        if now - self.caption_at > 2.6:
                            self.caption_at = now
                            self.caption_idx = (self.caption_idx + 1) % len(phrases)
                            self.canvas.itemconfig(self.caption_id, text=phrases[self.caption_idx])
                        cx = min(w - 250 * self.s, max(left * w + 70 * self.s, self.crest * w))
                        self.canvas.coords(self.caption_id, cx, mid - 34 * self.s + 3 * self.s * math.sin(t * 2.2))
                    elif now < self.celebrate_until:
                        cx = min(w - 250 * self.s, max(left * w + 70 * self.s, self.crest * w))
                        self.canvas.coords(self.caption_id, cx, mid - 34 * self.s)
                    else:
                        self.canvas.itemconfig(self.caption_id, text="")
                        self.caption_idx = -1
            except tk.TclError:
                return
            self.app.after(40, self._tick)

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
            w["adv.preview"].grid(row=5, column=0, columnspan=4, sticky="w", padx=12, pady=(4, 10))
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
                        "adv.creator", "adv.star", "adv.quality", "adv.device", "adv.engine", "adv.preview",
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
                "preview": self.preview_var.get(), "advanced_open": self.advanced_open, "geometry": self.geometry(),
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
                save_settings(self.collect_settings())
            except Exception:
                pass
            self.destroy()

    app = App()
    app.mainloop()
    return 0


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
