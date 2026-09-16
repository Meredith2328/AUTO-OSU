"""Desktop interface (customtkinter): pick a song, tick difficulties, press Generate.

Runs the pipeline in a worker thread and reports progress through a queue; all texts come from
`autoosu.i18n` so the window can switch between Chinese and English on the fly. Light and dark
looks share one palette taken from kanzei's OC: graphite black, amber-gold eyes and the electric-blue
cheek glow. The progress bar eases towards its target and the Generate button breathes while busy;
nothing else animates.
"""
from __future__ import annotations

import json
import math
import os
import queue
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
from .i18n import AUTHOR, language, set_language, tr
from .models import MODELS, app_root, ensure_model, find_model

from .audio_io import SUPPORTED_EXTS

AUDIO_EXT = tuple("*" + e for e in SUPPORTED_EXTS)
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
        img = img.crop(((img.width - side) // 2, 0, (img.width - side) // 2 + side, side))
        big = size * 4
        img = img.resize((big, big), Image.LANCZOS)
        disc = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        ImageDraw.Draw(disc).ellipse((0, 0, big - 1, big - 1), fill=(36, 36, 40, 255))   # graphite backdrop
        disc.alpha_composite(img)
        mask = Image.new("L", (big, big), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, big - 1, big - 1), fill=255)
        disc.putalpha(mask)
        ImageDraw.Draw(disc).ellipse((2, 2, big - 3, big - 3), outline=(230, 169, 60, 255), width=max(4, big // 40))
        return disc.resize((size, size), Image.LANCZOS)
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
        """Static banner: avatar (when assets/avatar.png exists), title, subtitle and the two toggles."""

        def __init__(self, app: "App", master) -> None:
            self.app = app
            self.frame = ctk.CTkFrame(master, fg_color="transparent")
            self.frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 4))
            self.frame.grid_columnconfigure(1, weight=1)
            img = _avatar_image(64)
            self._avatar = None
            if img is not None:
                self._avatar = ctk.CTkImage(light_image=img, dark_image=img, size=(64, 64))
                ctk.CTkLabel(self.frame, image=self._avatar, text="").grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 14))
            self.title = ctk.CTkLabel(self.frame, font=app.font_title, anchor="w", text_color=PALETTE["gold"])
            self.title.grid(row=0, column=1, sticky="sw", pady=(0, 1))
            self.subtitle = ctk.CTkLabel(self.frame, font=app.font, anchor="w", text_color=PALETTE["muted"])
            self.subtitle.grid(row=1, column=1, sticky="nw", pady=(1, 0))
            btns = ctk.CTkFrame(self.frame, fg_color="transparent")
            btns.grid(row=0, column=2, rowspan=2, sticky="ne")
            btn_kw = dict(width=92, height=30, font=app.font, fg_color="transparent", border_width=1,
                          border_color=PALETTE["border"])
            self.lang_btn = ctk.CTkButton(btns, text_color=PALETTE["gold"], command=app.toggle_language, **btn_kw)
            self.lang_btn.pack(side="left", padx=(0, 8))
            self.theme_btn = ctk.CTkButton(btns, text_color=PALETTE["cyan"], command=app.toggle_theme, **btn_kw)
            self.theme_btn.pack(side="left")

        def set_texts(self, title: str, subtitle: str) -> None:
            self.title.configure(text=title)
            self.subtitle.configure(text=subtitle)

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
            self.font_title = ctk.CTkFont(family=_ui_font(), size=25, weight="bold")
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
            w["song.hint"] = ctk.CTkLabel(song, font=self.font, anchor="w", justify="left", wraplength=980, text_color=PALETTE["muted"])
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
            """Title-bar / taskbar icon: the bundled .ico (set after customtkinter installs its own)."""
            ico = ASSETS / "icon.ico"
            try:
                if ico.exists() and sys.platform == "win32":
                    self.iconbitmap(str(ico))
                    self.after(1000, lambda: self.iconbitmap(str(ico)))
                elif AVATAR.exists():
                    from PIL import Image, ImageTk

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


FONTS_DIR = ASSETS / "fonts"
_FONT_FAMILY: Optional[str] = None


def _register_fonts() -> Optional[str]:
    """Load the bundled UI font for this process (Windows) and return its family name, else None."""
    global _FONT_FAMILY
    if _FONT_FAMILY is not None:
        return _FONT_FAMILY or None
    _FONT_FAMILY = ""
    if sys.platform == "win32" and FONTS_DIR.is_dir():
        try:
            import ctypes

            n = 0
            for f in sorted(FONTS_DIR.glob("*.ttf")):
                n += ctypes.windll.gdi32.AddFontResourceExW(str(f), 0x10, 0)   # FR_PRIVATE
            if n:
                _FONT_FAMILY = "AUTO-OSU Sans"
        except Exception:
            _FONT_FAMILY = ""
    return _FONT_FAMILY or None


def _ui_font() -> str:
    fam = _register_fonts()
    if fam:
        return fam
    if sys.platform == "win32":
        try:
            import tkinter.font as tkfont

            fams = set(tkfont.families())
            for cand in ("Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI"):
                if cand in fams:
                    return cand
        except Exception:
            pass
        return "Segoe UI"
    return "Helvetica"


if __name__ == "__main__":
    sys.exit(run_gui())
