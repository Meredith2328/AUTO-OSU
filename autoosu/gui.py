"""Desktop interface (customtkinter): pick a song, tick difficulties, press Generate.

Runs the pipeline in a worker thread and reports progress through a queue; all texts come from
`autoosu.i18n` so the window can switch between Chinese and English on the fly.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Dict, Optional

from . import __version__
from .difficulty import PRESETS
from .i18n import AUTHOR, language, set_language, tr
from .models import MODELS, app_root, candidate_dirs, ensure_model, find_model

AUDIO_EXT = ("*.mp3", "*.ogg", "*.wav", "*.flac", "*.m4a", "*.aac", "*.wma", "*.opus")
QUALITY_STEPS = {"fast": 50, "normal": 100, "high": 200}
ASSETS = Path(__file__).resolve().parent / "assets"
AVATAR = ASSETS / "avatar.png"          # kanzei's OC; header avatar + window icon when present
MUTED = "#9a937f"                       # secondary text on the dark gold theme
GOLD = "#d4a52c"


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


def run_gui() -> int:
    import customtkinter as ctk

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
    except Exception:  # drag and drop is optional
        DND_FILES = TkinterDnD = None

    ctk.set_appearance_mode("dark")
    theme = ASSETS / "theme.json"
    ctk.set_default_color_theme(str(theme) if theme.exists() else "blue")

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
            self.widgets: Dict[str, object] = {}
            self.q: "queue.Queue" = queue.Queue()
            self.busy = False
            self.last_osz: Optional[Path] = None
            self.advanced_open = bool(self.settings.get("advanced_open", False))
            self.font = ctk.CTkFont(family=_ui_font(), size=13)
            self.font_bold = ctk.CTkFont(family=_ui_font(), size=15, weight="bold")
            self.font_title = ctk.CTkFont(family=_ui_font(), size=22, weight="bold")
            self.geometry(self.settings.get("geometry", "760x760"))
            self.minsize(640, 600)
            self.protocol("WM_DELETE_WINDOW", self.on_close)
            self.build()
            self.refresh_texts()
            self.refresh_models()
            self.after(100, self.poll)

        # ------------------------------------------------------------------ layout
        def build(self) -> None:
            w = self.widgets
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(6, weight=1)

            head = ctk.CTkFrame(self, fg_color="transparent")
            head.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 4))
            head.grid_columnconfigure(1, weight=1)
            avatar = _load_avatar(ctk, 60)
            if avatar is not None:
                self._avatar = avatar
                ctk.CTkLabel(head, image=avatar, text="").grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 14))
            w["title"] = ctk.CTkLabel(head, font=self.font_title, anchor="w", text_color=GOLD)
            w["title"].grid(row=0, column=1, sticky="w")
            w["lang"] = ctk.CTkButton(head, width=90, font=self.font, command=self.toggle_language,
                                      fg_color="transparent", border_width=1, border_color="#5a5030", text_color=GOLD)
            w["lang"].grid(row=0, column=2, sticky="e")
            w["subtitle"] = ctk.CTkLabel(head, font=self.font, anchor="w", text_color=MUTED)
            w["subtitle"].grid(row=1, column=1, columnspan=2, sticky="w")
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
            w["song.hint"] = ctk.CTkLabel(song, font=self.font, anchor="w", text_color=MUTED)
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
            w["diff.hint"] = ctk.CTkLabel(diff, font=self.font, anchor="w", text_color=MUTED)
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
                                            text_color=GOLD, hover=False, command=self.toggle_advanced)
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
            w["status"] = ctk.CTkLabel(run, font=self.font, anchor="w", wraplength=680, justify="left")
            w["status"].grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 4))
            btns = ctk.CTkFrame(run, fg_color="transparent")
            btns.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
            w["run.generate"] = ctk.CTkButton(btns, height=42, width=200, font=self.font_bold, command=self.start)
            w["run.generate"].pack(side="left")
            w["run.open_osz"] = ctk.CTkButton(btns, height=42, font=self.font, fg_color="transparent",
                                              border_width=1, border_color="#5a5030", text_color=GOLD,
                                              command=lambda: self.last_osz and open_path(self.last_osz), state="disabled")
            w["run.open_osz"].pack(side="left", padx=8)
            w["run.open_folder"] = ctk.CTkButton(btns, height=42, font=self.font, fg_color="transparent",
                                                 border_width=1, border_color="#5a5030", text_color=GOLD,
                                                 command=lambda: open_path(Path(self.out_var.get())))
            w["run.open_folder"].pack(side="left")
            self.log_box = ctk.CTkTextbox(run, font=ctk.CTkFont(family="Consolas", size=12), height=120)
            self.log_box.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=12, pady=(4, 10))
            self.log_box.configure(state="disabled")

            w["about"] = ctk.CTkLabel(self, font=ctk.CTkFont(family=_ui_font(), size=11), text_color="#8a8478")
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
            w["title"].configure(text=tr("app.title"))
            w["subtitle"].configure(text=tr("app.subtitle"))
            w["lang"].configure(text=tr("lang.toggle"))
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

        # ------------------------------------------------------------------ actions
        def toggle_language(self) -> None:
            set_language("en" if language() == "zh" else "zh")
            self.refresh_texts()

        def toggle_advanced(self) -> None:
            self.advanced_open = not self.advanced_open
            if self.advanced_open:
                self.adv_frame.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 6))
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
                                             text_color="#ece7d8")
                w["models.download"].grid_remove()
            else:
                w["models.status"].configure(text=tr("models.missing"), text_color="#f0a060")
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
                "language": language(), "song": self.song_var.get(), "out_dir": self.out_var.get(),
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
            self.progress.set(0)
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
                        self.progress.set(max(0.0, min(1.0, item[1])))
                        self.widgets["status"].configure(text=self.status_text(item[2]))
                    elif kind == "models":
                        self.set_busy(False)
                        self.progress.set(0)
                        self.widgets["status"].configure(text=tr("status.idle"))
                        self.refresh_models()
                    elif kind == "done":
                        res = item[1]
                        self.set_busy(False)
                        self.progress.set(1.0)
                        self.last_osz = res.osz
                        self.widgets["run.open_osz"].configure(state="normal")
                        self.widgets["status"].configure(
                            text=tr("status.done", file=res.osz.name, secs=res.elapsed_s, device=res.device))
                        self.log(tr("result.summary", bpm=res.timing.bpm, n=len(res.diffs)))
                        for d in res.diffs:
                            s = d.summary()
                            self.log(tr("result.diff", name=d.preset.name, objects=s["objects"], sliders=s["sliders"], nps=s["nps"]))
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


def _load_avatar(ctk, size: int):
    """Round-cropped CTkImage of assets/avatar.png, or None when the file is absent."""
    if not AVATAR.exists():
        return None
    try:
        from PIL import Image, ImageDraw

        img = Image.open(AVATAR).convert("RGBA")
        side = min(img.size)
        img = img.crop(((img.width - side) // 2, 0, (img.width - side) // 2 + side, side)).resize((size * 4, size * 4), Image.LANCZOS)
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).ellipse((0, 0, img.width - 1, img.height - 1), fill=255)
        img.putalpha(mask)
        return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
    except Exception:
        return None


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
