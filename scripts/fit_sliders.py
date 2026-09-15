"""Post-fix a generated .osu: keep every slider path inside the playfield.

Sliders whose drawn path leaves the 512x384 field are re-fitted around their head (mirror /
rotate the shape; if nothing fits, shrink it and lower the slider velocity with a green line).
Optionally takes the timing points (with kiai flags) and metadata from another .osu of the same
song, and packs the result with the audio into a new .osz.

    python scripts/fit_sliders.py --osu coord.osu --timing-from rhythm.osz --audio audio.mp3 --out out/play
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoosu.beatmap import PLAYFIELD_H, PLAYFIELD_W  # noqa: E402
from autoosu.package import sanitize  # noqa: E402
from autoosu.sliderpath import fit_slider, path_points  # noqa: E402


def read_osu(path: str | Path) -> str:
    p = Path(path)
    if p.suffix.lower() == ".osz":
        with zipfile.ZipFile(p) as z:
            name = next(n for n in z.namelist() if n.endswith(".osu"))
            return z.read(name).decode("utf-8")
    return p.read_text(encoding="utf-8")


def sections(text: str) -> dict:
    out, cur = {}, None
    for ln in text.splitlines():
        if ln.startswith("[") and ln.endswith("]"):
            cur = ln
            out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return out


def kv(lines, key):
    for ln in lines:
        if ln.startswith(key + ":"):
            return ln.split(":", 1)[1].strip()
    return None


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--osu", required=True, help=".osu (or .osz) whose hit objects are fixed")
    ap.add_argument("--timing-from", help=".osu/.osz providing [TimingPoints] and metadata (kiai flags)")
    ap.add_argument("--audio", help="audio file to pack into the .osz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--creator", default="AUTO-OSU")
    ap.add_argument("--margin", type=float, default=2.0)
    a = ap.parse_args()

    src = sections(read_osu(a.osu))
    meta_src = sections(read_osu(a.timing_from)) if a.timing_from else src
    diff = src["[Difficulty]"]
    sm = float(kv(diff, "SliderMultiplier"))
    tps = [ln for ln in meta_src["[TimingPoints]"] if ln.strip()]
    parsed = []
    for ln in tps:
        p = ln.split(",")
        parsed.append([float(p[0]), float(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6]), int(p[7])])

    def red_at(t):
        best = None
        for tp in parsed:
            if tp[6] == 1 and (tp[0] <= t or best is None):
                best = tp
        return best

    def sv_at(t):
        sv = 1.0
        for tp in sorted(parsed, key=lambda q: q[0]):
            if tp[0] <= t and tp[6] == 0:
                sv = -100.0 / tp[1]
        return sv

    def kiai_at(t):
        k = 0
        for tp in sorted(parsed, key=lambda q: q[0]):
            if tp[0] <= t:
                k = tp[7]
        return k

    fixed, added, objs = 0, 0, []
    for ln in src["[HitObjects]"]:
        if not ln.strip():
            continue
        p = ln.split(",")
        x, y, t, typ = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        if not typ & 2:
            objs.append(ln)
            continue
        curve = p[5].split("|")
        ct = curve[0]
        anchors = [(x, y)] + [tuple(map(int, c.split(":"))) for c in curve[1:]]
        repeats, length = int(p[6]), float(p[7])
        pts = path_points(ct, anchors)
        inside = (pts[:, 0].min() >= 0 and pts[:, 0].max() <= PLAYFIELD_W
                  and pts[:, 1].min() >= 0 and pts[:, 1].max() <= PLAYFIELD_H)
        if inside:
            objs.append(ln)
            continue
        f = fit_slider(ct, anchors, length, margin=a.margin)
        fixed += 1
        p[5] = ct + "|" + "|".join(f"{px}:{py}" for px, py in f.anchors[1:])
        p[7] = f"{f.length:.4f}"
        objs.append(",".join(p))
        if f.sv_scale < 0.999:
            red = red_at(t)
            sv = sv_at(t)
            per_slide = length / (100.0 * sm * sv) * red[1]
            end = int(round(t + per_slide * repeats))
            k = kiai_at(t)
            parsed.append([t, -100.0 / (sv * f.sv_scale), 4, 2, 0, 70, 0, k])
            parsed.append([end, -100.0 / sv, 4, 2, 0, 70, 0, kiai_at(end)])
            added += 2
            print(f"  slider at {t}: shortened to {f.sv_scale:.2f}x, green lines at {t} and {end}")

    # rebuild the file
    out_lines = []
    for sec in ["[General]", "[Editor]", "[Metadata]", "[Difficulty]", "[Events]", "[TimingPoints]",
                "[Colours]", "[HitObjects]"]:
        out_lines.append(sec)
        if sec == "[TimingPoints]":
            parsed.sort(key=lambda q: (q[0], -q[6]))
            for tp in parsed:
                out_lines.append(f"{int(tp[0])},{tp[1]:.6f},{tp[2]},{tp[3]},{tp[4]},{tp[5]},{tp[6]},{tp[7]}")
        elif sec == "[HitObjects]":
            out_lines.extend(objs)
        elif sec == "[Metadata]":
            for ln in meta_src[sec]:
                if ln.startswith("Creator:"):
                    ln = f"Creator:{a.creator}"
                elif ln.startswith("Tags:"):
                    ln = "Tags:autoosu generated"
                if ln.strip():
                    out_lines.append(ln)
        elif sec in ("[General]", "[Events]", "[Colours]"):
            out_lines.extend(ln for ln in meta_src.get(sec, src.get(sec, [])) if ln.strip())
        else:
            out_lines.extend(ln for ln in src[sec] if ln.strip())
        out_lines.append("")
    text = "osu file format v14\r\n\r\n" + "\r\n".join(out_lines)

    meta = meta_src["[Metadata]"]
    artist, title, version = kv(meta, "Artist"), kv(meta, "Title"), kv(src["[Metadata]"], "Version")
    name = sanitize(f"{artist} - {title} ({a.creator}) [{version}]")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    osu_path = out_dir / f"{name}.osu"
    osu_path.write_text(text, encoding="utf-8")
    print(f"{fixed} slider(s) re-fitted, {added} green line(s) added -> {osu_path.name}")
    if a.audio:
        audio = Path(a.audio)
        general = meta_src["[General]"]
        audio_name = kv(general, "AudioFilename") or audio.name
        osz = out_dir / sanitize(f"{artist} - {title} [{version}] ({a.creator}).osz")
        with zipfile.ZipFile(osz, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(audio, audio_name)
            z.writestr(f"{name}.osu", text.encode("utf-8"))
        print(f"-> {osz}")


if __name__ == "__main__":
    main()
