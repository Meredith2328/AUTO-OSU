"""Metadata reading, audio conversion and .osz packaging."""
from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from .beatmap import Beatmap

OSU_AUDIO_EXT = {".mp3", ".ogg"}
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize(name: str) -> str:
    name = _ILLEGAL.sub("", name).strip().rstrip(".")
    return name or "untitled"


def read_metadata(path: Path) -> Tuple[str, str]:
    """Return (title, artist) from tags, falling back to an 'Artist - Title' file name."""
    title = artist = ""
    try:
        import mutagen

        tags = mutagen.File(str(path), easy=True)
        if tags:
            title = (tags.get("title") or [""])[0]
            artist = (tags.get("artist") or [""])[0]
    except Exception:
        pass
    stem = path.stem
    if not title:
        if " - " in stem:
            a, t = stem.split(" - ", 1)
            artist = artist or a.strip()
            title = t.strip()
        else:
            title = stem
    return sanitize(title), sanitize(artist or "Unknown Artist")


def find_ffmpeg() -> Optional[str]:
    """ffmpeg on PATH, else the binary shipped by imageio-ffmpeg (bundled in the exe), else None."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def encode_mp3(src: Path, dst: Path, bitrate: str = "192k") -> Path:
    """Transcode any audio file to mp3: ffmpeg if available, else libsndfile (mp3/ogg/wav/flac input)."""
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vn",
                        "-codec:a", "libmp3lame", "-b:a", bitrate, str(dst)], check=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return dst
    import soundfile as sf

    if "MP3" not in sf.available_formats():
        raise RuntimeError(f"cannot convert {src.suffix} to mp3: install ffmpeg or use an mp3/ogg file")
    data, sr = sf.read(str(src), always_2d=True)
    sf.write(str(dst), data, sr, format="MP3")
    return dst


def prepare_audio(src: Path, workdir: Path) -> Path:
    """Copy (or transcode to mp3) the song so osu! can play it. Returns the file in workdir."""
    workdir.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in OSU_AUDIO_EXT:
        dst = workdir / f"audio{src.suffix.lower()}"
        shutil.copyfile(src, dst)
        return dst
    return encode_mp3(src, workdir / "audio.mp3")


def write_osz(beatmaps: List[Beatmap], audio: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    first = beatmaps[0]
    osz = out_dir / sanitize(f"{first.artist} - {first.title} ({first.creator}).osz")
    with zipfile.ZipFile(osz, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(audio, audio.name)
        for bm in beatmaps:
            zf.writestr(sanitize(bm.osu_filename()), bm.to_osu().encode("utf-8"))
    return osz
