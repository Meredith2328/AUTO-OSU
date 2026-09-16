"""Metadata reading, audio conversion and .osz packaging."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import List, Tuple

from .audio_io import find_ffmpeg, prepare_for_osu, transcode_mp3  # noqa: F401  (re-exported)
from .beatmap import Beatmap

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


encode_mp3 = transcode_mp3


def prepare_audio(src: Path, workdir: Path) -> Path:
    """Copy (mp3 / ogg-vorbis) or transcode (anything else) the song so osu! can play it."""
    return prepare_for_osu(src, workdir)


def write_osz(beatmaps: List[Beatmap], audio: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    first = beatmaps[0]
    osz = out_dir / sanitize(f"{first.artist} - {first.title} ({first.creator}).osz")
    with zipfile.ZipFile(osz, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(audio, audio.name)
        for bm in beatmaps:
            zf.writestr(sanitize(bm.osu_filename()), bm.to_osu().encode("utf-8"))
    return osz
