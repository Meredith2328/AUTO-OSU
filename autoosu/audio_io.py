"""Input normalisation: decode any audio or video file to a waveform, transcode for osu!.

Two decoders, tried in order:
1. libsndfile (via soundfile): wav, flac, ogg/vorbis, ogg/opus, mp3, aiff, w64, caf ...
2. ffmpeg (on PATH, or the binary shipped with imageio-ffmpeg inside the exe): everything else
   (m4a/aac, wma, ape, alac, opus, and the audio track of mp4/mkv/webm/mov/avi ...).

osu! itself only plays mp3 and ogg/vorbis, so `prepare_for_osu` copies those through and transcodes
anything else to a 192 kbps mp3.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

AUDIO_EXTS = (
    ".mp3", ".ogg", ".oga", ".opus", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".aac", ".wma", ".ape",
    ".alac", ".caf", ".w64", ".wv", ".tta", ".mpc", ".ac3", ".dts", ".amr", ".mid",
)
VIDEO_EXTS = (".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi", ".flv", ".ts", ".wmv", ".3gp")
SUPPORTED_EXTS = tuple(e for e in AUDIO_EXTS if e != ".mid") + VIDEO_EXTS
OSU_NATIVE = {".mp3", ".ogg"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


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


def _decode_soundfile(path: Path, sr: Optional[int], mono: bool) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    data, native = sf.read(str(path), dtype="float32", always_2d=True)
    y = data.mean(axis=1) if mono else data.T
    if sr and sr != native:
        import librosa

        y = librosa.resample(y, orig_sr=native, target_sr=sr)
        native = sr
    return np.ascontiguousarray(y, dtype=np.float32), native


def _decode_ffmpeg(path: Path, sr: Optional[int], mono: bool) -> Tuple[np.ndarray, int]:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(f"cannot decode {path.suffix or path.name}: install ffmpeg or use mp3/ogg/wav/flac")
    rate = sr or 44100
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-vn", "-map", "0:a:0",
           "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(rate), "-ac", "1" if mono else "2", "-"]
    proc = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg could not decode {path.name}: {err[-1] if err else 'no audio stream'}")
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if not mono:
        y = y.reshape(-1, 2).T
    return np.ascontiguousarray(y), rate


def decode(path: str | Path, sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Waveform (float32 in -1..1, mono (n,) or stereo (2, n)) and sample rate of any supported file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    errors = []
    if path.suffix.lower() not in VIDEO_EXTS:
        try:
            y, rate = _decode_soundfile(path, sr, mono)
            if y.size:
                return y, rate
            errors.append("empty audio")
        except Exception as e:  # noqa: BLE001 - fall through to ffmpeg
            errors.append(str(e).splitlines()[0][:120])
    try:
        return _decode_ffmpeg(path, sr, mono)
    except Exception as e:  # noqa: BLE001
        errors.append(str(e))
    raise RuntimeError("cannot decode %s (%s)" % (path.name, "; ".join(errors)))


def probe_duration(path: str | Path) -> Optional[float]:
    """Length in seconds when cheaply available (soundfile header), else None."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def transcode_mp3(src: Path, dst: Path, bitrate: str = "192k") -> Path:
    """Any input -> mp3 (44.1 kHz). ffmpeg when available, else libsndfile for the formats it reads."""
    ffmpeg = find_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-v", "error", "-nostdin", "-i", str(src), "-vn", "-map", "0:a:0",
               "-codec:a", "libmp3lame", "-b:a", bitrate, "-ar", "44100", str(dst)]
        proc = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError(f"ffmpeg could not convert {src.name}: {err[-1] if err else 'unknown error'}")
        return dst
    import soundfile as sf

    if "MP3" not in sf.available_formats():
        raise RuntimeError(f"cannot convert {src.suffix} to mp3: install ffmpeg or use an mp3/ogg file")
    y, rate = _decode_soundfile(src, 44100, mono=False)
    sf.write(str(dst), y.T, rate, format="MP3")
    return dst


def _is_vorbis_ogg(path: Path) -> bool:
    try:
        import soundfile as sf

        return sf.info(str(path)).subtype.upper() == "VORBIS"
    except Exception:
        return False


def prepare_for_osu(src: str | Path, workdir: Path) -> Path:
    """The song as osu! can play it: mp3 and ogg/vorbis are copied, everything else becomes audio.mp3."""
    src = Path(src)
    workdir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    if ext == ".mp3" or (ext == ".ogg" and _is_vorbis_ogg(src)):
        dst = workdir / f"audio{ext}"
        shutil.copyfile(src, dst)
        return dst
    return transcode_mp3(src, workdir / "audio.mp3")
