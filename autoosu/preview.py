"""Audio preview: the song mixed with click sounds at every hit object, for checking rhythm by ear."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .beatmap import Beatmap, Hold, Slider, Spinner


def _tick(sr: int, freq: float, ms: float, amp: float) -> np.ndarray:
    n = int(sr * ms / 1000)
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * 60) * amp


def render_preview(audio: Path, beatmap: Beatmap, out: Path, song_gain: float = 0.55,
                   click_shift_ms: int = 0) -> Path:
    """Write <out> (mp3 if ffmpeg is available, else wav) with clicks on hit objects.

    click_shift_ms is added to every object time: pass the osu! timing shift so the clicks land
    on the audio transients instead of 26 ms before them.

    circle / slider head: high tick, louder + brighter on a new combo; slider end: low tick;
    spinner: soft noise for its whole duration.
    """
    from .audio_io import decode

    y, sr = decode(audio, sr=44100, mono=True)
    mix = y * song_gain
    head = _tick(sr, 1500.0, 40, 0.7)
    head_nc = _tick(sr, 2200.0, 50, 0.85)
    tail = _tick(sr, 700.0, 50, 0.5)

    def add(t_ms: int, sig: np.ndarray) -> None:
        i = int(round((t_ms + click_shift_ms) * sr / 1000))
        j = min(len(mix), i + len(sig))
        if 0 <= i < j:
            mix[i:j] += sig[: j - i]

    for o in beatmap.hit_objects:
        if isinstance(o, Spinner):
            n = int((o.end - o.time) * sr / 1000)
            noise = np.random.default_rng(0).normal(size=n) * 0.03
            add(o.time, noise)
            continue
        add(o.time, head_nc if o.new_combo else head)
        if isinstance(o, (Slider, Hold)):
            add(o.end_time, tail)

    peak = float(np.abs(mix).max()) or 1.0
    mix = (mix / peak * 0.95).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".mp3":
        from .audio_io import transcode_mp3 as encode_mp3

        wav = out.with_suffix(".tmp.wav")
        sf.write(str(wav), mix, sr)
        try:
            encode_mp3(wav, out, bitrate="160k")
        except Exception:
            out = out.with_suffix(".wav")
            sf.write(str(out), mix, sr)
        finally:
            wav.unlink(missing_ok=True)
    else:
        out = out.with_suffix(".wav")
        sf.write(str(out), mix, sr)
    return out
