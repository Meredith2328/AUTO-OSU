import subprocess
from pathlib import Path

import numpy as np
import pytest

from autoosu import audio_io
from autoosu.generate import generate

from synth import make_song

FFMPEG = audio_io.find_ffmpeg()


@pytest.fixture(scope="module")
def wav(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("formats")
    return make_song(d / "Test Artist - Format Song.wav", bpm=120.0, bars=8, offset_s=0.3)


def convert(src: Path, ext: str, *extra: str) -> Path:
    dst = src.with_suffix(ext)
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(src), *extra, str(dst)], check=True)
    return dst


def test_decode_wav_via_soundfile(wav):
    y, sr = audio_io.decode(wav, sr=22050)
    assert sr == 22050 and y.ndim == 1 and y.dtype == np.float32
    ref = audio_io.probe_duration(wav)
    assert abs(len(y) / sr - ref) < 0.01


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")
@pytest.mark.parametrize("ext,args", [
    (".m4a", ("-c:a", "aac")),
    (".opus", ("-c:a", "libopus")),
    (".wma", ("-c:a", "wmav2")),
    (".mp4", ("-f", "lavfi", "-i", "color=c=black:s=64x64:r=5", "-shortest", "-c:v", "libx264", "-c:a", "aac")),
])
def test_decode_other_formats(wav, ext, args):
    if ext == ".mp4":   # video container: the song is the second input, the black frames the first
        dst = wav.with_suffix(ext)
        subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=5",
                        "-i", str(wav), "-shortest", "-c:v", "libx264", "-c:a", "aac", "-map", "0:v", "-map", "1:a", str(dst)], check=True)
    else:
        dst = convert(wav, ext, *args)
    y, sr = audio_io.decode(dst, sr=22050)
    ref = audio_io.probe_duration(wav)
    assert sr == 22050 and abs(len(y) / sr - ref) < 0.15, (ext, len(y) / sr, ref)
    assert float(np.abs(y).max()) > 0.05


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")
def test_prepare_for_osu_transcodes(wav, tmp_path):
    m4a = convert(wav, ".m4a", "-c:a", "aac")
    out = audio_io.prepare_for_osu(m4a, tmp_path / "work")
    assert out.suffix == ".mp3" and out.stat().st_size > 10_000
    y, sr = audio_io.decode(out, sr=22050)
    assert abs(len(y) / sr - audio_io.probe_duration(wav)) < 0.15


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")
def test_generate_from_m4a(wav, tmp_path):
    m4a = convert(wav, ".m4a", "-c:a", "aac")
    res = generate(m4a, ["Normal"], tmp_path / "out", log=lambda *_: None)
    assert res.osz.exists() and abs(res.timing.bpm - 120) < 0.5
