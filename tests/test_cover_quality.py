import subprocess
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from autoosu import audio_io, package
from autoosu.generate import generate

from synth import make_song

FFMPEG = audio_io.find_ffmpeg()


@pytest.fixture(scope="module")
def wav(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cover")
    return make_song(d / "Cover Artist - Cover Song.wav", bpm=120.0, bars=8, offset_s=0.3)


def cover_png(path: Path) -> bytes:
    Image.new("RGB", (640, 480), (230, 169, 60)).save(path, "PNG")
    return path.read_bytes()


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")
def test_flac_source_gets_320k_and_cover_background(wav, tmp_path):
    flac = wav.with_suffix(".flac")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(wav), str(flac)], check=True)
    from mutagen.flac import FLAC, Picture

    f = FLAC(str(flac))
    pic = Picture()
    pic.data = cover_png(tmp_path / "cover.png")
    pic.mime = "image/png"
    pic.type = 3
    f.add_picture(pic)
    f["title"] = "Cover Song"
    f["artist"] = "Cover Artist"
    f.save()
    assert audio_io.choose_bitrate(flac) == "320k"
    assert package.extract_cover(flac) is not None
    res = generate(flac, ["Hard"], tmp_path / "out", log=lambda *_: None)
    with zipfile.ZipFile(res.osz) as z:
        names = z.namelist()
        assert "bg.png" in names and "audio.mp3" in names
        osu = z.read([n for n in names if n.endswith(".osu")][0]).decode("utf-8")
    assert '0,0,"bg.png",0,0' in osu


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")
def test_probe_reads_codec_and_rate(wav):
    low = wav.with_suffix(".low.m4a")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(wav), "-c:a", "aac", "-b:a", "96k", str(low)], check=True)
    info = audio_io.probe_stream(low)
    assert info.get("codec") == "aac" and 0 < info.get("kbps", 0) <= 96
    assert audio_io.choose_bitrate(low) == "192k"


def test_bitrate_choice_follows_the_source(monkeypatch, tmp_path):
    src = tmp_path / "x.m4a"
    src.write_bytes(b"")
    cases = [({"codec": "flac"}, "320k"), ({"codec": "pcm_s16le"}, "320k"), ({"codec": "aac", "kbps": 96}, "192k"),
             ({"codec": "aac", "kbps": 200}, "224k"), ({"codec": "aac", "kbps": 256}, "256k"),
             ({"codec": "opus", "kbps": 400}, "320k"), ({}, "192k")]
    for info, expected in cases:
        monkeypatch.setattr(audio_io, "probe_stream", lambda _p, info=info: dict(info))
        assert audio_io.choose_bitrate(src) == expected, info


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")
def test_video_frame_becomes_background(wav, tmp_path):
    mp4 = wav.with_suffix(".mp4")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x4fb8ff:s=320x240:r=5", "-i", str(wav),
                    "-shortest", "-c:v", "libx264", "-c:a", "aac", "-map", "0:v", "-map", "1:a", str(mp4)], check=True)
    res = generate(mp4, ["Normal"], tmp_path / "out", log=lambda *_: None)
    with zipfile.ZipFile(res.osz) as z:
        assert "bg.jpg" in z.namelist()
