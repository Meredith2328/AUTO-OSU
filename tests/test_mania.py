from __future__ import annotations

from types import SimpleNamespace
import subprocess
import sys
import zipfile

import numpy as np

from autoosu.beatmap import Circle, Hold
from autoosu.difficulty import PRESETS
from autoosu.mania import LANE_X, STYLES, ManiaStyle, build_mania_objects
from autoosu.rhythm import Sections, Tick
from autoosu.timing import Timing
from autoosu.cli import main


def ticks(count=64):
    return [
        Tick(beat=i * 0.5, time=500 + i * 250, kick=0.9 if i % 2 == 0 else 0.35,
             snare=0.8 if i % 2 else 0.25, hat=0.4, melody=0.7, sustain_beats=1.5, loud=0.8)
        for i in range(count)
    ]


def generated(monkeypatch, difficulty="Insane", seed=7):
    source = ticks()
    monkeypatch.setattr("autoosu.mania.tick_features", lambda *args: source)
    monkeypatch.setattr("autoosu.mania.select_ticks", lambda *args: source)
    analysis = SimpleNamespace(onsets=[])
    timing = Timing(bpm=120, offset_ms=500)
    sections = Sections(np.full(16, 0.9), [])
    return build_mania_objects(analysis, timing, PRESETS[difficulty], np.random.default_rng(seed), sections)


def signature(objects):
    return [(type(obj).__name__, obj.x, obj.time, obj.end_time) for obj in objects]


def test_mania_uses_exactly_four_lanes_and_valid_hold_lines(monkeypatch):
    events, objects = generated(monkeypatch)
    assert events and objects
    assert set(obj.x for obj in objects) == set(LANE_X)
    assert all(obj.y == 192 for obj in objects)
    assert any(isinstance(obj, Hold) for obj in objects)
    assert any(isinstance(obj, Circle) for obj in objects)
    for obj in objects:
        fields = obj.to_line().split(",")
        assert int(fields[3]) & (128 if isinstance(obj, Hold) else 1)
        if isinstance(obj, Hold):
            assert obj.end > obj.time
            assert fields[5].startswith(f"{obj.end}:")


def test_chords_never_duplicate_a_lane_and_lane_intervals_do_not_overlap(monkeypatch):
    _, objects = generated(monkeypatch, seed=11)
    at_time = {}
    for obj in objects:
        at_time.setdefault(obj.time, []).append(obj.x)
    assert any(len(lanes) > 1 for lanes in at_time.values())
    assert all(len(lanes) == len(set(lanes)) for lanes in at_time.values())

    for x in LANE_X:
        lane = sorted((obj for obj in objects if obj.x == x), key=lambda obj: obj.time)
        assert all(previous.end_time < following.time for previous, following in zip(lane, lane[1:]))


def test_seed_is_repeatable_and_changes_lane_pattern(monkeypatch):
    _, a = generated(monkeypatch, seed=23)
    _, b = generated(monkeypatch, seed=23)
    _, c = generated(monkeypatch, seed=24)
    assert signature(a) == signature(b)
    assert signature(a) != signature(c)


def test_beatmap_serialises_mania_mode_and_four_keys(monkeypatch):
    from autoosu.generate import build_beatmap

    _, objects = generated(monkeypatch, difficulty="Hard")
    beatmap = build_beatmap(PRESETS["Hard"], Timing(120, 500), objects, [], "audio.mp3",
                            "Title", "Artist", "Mapper", mode="mania4k")
    text = beatmap.to_osu()
    assert "Mode: 3" in text
    assert "CircleSize:4" in text
    assert "Version:Hard 4K" in text
    assert "rules-generated mania 4k" in text


def test_standard_serialisation_stays_the_default():
    from autoosu.beatmap import Beatmap

    text = Beatmap("audio.mp3", "Title", "Artist", "Hard").to_osu()
    assert "Mode: 0" in text
    assert "CircleSize:4" in text


def test_cli_rejects_standard_model_options_for_mania(tmp_path, capsys):
    audio = tmp_path / "song.wav"
    audio.touch()
    for options in (("--download",), ("--density=4",), ("--device", "cuda"), ("--coord-steps=10",)):
        assert main([str(audio), "--mode", "mania4k", "-o", str(tmp_path / "out"), *options]) == 2
        assert "rules-only" in capsys.readouterr().err
        assert not (tmp_path / "out").exists()


def test_python_module_cli_returns_error_without_output(tmp_path):
    audio = tmp_path / "song.wav"
    audio.touch()
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-m", "autoosu", str(audio), "--mode", "mania4k",
                           "--density=4", "-o", str(out)], capture_output=True, text=True)
    assert proc.returncode == 2
    assert "rules-only" in proc.stderr
    assert not out.exists()


def test_mania_objects_stay_within_shifted_audio_bounds(monkeypatch):
    source = [Tick(beat=0, time=0, kick=0.9, snare=0.1, hat=0.1,
                   melody=0.8, sustain_beats=2, loud=0.8),
              Tick(beat=9, time=4500, kick=0.9, snare=0.1,
                                 hat=0.1, melody=0.8, sustain_beats=2,
                                 loud=0.8)]
    monkeypatch.setattr("autoosu.mania.tick_features", lambda *args: source)
    monkeypatch.setattr("autoosu.mania.select_ticks", lambda *args: source)
    monkeypatch.setitem(STYLES, "Easy", ManiaStyle(0, 0, 1, 2))
    _, objects = build_mania_objects(SimpleNamespace(onsets=[]), Timing(120, 0),
                                      PRESETS["Easy"], np.random.default_rng(1),
                                      min_time_ms=26, max_time_ms=5000)
    assert len(objects) == 1
    assert isinstance(objects[0], Hold)
    assert objects[0].time >= 26
    assert objects[0].end_time <= 5000


def test_generate_mania_osz_through_real_pipeline(tmp_path):
    from synth import make_song
    from autoosu.generate import generate
    from autoosu.audio_io import decode

    wav = make_song(tmp_path / "Test Artist - Mania Song.wav", bpm=128, bars=6)
    result = generate(wav, ["Easy", "Hard"], tmp_path / "out", seed=13,
                      bpm=128, offset_ms=500, mode="mania4k", log=lambda *_: None)
    assert result.device == "cpu"
    assert result.osz.name.endswith(" [mania 4K].osz")
    with zipfile.ZipFile(result.osz) as archive:
        audio = next(name for name in archive.namelist() if name.endswith(".mp3"))
        extracted = archive.extract(audio, tmp_path / "unpacked")
        waveform, sample_rate = decode(extracted, sr=22050)
        assert len(waveform) / sample_rate > 10
        maps = sorted(name for name in archive.namelist() if name.endswith(".osu"))
        assert len(maps) == 2
        for name in maps:
            text = archive.read(name).decode("utf-8")
            assert "Mode: 3" in text and "CircleSize:4" in text
            assert "[HitObjects]" in text
