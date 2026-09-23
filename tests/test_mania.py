import zipfile
from collections import defaultdict

import pytest

from autoosu.generate import generate
from autoosu.mania import MANIA_PRESETS, star_rating

from synth import make_song


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("song7k")
    wav = make_song(d / "Test Artist - Synth Song.wav", bpm=150.0, bars=40, offset_s=0.5)
    return generate(wav, list(MANIA_PRESETS), d / "out", seed=1, mode="mania7k", log=lambda *_: None)


def parse(text):
    section, meta, notes = None, {}, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
        elif section in ("[General]", "[Difficulty]") and ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
        elif section == "[HitObjects]" and line:
            p = line.split(",")
            col = int(int(p[0]) * 7 / 512)
            start = int(p[2])
            end = int(p[5].split(":")[0]) if int(p[3]) & 128 else start
            notes.append((col, start, end, int(p[3])))
    return meta, notes


def test_osz_has_all_7k_diffs(result):
    with zipfile.ZipFile(result.osz) as zf:
        osus = [n for n in zf.namelist() if n.endswith(".osu")]
        assert len(osus) == len(MANIA_PRESETS)
        for name in osus:
            meta, notes = parse(zf.read(name).decode("utf-8"))
            assert meta["Mode"] == "3"
            assert meta["CircleSize"] == "7"
            assert notes


def diff_texts(result):
    with zipfile.ZipFile(result.osz) as zf:
        return [zf.read(n).decode("utf-8") for n in zf.namelist() if n.endswith(".osu")]


def test_notes_playable(result):
    for text in diff_texts(result):
        meta, notes = parse(text)
        by_col = defaultdict(list)
        for col, start, end, typ in notes:
            assert 0 <= col <= 6
            assert typ & (1 | 128)
            if typ & 128:
                assert end > start
            by_col[col].append((start, end))
        for col, spans in by_col.items():
            spans.sort()
            for (s0, e0), (s1, _) in zip(spans, spans[1:]):
                assert s1 > e0, f"overlap in column {col} at {s1}"
        rows = defaultdict(int)
        for _, start, _, _ in notes:
            rows[start] += 1
        assert max(rows.values()) <= 7


def test_difficulty_ladder(result):
    counts = [d.summary()["objects"] for d in result.diffs]
    assert counts == sorted(counts) and counts[0] < counts[-1]
    stars = [star_rating(d.beatmap) for d in result.diffs]
    if None not in stars:
        assert stars == sorted(stars)
