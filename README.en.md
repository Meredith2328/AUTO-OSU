# AUTO-OSU · osu!mania 4K

[中文](README.md) · [Design & evaluation](docs/mania4k_engine.md) · [Original README (osu!standard)](docs/original/README.en.md)

**Drop in a song, get an osu!mania 4K beatmap that is on the beat and calibrated to a star rating.**

This repository is a fork of [kanze1/AUTO-OSU](https://github.com/kanze1/AUTO-OSU), which generates **osu!standard** maps (rhythm transformer + coordinate diffusion, desktop GUI, Windows exe). All of that still works; see the [original README](docs/original/README.en.md). This fork focuses on **osu!mania 4K**.

## Use

```bash
pip install -e .                       # Python 3.10+, CPU is fine
python -m autoosu song.mp3 --mode mania4k -d Easy Normal Hard Insane Expert -o out
python -m autoosu song.mp3 --mode mania4k -d Hard --mania-stars 3.5     # a specific star rating
python -m autoosu song.mp3 --mode mania4k --mania-style jack             # chart style: auto/speed/jack/ln/hybrid
python -m autoosu                       # GUI; pick "osu!mania 4K" as the game mode
```

The result is an `.osz` that osu!/lazer imports on double-click. The first run downloads the 80 MB beat-tracker weights (or put `beat_this-final0.ckpt` into `models/`). A 3-minute song with five difficulties takes 1–2 minutes on one CPU thread.

## What it does

| Stage | How |
| --- | --- |
| Timing | Beat This! neural beat tracking + a BPM/phase search that puts the song's attacks on the 1/4 grid; tempo changes, multiple red lines, integer BPMs, red lines on downbeats; written with the 24 ms convention measured on ranked maps |
| Sync (hard constraint) | every note on the grid with a distinct attack within ±8 ms; no straight/triplet mix inside a beat; `verify_chart` re-checks every chart and nothing failing is written |
| Notes | a beat-grid TCN conditioned on star rating, learned from 300 ranked 4K sets: where notes go, chord sizes, long notes |
| Patterns | a lane-pattern model learned from 1.5 M ranked rows, decoded under hard constraints (held lanes, minimum jack interval) |
| Difficulty | bisected with osu!'s star-rating algorithm (rosu-pp); OD/HP and chord share from ranked charts of the same star range; difficulties a song cannot support are skipped, not padded |

## Quality (v1, 22 unseen songs / 67 ranked charts)

| | old rules engine | v1 | ranked |
| --- | --- | --- | --- |
| notes on the human chart's grid | 95.8 % | **98.4 %** | |
| passing sync/playability checks | 61 % | **100 %** | |
| star-rating error | 1.44★ | **0.21★** | |
| notes/s / chord rows | 3.5 / 9.9 % | **8.6 / 35.1 %** | 9.2 / 35.6 % |
| timing matches human red lines (218 single-BPM songs) | 77.9 % | **93.5 %** | |

## Iterations

- **v1**: exact timing, sync as a hard constraint, star calibration. Human review 70/100: sync and difficulty right, but patterns monotonous (almost all speed), weak sections, no emotional arc.
- **v2**: pick a chart style first (speed / jack / LN / hybrid, `--mania-style`, auto by star rating and music), cut the song into music sections rated by energy, and plan one main pattern type (optionally a second) and a density per section from the distributions of ranked charts of the same style; the pattern model is conditioned on the section type. Against human charts of the same style: speed charts jumpstream/stream/roll 38/24/14 % (human 40/23/13 %), jack charts chordjack 32 % (35 %), LN charts LN 80 % (77 %); sync, star and playability constraints unchanged. See [docs/mania4k_engine.md](docs/mania4k_engine.md#v2-archetypes-sections-and-intensity).

Known limits: live recordings with drifting tempo, swing/jazz and tournament tracks with many tempo changes; no SV or keysounds.

## Reproduce

Corpus, training and evaluation scripts are `scripts/mania4k_*.py`; method and data in [docs/mania4k_engine.md](docs/mania4k_engine.md).

License: MIT with an attribution condition, see [LICENSE](LICENSE).
