# osu!mania 4K: the ranked-calibrated engine

`--mode mania4k` (and the GUI's *osu!mania 4K (ranked-calibrated)*) uses `autoosu/mania4k/`. It replaces
the earlier rule-based generator as the default; that one is still available with `--mania-engine rules`.

Design goal, in priority order:

1. **Every note is on the beat.** Exact red lines, every head and tail on the grid, every head on an
   audible attack. Checked for every generated chart before it is written.
2. **Charts read like ranked charts.** Density, chords, long notes and lane patterns are learned from
   ranked 4K charts, and each difficulty is calibrated to a star rating computed exactly like osu! does.
3. **Playable by construction.** No overlaps, no impossible jacks, difficulty-appropriate snaps and chords.

## Reference corpus

`scripts/mania4k_fetch.py` downloads ranked 4K sets (audio + 4K `.osu` only) from an osu! mirror,
newest ranked first: 300 sets, 1,000+ charts from 0.8 to 9 stars. Sets are split by a hash of the set id
into train / validation / test (82 / 6 / 12 %). Nothing from the corpus is shipped.

What the corpus says (medians per star band) and what the engine adopts:

| stars | notes/s | chord rows | LN share | same-lane repeat, 1st pct | closest rows, 1st pct | OD / HP |
| --- | --- | --- | --- | --- | --- | --- |
| < 1.8 | 2.9 | 18 % | 21 % | 300 ms | 125 ms | 6.5 / 6.5 |
| 1.8–2.6 | 5.1 | 33 % | 23 % | 166 ms | 75 ms | 7.0 / 7.0 |
| 2.6–3.5 | 7.6–10 | 40 % | 22 % | 131 ms | 58 ms | 7.5 / 7.7 |
| 3.5–4.5 | 12.3 | 39 % | 20 % | 100 ms | 32 ms | 8.0 / 8.0 |
| 4.5–6 | 14–16 | 36 % | 21 % | 83 ms | 25 ms | 8.0 / 8.0 |

## Timing (`timing.py`)

1. **Beat This!** (Foscarin, Schlüter, Widmer, ISMIR 2024) gives beat/downbeat activations at 50 fps.
   Its dominant inter-beat interval sets the tempo basin; the octave follows the ranked convention
   (main BPM in 100–250: of the corpus' main BPMs, 92 % lie there).
2. Period and phase are searched coarse-to-fine (to 0.0002 ms/beat) to maximise how many onset peaks of
   a 2.9 ms log-flux envelope land on the 1/4 grid of the *whole song*. Local tracker slips (double
   time, dotted pulses) cannot move this global optimum.
3. Tempo changes: 12-beat windows whose onsets fit a *local* grid much better than the global one, and
   stretches where the tracker's local tempo departs by > 2 %, become candidate regions. Their edges are
   moved beat by beat to where onsets switch grids, then refitted; a region keeps its own red line only
   if its tempo differs (snap rate + 0.10) or, much more strictly, only its phase (+ 0.20, ≥ 16 beats),
   and it is not a simple metrical ratio (1/2, 2/3, 3/4 ...) of the global tempo. Pieces that snap back
   onto the global grid are merged into it.
4. BPMs snap to the simplest value (integer first) whose drift over the segment stays within 2 ms
   (6 ms for integers); red lines sit on downbeats from the tracker's downbeat activation.
5. **Offset convention.** Ranked charts are timed before the attack in decoded audio; measured per
   chart over the corpus the offset is 24.5 ms (IQR 22.5–27.5, same for mp3 and ogg), and 24.4 ms from
   the timing audit. Charts are written `OSU_SHIFT_MS` = 24 ms early so players' offsets calibrated on
   ranked maps apply unchanged.
6. **Octave.** When strong attacks fall on the off-beat as often as on the beat, mappers write the
   doubled BPM (if ≤ 250), and so does the engine.

## Notes (`model.py`, `features.py`)

Candidate ticks are the 1/8 ∪ 1/6 positions (12 per beat) of the red lines. A 0.8 M-parameter model
(dilated conv over log-mel, gathered at ticks, plus 17 onset-envelope taps per band at ±24 ms; then an
8-block dilated residual TCN over ticks, ±128 ticks of context; FiLM on target stars) predicts per tick:
chord size 0–4, long-note head, long-note length class. It is trained on the ranked charts' own grids,
with each chart aligned to the audio by its measured offset.

Selection (`generate.py`): ticks above a threshold θ, **only where a distinct attack (a prominent
onset peak in any band) lies within ±8 ms** (stricter than human mappers: ~14 % of ranked notes have
no such peak), only on the difficulty's snaps (1/8 only when ≥ 55 ms apart), with non-maximum
suppression at the difficulty's minimum row gap. One rhythm family per beat (straight or triplet), and
triplets only inside triplet passages. Chords go to the rows the model ranks most chord-like, in the
proportion the model expects but at most the ranked 75th percentile for the star range; long notes to
the most sustained rows (up to 12 %). θ is bisected until rosu-pp (osu!'s star rating algorithm) gives
the target star rating ± 0.08; if the song is too sparse, the chord share is raised in steps.

## Lane patterns (`patterns.py`)

An MLP predicts the lane combination of each row from: chord size, gap to the previous row (ms and
beats), lanes held by long notes, time since each lane was pressed, long-note flag, the last 8 rows
(lanes + gaps), target stars. Trained on 1.5 M ranked rows (mirrored). Decoding samples at temperature
0.75 under hard constraints: held lanes (plus a release gap) are unavailable, same-lane repeats must
respect the difficulty minimum; if no combination fits, the chord shrinks.

## Verification (`verify.py`)

`verify_chart` re-checks every generated chart: heads and tails on the grid (±1 ms), heads on an
attack, no overlaps, no too-fast jacks, chord size, row spacing, nothing before the first red line.
`generate()` refuses to write a chart that fails.

`scripts/mania4k_eval_timing.py` compares the timing engine with ranked red lines;
`scripts/mania4k_eval.py` compares generated charts with the ranked charts of held-out songs (and runs
the earlier generator with `--baseline`). Results: see below.

## Results

All numbers are on songs the models never saw (test split, 22 songs / 67 ranked charts), or for timing
on every corpus song whose tracker output was cached (218 songs, 154 with one red line in the ranked
map, 64 with several).

### Timing against human red lines

A ranked note counts as *consistent* when it lies within 5 ms of our grid at the musically equivalent
snap (octave-aware), after removing the song's constant offset; a song passes at ≥ 98 %.

| | constant-tempo songs (154) | songs whose ranked map has several red lines (64) | all (218) |
| --- | --- | --- | --- |
| previous estimator (`autoosu.timing`, one red line) | 77.9 % (mean consistency 89.5 %) | 34.4 % (mean 60.9 %) | 65.1 % (mean 81.1 %) |
| this engine without tempo-change detection | 94.2 % (mean 97.4 %) | 46.9 % (mean 78.1 %) | 80.3 % (mean 91.7 %) |
| **this engine** | **93.5 % (mean 97.3 %)** | **46.9 % (mean 81.6 %)** | **79.8 % (mean 92.7 %)** |

Main BPM identical to the mapper's: 73.9 % before, 80.7 % now (the rest are mostly octave choices,
which do not affect sync).

Median offset of our grid against ranked notes: +3.6 ms with a 28 ms convention, i.e. ranked charts
are 24.4 ms ahead of the decoded attack, matching the per-chart estimate (24.5 ms); `OSU_SHIFT_MS = 24`.

Remaining failures: live recordings whose tempo drifts continuously (ranked maps use dozens to
hundreds of red lines), tournament tracks with many tempo changes, swung hip-hop/jazz, ternary
(12/8) songs mapped at a third of our BPM. In the 4-beat-phase cases the 1/4 grid points are right
but the beat labelling is a quarter off.

### Generated charts against the ranked charts of the same songs

One chart generated per ranked chart, at that chart's star rating.

| | previous rules engine | **this engine** | ranked charts |
| --- | --- | --- | --- |
| generated notes on the ranked chart's own grid (±5 ms) | 95.8 % (81 % of charts 100 %) | **98.4 % (82 % of charts 100 %)** | |
| charts passing `verify_chart` | 61 % | **100 %** | |
| \|star rating − target\| | 1.44 ★ (1.5 % within ±0.2) | **0.21 ★ (76 % within ±0.2)** | |
| recall of ranked note heads (±12 ms) | 39.5 % | **73.7 %** | |
| precision against all ranked heads of the song | 84.6 % | 84.4 % (at 2.5× the notes) | |
| notes / s | 3.5 | **8.6** | 9.2 |
| chord rows | 9.9 % | **35.1 %** | 35.6 % |
| long notes | 6.0 % | 9.7 % | 21.3 % |
| consecutive rows sharing a lane (jacks) | 10.0 % | 11.6 % | 17.8 % |
| same-hand run (p95) / longest anchor | 1.0 / 1.0 (strict alternation) | **2.0 / 3.2** | 2.0 / 3.3 |
| left-hand share | 0.50 | 0.49 | 0.50 |

When a song cannot reach a difficulty's star target without overmapping (the ranked charts of the
song stop lower too), the engine falls short instead of padding notes, and skips a difficulty that would
land within 0.4 ★ of the previous one.

Note model on held-out charts (tick level, given the chart's star rating): precision 0.867,
recall 0.826, F1 0.846. Pattern model: 56 % top-1 lane-combination accuracy (chance 25 % for single notes).

### Reproduce

```bash
python scripts/mania4k_fetch.py --out ~/data/mania4k --sets 300
python scripts/mania4k_prepare.py --corpus ~/data/mania4k --out ~/data/prepared
python scripts/mania4k_train_notes.py --data ~/data/prepared --out notes.pt --epochs 20      # best epoch kept
python scripts/mania4k_train_patterns.py --data ~/data/prepared --out patterns.pt
python scripts/mania4k_eval_timing.py --corpus ~/data/mania4k [--baseline]
python scripts/mania4k_eval.py --prepared ~/data/prepared --corpus ~/data/mania4k [--baseline]
python scripts/mania4k_preview.py chart.osu -o chart.png --start 30 --seconds 20
```

## v2: archetypes, sections and intensity

Human review of v1 (70/100): sync and difficulty right, but patterns monotonous (almost all 切),
no section feel, no emotional arc. v2 is built on what human charts actually do, measured on the
corpus with `autoosu/mania4k/structure.py`:

* **Pattern types per measure**: stream, trill, roll, jumpstream, handstream (切); jack,
  chordjack (叠); LN; mixed (乱); light.
* **Charts first commit to an archetype**, and the distributions only make sense within one:

  | archetype | charts | median ★ | main types |
  | --- | --- | --- | --- |
  | 切 stream | 337 | 2.6 | jumpstream 39 %, stream 24 %, roll 11 % |
  | LN | 276 | 3.4 | LN 73 %, jumpstream 10 % |
  | hybrid | 166 | 3.4 | jumpstream 27 %, LN 27 %, mixed 16 %, chordjack 9 % |
  | 叠 jack | 60 | 4.2 | chordjack 46 %, mixed 15 %, jumpstream 14 % |

* **Sections**: on music sections found by self-similarity novelty of measure spectra, human charts
  change density 2.7× more at boundaries than elsewhere (|Δlog nps| 0.38 vs 0.14) and change type
  81 % vs 64 % of the time; inside a section one type covers 69 % of measures (81 % in LN charts).
* **Intensity**: measure density follows loudness / high-band flux (Spearman ≈ +0.5), much more
  than raw attack count (+0.2); by section energy level (rest / low / mid / climax) density goes
  from −45…−60 % to +5 % of the chart mean, and the harder subtype takes over at climaxes
  (切: stream → jumpstream; 叠: chordjack 29 % → 59 %).

The generator follows the same order (`planner.py`): choose an archetype (`--mania-style`, or auto
from the star rating and how sustained the music is; one per set; no 叠 below 2★), cut the song
into music sections, rate their energy, sample one type per section from the human
P(type | archetype, energy level, previous type) with a secondary type on 4-measure phrases (rate
per archetype), and scale density by the human density curve. Execution: per-type targets for
notes per row, triples and LN rows; the pattern model is retrained with section type and
archetype as inputs (held-out NLL 0.976 vs 0.997), plus small logit biases that make rolls,
trills, jacks and anti-jack streams recognisable; LN sections hold through the flow.

### Structure harness (`scripts/mania4k_eval_structure.py`)

Generated charts are compared with the ranked chart they were made for (same song and star rating,
generated in that chart's archetype with `mania4k_eval.py --match-archetype`), and results are
reported **per archetype**. Held-out test songs, 65 charts:

| | v1 | v2 | human |
| --- | --- | --- | --- |
| generated in the ranked chart's archetype | 44 % | **86 %** | |
| 切 charts: jumpstream / stream / roll | 47 / 22 / 10 % | **38 / 24 / 14 %** | 40 / 23 / 13 % |
| 叠 charts: chordjack / jumpstream | 4 / 55 % | **32 / 22 %** | 35 / 18 % |
| LN charts: LN | 8 % | **80 %** | 77 % |
| in-section dominant type | 0.59 | 0.75 | 0.69 |
| type change at music boundaries | 0.62 | 0.60 | 0.68 |
| density change at music boundaries | 0.32 | 0.50 | 0.37 |
| density ↔ loudness (Spearman) | 0.42 | 0.38 | 0.40 |

The hard constraints are unchanged: 100 % of charts pass `verify_chart`, 98.4 % of notes lie on
the ranked chart's own grid, star error 0.15★ on average.

Open points: hybrid charts drift towards 切 (their LN sections are under-realised); intensity
coupling in LN and hybrid charts is weaker than human (0.29–0.42 vs 0.45–0.51); density jumps at
boundaries are stronger than human.
