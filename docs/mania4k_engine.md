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
   if its tempo or phase really differs, it is not a simple metrical ratio (1/2, 2/3, 3/4 ...) of the
   global tempo, and the snap rate improves by ≥ 0.10. Songs no constant grid explains get a red line
   per beat, locked to onsets.
4. BPMs snap to the simplest value (integer first) whose drift over the segment stays within 2 ms
   (6 ms for integers); red lines sit on downbeats from the tracker's downbeat activation.
5. **Offset convention.** Ranked charts are timed before the attack in decoded audio; measured per
   chart over the corpus the offset is 24.5 ms (IQR 22.5–27.5, same for mp3 and ogg). Charts are
   written `OSU_SHIFT_MS` early so players' offsets calibrated on ranked maps apply unchanged.

## Notes (`model.py`, `features.py`)

Candidate ticks are the 1/8 ∪ 1/6 positions (12 per beat) of the red lines. A 0.8 M-parameter model
(dilated conv over log-mel, gathered at ticks, plus 17 onset-envelope taps per band at ±24 ms; then an
8-block dilated residual TCN over ticks, ±128 ticks of context; FiLM on target stars) predicts per tick:
chord size 0–4, long-note head, long-note length class. It is trained on the ranked charts' own grids,
with each chart aligned to the audio by its measured offset.

Selection (`generate.py`): ticks above a threshold θ, **only where the onset envelope peaks within
±10 ms** (any band), only on the difficulty's snaps, with non-maximum suppression at the difficulty's
minimum row gap. Chords and long notes are given to the rows the model ranks most chord-/LN-like, in the
proportion the model expects. θ is bisected until rosu-pp (osu!'s star rating algorithm) gives the
target star rating ± 0.08.

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

(filled in from the evaluation runs)
