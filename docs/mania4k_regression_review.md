# Synthetic regression fixes

Base: `0707c310e68158b9d5476b3ee2120ed1f3c41b2e`. These fixes address independently
reproduced source behavior. They do not establish musical quality or real-audio accuracy.
The author-reported chart/timing results in mania4k_engine.md predate these changes.

| Controlled input | Original behavior | Intended behavior |
| --- | --- | --- |
| 179.9 BPM, 96 beats, exact synthetic attacks | integer rounding yields 180 BPM and 12.97ms worst error | retain 179.9 and the measured grid |
| already identified 9ms phase change affecting 100/140 attacks | fold into global phase; 7ms coverage 28.57% | preserve both identified segments |
| supported straight p=.9, neighboring triplets p=.6 each; lower theta .8 to .5 | delete both previously selected straight rows | retain supported rows across thresholds |
| explicitly nonmonotonic synthetic search response with reachable target | bisection can miss the other half | bounded fallback probes cover both halves |
| initial boost overshoots; a later boost can hit | stop before testing the later boost | stop on a tolerance hit, otherwise keep the best and continue |

All attacks and probabilities above are synthetic. Search responses are controlled units,
not actual star ratings. Tests load the real pure function bodies through AST selection so
they do not import torch, load weights, decode audio or invoke the tracker. The original
changed function bodies are retained as a pinned before/after fixture, with source blob IDs.

Timing uses the stated 2ms half-segment budget for every BPM rounding quantum. Phase folding
only combines identical grids. Removing threshold-dependent hard rhythm-family rejection
preserves previously supported rows; it does not claim that every mixed rhythm is desirable.

Calibration retains all 12 original bisection probes per boost. If the target remains missed,
it evaluates 4 additional thresholds covering the interval endpoints and both halves. The
three boost levels therefore score at most 48 charts rather than the previous maximum 36.
Already successful refinement stops immediately as before. This bounded extra work avoids
the demonstrated missed targets, but cannot guarantee a hit for arbitrary unsampled optima.
The achieved star rating and its error remain the result when the target is unreachable.

The 24ms export convention is unchanged. Native grid agreement still removes median offset;
that residual statistic must not be called absolute alignment. Same-pair raw onset/lane
matching, checkpoint/run provenance, cross-set audio deduplication and genuine playtests
remain separate acceptance evidence. No generated music files or personal data are included.

Focused reproduction: `python -m pytest -q tests/test_mania4k_regressions.py`.
