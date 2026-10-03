"""Fit the section planner's tables from ranked charts (train split).

* P(section type | archetype, energy level, previous section type)  (Markov, smoothed, backed off)
* relative log density per (archetype, energy level)
* P(archetype | star band)
* per-type execution targets: notes per row, long-note rows, jack rate

    python scripts/mania4k_fit_plan.py --sections ~/data/sections.npz --prepared ~/data/prepared \\
        --out autoosu/mania4k/weights/plan.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoosu.mania4k.chart import RedLine                                  # noqa: E402
from autoosu.mania4k.sections import energy_levels                         # noqa: E402
from autoosu.mania4k.structure import ARCHETYPES, TYPES, chart_profile      # noqa: E402

STAR_BANDS = (0.0, 2.0, 3.0, 4.0, 5.0, 99.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--prepared", default=os.environ.get("MANIA4K_PREPARED", str(Path.home() / "data" / "prepared")))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = np.load(args.sections)
    train = d["split"] == "train"
    X, y, dens, arch, chart, stars = d["X"], d["y"], d["dens"], d["arch"], d["chart"], d["stars"]
    level = np.zeros(len(y), int)
    for c in np.unique(chart):
        m = chart == c
        level[m] = energy_levels(np.nan_to_num(X[m, 0]))
    nT, nA = len(TYPES), len(ARCHETYPES)
    first = np.ones((nA, 4, nT))                # P(type | arch, level) for the first section / backoff
    trans = np.ones((nA, 4, nT, nT)) * 0.5      # P(type | arch, level, prev)
    dsum = np.zeros((nA, 4))
    dcnt = np.zeros((nA, 4))
    order = np.lexsort((d["idx"], chart))
    prev_chart, prev_t = None, None
    for i in order:
        if not train[i]:
            continue
        a, L, t = arch[i], level[i], y[i]
        first[a, L, t] += 1
        if chart[i] == prev_chart:
            trans[a, L, prev_t, t] += 1
        prev_chart, prev_t = chart[i], t
        if np.isfinite(dens[i]):
            dsum[a, L] += dens[i]
            dcnt[a, L] += 1
    first /= first.sum(-1, keepdims=True)
    trans = 0.7 * trans / trans.sum(-1, keepdims=True) + 0.3 * first[:, :, None, :]
    # archetype prior per star band (one vote per chart)
    prior = np.ones((len(STAR_BANDS) - 1, nA))
    for c in np.unique(chart[train]):
        i = np.where(chart == c)[0][0]
        b = int(np.searchsorted(STAR_BANDS, stars[i], side="right") - 1)
        prior[min(b, len(prior) - 1), arch[i]] += 1
    prior /= prior.sum(-1, keepdims=True)
    # execution targets per window type
    acc = collections.defaultdict(list)
    for f in sorted(Path(args.prepared).glob("*.npz"))[::2]:
        z = np.load(f)
        meta = json.loads(bytes(z["meta"]))
        if meta["split"] != "train":
            continue
        for c in meta["charts"]:
            p = chart_profile(z[f"{c['key']}_notes"], [RedLine(*r) for r in c["reds"]])
            for t, s in zip(p["types"], p["stats"]):
                acc[t].append((s.chord, s.ln, s.jack, s.triple))
    targets = {t: dict(zip(("notes_per_row", "ln_rows", "jack", "triple"),
                           [float(v) for v in np.mean(acc[t], axis=0)])) for t in TYPES if acc[t]}
    plan = dict(types=list(TYPES), archetypes=list(ARCHETYPES), star_bands=list(STAR_BANDS),
                first=first.round(4).tolist(), trans=trans.round(4).tolist(),
                density=(dsum / np.maximum(dcnt, 1)).round(3).tolist(), arch_prior=prior.round(3).tolist(),
                targets=targets)
    Path(args.out).write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    print("archetype prior per star band:", np.round(prior, 2).tolist())
    print("density per archetype/level:", np.round(dsum / np.maximum(dcnt, 1), 2).tolist())


if __name__ == "__main__":
    main()
