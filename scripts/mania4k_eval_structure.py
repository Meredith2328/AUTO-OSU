"""Structure harness: do generated charts have human-like pattern variety, sections and intensity?

Compares charts saved by ``mania4k_eval.py --save-dir`` with the ranked charts they were generated
for (same song, same star rating). Per chart, on 1-measure windows:

* pattern types (stream, trill, roll, jumpstream, handstream, jack, chordjack, ln, mixed, light)
  -> share of the most common type (monotony) and type entropy;
* results are reported per archetype of the ranked chart (切 / 叠 / LN / hybrid): human charts of
  different archetypes have very different distributions and must not be pooled;
* music sections (self-similarity novelty of measure spectra) -> dominant-type share inside a
  section (purity), and at section boundaries the change in density |dlog nps| and in type;
* intensity: Spearman correlation of measure density with loudness / high-band flux, and the
  measure density range p90/p10.

    python scripts/mania4k_eval_structure.py --gen ~/data/gen_v1 --prepared ~/data/prepared
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoosu.mania4k.chart import RedLine                                       # noqa: E402
from autoosu.mania4k.features import MEL_HOP                                     # noqa: E402
from autoosu.mania4k.onsets import HOP, SR                                       # noqa: E402
from autoosu.mania4k.structure import (FAMILY, chart_archetype, chart_profile,  # noqa: E402
                                       measure_features, novelty, section_bounds)


def structure_metrics(notes, reds, mel, env_high, rng) -> dict | None:
    p = chart_profile(notes, reds)
    n = len(p["windows"])
    if n < 16:
        return None
    types = p["types"]
    act = [t for t in types if t != "light"]
    if len(act) < 4:
        return None
    vals, cnt = np.unique(act, return_counts=True)
    shares = cnt / cnt.sum()
    fam_vals, fam_cnt = np.unique([FAMILY[t] for t in act], return_counts=True)
    feats = measure_features(mel, SR / MEL_HOP, p["windows"])
    bounds = section_bounds(novelty(feats))
    lnps = np.log1p(p["nps"])

    def change(i):
        return abs(lnps[i:i + 2].mean() - lnps[i - 2:i].mean()), float(types[i] != types[i - 1])

    at = [change(i) for i in bounds[1:] if 2 <= i <= n - 2]
    others = [i for i in range(2, n - 2) if all(abs(i - j) > 1 for j in bounds)]
    rnd = [change(i) for i in rng.choice(others, size=min(len(others), max(1, len(at))), replace=False)] if others else []
    purity = []
    for s, e in zip(bounds, bounds[1:] + [n]):
        seg = [t for t in types[s:e] if t != "light"]
        if len(seg) >= 3:
            _, c = np.unique(seg, return_counts=True)
            purity.append(c.max() / len(seg))
    fps_e = SR / HOP
    high = np.array([env_high[int(max(a, 0) / 1000 * fps_e):max(int(max(a, 0) / 1000 * fps_e) + 1,
                                                                 int(b / 1000 * fps_e))].mean()
                     for a, b in p["windows"]])
    loud = feats[:, :feats.shape[1] // 2].mean(1)
    ok = np.isfinite(high) & np.isfinite(loud)
    nz = p["nps"][p["nps"] > 0]
    return dict(
        archetype=chart_archetype(types),
        top_type=float(shares.max()), top_name=str(vals[shares.argmax()]),
        entropy=float(-(shares * np.log2(shares)).sum()),
        family_top=float(fam_cnt.max() / fam_cnt.sum()),
        purity=float(np.mean(purity)) if purity else np.nan,
        contrast=float(np.mean([a for a, _ in at]) / max(np.mean([a for a, _ in rnd]), 1e-3)) if at and rnd else np.nan,
        boundary_dnps=float(np.mean([a for a, _ in at])) if at else np.nan,
        boundary_type_change=float(np.mean([b for _, b in at])) if at else np.nan,
        corr_loud=float(spearmanr(p["nps"][ok], loud[ok]).correlation),
        corr_high=float(spearmanr(p["nps"][ok], high[ok]).correlation),
        range=float(np.percentile(nz, 90) / max(np.percentile(nz, 10), 1e-6)) if len(nz) > 4 else np.nan,
        shares={str(v): float(s) for v, s in zip(vals, shares)},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--prepared", default=os.environ.get("MANIA4K_PREPARED", str(Path.home() / "data" / "prepared")))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    cache = {}
    for f in sorted(Path(args.gen).glob("*.json")):
        g = json.loads(f.read_text())
        if g["set"] not in cache:
            cache[g["set"]] = np.load(Path(args.prepared) / f"{g['set']}.npz")
        z = cache[g["set"]]
        meta = json.loads(bytes(z["meta"]))
        c = next(c for c in meta["charts"] if c["key"] == g["key"])
        mel, high = z["mel"], z["env"][3].astype(np.float32)
        mg = structure_metrics(np.array(g["notes"]), [RedLine(*r) for r in g["reds"]], mel, high, rng)
        mr = structure_metrics(z[f"{c['key']}_notes"], [RedLine(*r) for r in c["reds"]], mel, high, rng)
        if mg and mr:
            rows.append(dict(set=g["set"], key=g["key"], stars=c["stars"], gen=mg, ref=mr))
    keys = ("top_type", "entropy", "family_top", "purity", "boundary_dnps", "boundary_type_change",
            "corr_loud", "corr_high", "range")
    groups = [("all", rows)] + [(a, [r for r in rows if r["ref"]["archetype"] == a])
                                for a in ("切", "叠", "LN", "hybrid")]
    for name, rs in groups:
        if not rs:
            continue
        same = np.mean([r["gen"]["archetype"] == r["ref"]["archetype"] for r in rs])
        print(f"\n== ranked archetype {name}: {len(rs)} charts (generated in the same archetype: {same:.0%})")
        print(f"{'metric':>22} {'generated':>10} {'ranked':>10}")
        for k in keys:
            a = np.nanmean([r["gen"][k] for r in rs])
            b = np.nanmean([r["ref"][k] for r in rs])
            print(f"{k:>22} {a:10.3f} {b:10.3f}")
        for who in ("gen", "ref"):
            agg = {}
            for r in rs:
                for t, s in r[who]["shares"].items():
                    agg[t] = agg.get(t, 0.0) + s / len(rs)
            print(f"{who:>4}", " ".join(f"{t}:{v:.2f}" for t, v in sorted(agg.items(), key=lambda kv: -kv[1])))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
