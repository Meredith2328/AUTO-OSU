"""Fit P(chart archetype | music, stars) on the train split; report held-out accuracy against the
star-only prior; write autoosu/mania4k/weights/style.json.

    python scripts/mania4k_fit_style.py --prepared ~/data/prepared
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoosu.mania4k.chart import RedLine, main_bpm                          # noqa: E402
from autoosu.mania4k.onsets import HOP, SR                                    # noqa: E402
from autoosu.mania4k.planner import load_plan                                 # noqa: E402
from autoosu.mania4k.structure import ARCHETYPES, chart_archetype, chart_profile  # noqa: E402
from autoosu.mania4k.style import DESCRIPTORS, STYLE_FILE, song_descriptors    # noqa: E402


def dataset(prepared: Path):
    rows = []
    for f in sorted(prepared.glob("*.npz")):
        z = np.load(f)
        meta = json.loads(bytes(z["meta"]))
        env = z["env"].astype(np.float32)
        for c in meta["charts"]:
            reds = [RedLine(*r) for r in c["reds"]]
            notes = z[f"{c['key']}_notes"]
            if len(notes) < 50:
                continue
            x = song_descriptors(z["mel"], *env, SR / HOP, main_bpm(reds, float(notes[-1, 0])), c["stars"])
            y = ARCHETYPES.index(chart_archetype(chart_profile(notes, reds)["types"]))
            rows.append((meta["split"], meta["set"], x, y, c["stars"]))
    return rows


def prior_pred(stars: float) -> int:
    plan = load_plan()
    b = min(int(np.searchsorted(plan["star_bands"], stars, side="right") - 1), len(plan["arch_prior"]) - 1)
    return int(np.argmax(plan["arch_prior"][b]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default=os.environ.get("MANIA4K_PREPARED", str(Path.home() / "data" / "prepared")))
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression

    rows = dataset(Path(args.prepared))
    tr = [r for r in rows if r[0] == "train"]
    ho = [r for r in rows if r[0] != "train"]
    X, y = np.array([r[2] for r in tr]), np.array([r[3] for r in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    print("train charts", len(tr), "held-out", len(ho), "label shares train",
          {ARCHETYPES[k]: round(float((y == k).mean()), 2) for k in range(len(ARCHETYPES))})
    # charts of one set share audio: weight sets equally so big sets do not dominate
    sets = np.array([r[1] for r in tr])
    wt = np.array([1.0 / (sets == s).sum() for s in sets])
    clf = LogisticRegression(C=args.C, max_iter=2000)
    clf.fit((X - mu) / sd, y, sample_weight=wt)
    Xh, yh = np.array([r[2] for r in ho]), np.array([r[3] for r in ho])
    ph = clf.predict((Xh - mu) / sd)
    pp = np.array([prior_pred(r[4]) for r in ho])
    acc = lambda p: float((p == yh).mean())                                  # noqa: E731
    bal = lambda p: float(np.mean([(p[yh == k] == k).mean() for k in np.unique(yh)]))  # noqa: E731
    print(f"held-out accuracy: model {acc(ph):.2f} (balanced {bal(ph):.2f})   star prior {acc(pp):.2f} "
          f"(balanced {bal(pp):.2f})")
    for k, a in enumerate(ARCHETYPES):
        m = yh == k
        if m.any():
            print(f"  {a:4s} n={m.sum():3d} model recall {(ph[m] == k).mean():.2f}  prior recall {(pp[m] == k).mean():.2f}"
                  f"  predicted {(ph == k).sum()}")
    # grouped 5-fold CV over all songs (more charts than the held-out split; folds never share a set)
    from sklearn.model_selection import GroupKFold

    Xa, ya = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
    ga, sa = np.array([r[1] for r in rows]), np.array([r[4] for r in rows])
    pc = np.zeros(len(ya), int)
    for a, b in GroupKFold(5).split(Xa, ya, ga):
        m_, s_ = Xa[a].mean(0), Xa[a].std(0) + 1e-9
        pc[b] = LogisticRegression(C=args.C, max_iter=2000).fit((Xa[a] - m_) / s_, ya[a]).predict((Xa[b] - m_) / s_)
    pa = np.array([prior_pred(s) for s in sa])
    print(f"grouped CV ({len(ya)} charts): model {(pc == ya).mean():.2f}, star prior {(pa == ya).mean():.2f}; recall "
          + " ".join(f"{ARCHETYPES[k]} {(pc[ya == k] == k).mean():.2f}/{(pa[ya == k] == k).mean():.2f}" for k in range(4)))
    coef = clf.coef_ if len(clf.classes_) == len(ARCHETYPES) else None
    print("coefficients (z-scored descriptors):")
    for j, d in enumerate(DESCRIPTORS):
        print(f"  {d:12s} " + " ".join(f"{ARCHETYPES[k]} {clf.coef_[i, j]:+.2f}" for i, k in enumerate(clf.classes_)))
    if args.write and coef is not None:
        STYLE_FILE.write_text(json.dumps(dict(descriptors=list(DESCRIPTORS), archetypes=list(ARCHETYPES),
                                              mean=mu.tolist(), std=sd.tolist(), coef=coef.tolist(),
                                              intercept=clf.intercept_.tolist(),
                                              heldout=dict(accuracy=acc(ph), balanced=bal(ph),
                                                           prior_accuracy=acc(pp), prior_balanced=bal(pp)),
                                              grouped_cv=dict(accuracy=float((pc == ya).mean()),
                                                              prior_accuracy=float((pa == ya).mean()))),
                                         indent=1), encoding="utf-8")
        print("wrote", STYLE_FILE)


if __name__ == "__main__":
    main()
