"""Section-level dataset from ranked charts: music sections, their audio energy and the human
chart's dominant pattern type and relative density in each.

    python scripts/mania4k_sections_data.py --prepared ~/data/prepared --out ~/data/sections.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoosu.mania4k.chart import RedLine                                          # noqa: E402
from autoosu.mania4k.features import MEL_HOP                                        # noqa: E402
from autoosu.mania4k.onsets import HOP, SR, OnsetEnvelopes, attack_times            # noqa: E402
from autoosu.mania4k.structure import (ARCHETYPES, TYPES, chart_archetype, chart_profile,  # noqa: E402
                                       measure_features, novelty, section_bounds)
from autoosu.mania4k.sections import section_audio_features                         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default=os.environ.get("MANIA4K_PREPARED", str(Path.home() / "data" / "prepared")))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    X, Y, D, SRS, SPLIT, SONG, ARCH, IDX, CH = [], [], [], [], [], [], [], [], []
    for f in sorted(Path(args.prepared).glob("*.npz")):
        z = np.load(f)
        meta = json.loads(bytes(z["meta"]))
        e = z["env"].astype(np.float32)
        env = OnsetEnvelopes(SR / HOP, e[0], e[1], e[2], e[3], len(e[0]) * HOP / SR)
        att = attack_times(env)
        for c in meta["charts"]:
            p = chart_profile(z[f"{c['key']}_notes"], [RedLine(*r) for r in c["reds"]])
            n = len(p["windows"])
            if n < 16:
                continue
            bounds = section_bounds(novelty(measure_features(z["mel"], SR / MEL_HOP, p["windows"])))
            feats = section_audio_features(z["mel"], env, att, p["windows"], bounds)
            mean_nps = float(np.mean(p["nps"][p["nps"] > 0]))
            arch = ARCHETYPES.index(chart_archetype(p["types"]))
            for k, (s, e_) in enumerate(zip(bounds, bounds[1:] + [n])):
                seg = [t for t in p["types"][s:e_]]
                act = [t for t in seg if t != "light"]
                label = max(set(act), key=act.count) if len(act) >= max(2, len(seg) // 2) else "light"
                X.append(feats[k])
                Y.append(TYPES.index(label))
                D.append(np.log(max(float(np.mean(p["nps"][s:e_])), 0.05) / mean_nps))
                SRS.append(c["stars"])
                SPLIT.append(meta["split"])
                SONG.append(meta["set"])
                ARCH.append(arch)
                IDX.append(k)
                CH.append(f"{meta['set']}_{c['key']}")
    np.savez(args.out, X=np.array(X, np.float32), y=np.array(Y), dens=np.array(D, np.float32),
             stars=np.array(SRS, np.float32), split=np.array(SPLIT), song=np.array(SONG),
             arch=np.array(ARCH), idx=np.array(IDX), chart=np.array(CH))
    print(len(Y), "sections")


if __name__ == "__main__":
    main()
