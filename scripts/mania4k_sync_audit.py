"""Independent sync audit: do note heads sit on attacks seen by a *different* onset detector?

verify_chart checks notes against the generator's own onset envelope, which is circular. This script
uses librosa's onset strength (its own spectral flux and frame grid) instead: for each chart, the mean
strength at the note heads shifted by -60..+60 ms. Notes on attacks give a sharp peak at the detector
latency; notes on a drifting grid give a flat or displaced curve. Human and generated charts are
measured the same way, so they can be compared on equal terms.

    python scripts/mania4k_sync_audit.py --human --split test --out sync_human.json
    python scripts/mania4k_sync_audit.py --gen-dir ~/data/gen_v2e --out sync_gen.json
    python scripts/mania4k_sync_audit.py --gen-dir ~/data/gen_unseen --out sync_unseen.json

The detector's latency is a constant; it is estimated once from the human charts (peak of their
mean curve) and passed with --latency to the other runs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

SR, HOP = 22050, 128


def onset_curve(path: str) -> np.ndarray:
    """librosa's onset strength (its own mel flux, lag 1, max filter 3), z-scored per song."""
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, lag=1, max_size=3)
    return (env - env.mean()) / (env.std() + 1e-9)


SHIFTS = np.arange(-60.0, 60.5, 1.0)


def alignment(heads: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Mean onset strength at heads + shift (ms) for every shift in SHIFTS."""
    t = np.arange(len(env)) * HOP / SR * 1000.0
    return np.array([np.interp(heads + d, t, env).mean() for d in SHIFTS])


def chart_row(label: str, heads: np.ndarray, curve: np.ndarray, latency: float) -> dict:
    """peak: where the alignment curve peaks relative to the detector latency (0 = on the attacks);
    lift: strength at the latency minus strength 25-60 ms away (how much the notes sit on attacks)."""
    at = float(np.interp(latency, SHIFTS, curve))
    away = np.abs(SHIFTS - latency) >= 25
    return dict(chart=label, heads=int(len(heads)), peak=float(SHIFTS[np.argmax(curve)] - latency),
                lift=at - float(curve[away].mean()), at=at)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--gen-dir", default="")
    ap.add_argument("--prepared", default=os.environ.get("MANIA4K_PREPARED", str(Path.home() / "data" / "prepared")))
    ap.add_argument("--corpus", default=os.environ.get("MANIA4K_CORPUS", str(Path.home() / "data" / "mania4k")))
    ap.add_argument("--split", default="test")
    ap.add_argument("--latency", type=float, default=None, help="detector latency (ms); default: from these charts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    jobs = []                                            # (label, audio, heads)
    if args.human or args.gen_dir:
        audio_of = {}
        for f in sorted(Path(args.prepared).glob("*.npz")):
            z = np.load(f)
            meta = json.loads(bytes(z["meta"]))
            audio_of[meta["set"]] = str(Path(args.corpus) / meta["set"] / meta["audio"])
            if args.human and meta["split"] == args.split:
                for c in meta["charts"]:
                    jobs.append((f"{meta['set']}_{c['key']}", audio_of[meta["set"]], z[f"{c['key']}_notes"][:, 0]))
        if args.gen_dir:
            for f in sorted(Path(args.gen_dir).glob("*.json")):
                d = json.loads(f.read_text())
                audio = d.get("audio") or audio_of.get(d.get("set"))
                if audio and d["notes"]:
                    jobs.append((f.stem, audio, np.array([n[0] for n in d["notes"]], float)))
    cache: dict = {}
    raw = []
    for label, audio, heads in jobs:
        if audio not in cache:
            cache = {audio: onset_curve(audio)}                  # jobs are grouped by song
        heads = np.unique(np.round(heads, 1))
        raw.append((label, heads, alignment(heads, cache[audio])))
    lat = args.latency
    if lat is None:                                     # head-weighted mean curve of these charts
        tot = sum(len(h) * c for _, h, c in raw) / sum(len(h) for _, h, _ in raw)
        lat = float(SHIFTS[np.argmax(tot)])
    rows = [chart_row(lbl, h, c, lat) for lbl, h, c in raw]
    agg = dict(lift=float(np.median([r["lift"] for r in rows])),
               peak_abs=float(np.median([abs(r["peak"]) for r in rows])),
               peak_within5=float(np.mean([abs(r["peak"]) <= 5 for r in rows])))
    print(f"charts {len(rows)}  latency {lat:+.0f} ms  median lift {agg['lift']:.2f} z  "
          f"median |peak| {agg['peak_abs']:.1f} ms  peak within 5 ms {agg['peak_within5']:.0%}")
    Path(args.out).write_text(json.dumps(dict(latency=lat, summary=agg, rows=rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
