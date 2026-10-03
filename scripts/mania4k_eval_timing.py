"""Audit timing engines against the red lines of ranked 4K charts of the same audio.

    python scripts/mania4k_eval_timing.py --corpus ~/data/mania4k --cache ~/data/cache [--baseline]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoosu.audio import load_audio                                 # noqa: E402
from autoosu.mania4k.chart import RedLine, load_osu                  # noqa: E402
from autoosu.mania4k.onsets import onset_envelopes                   # noqa: E402
from autoosu.mania4k.timing import estimate_timing, pick_beats, tracker_activations, tracker_deviation  # noqa: E402
from autoosu.mania4k.verify import timing_report                     # noqa: E402


def reference(d: Path):
    charts = [c for c in (load_osu(f) for f in sorted(d.glob("*.osu"))) if c]
    return max(charts, key=lambda c: len(c.notes)) if charts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.environ.get("MANIA4K_CORPUS", str(Path.home() / "data" / "mania4k")))
    ap.add_argument("--cache", default=os.environ.get("MANIA4K_CACHE", str(Path.home() / "data" / "cache")))
    ap.add_argument("--shift", type=float, default=28.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline", action="store_true", help="evaluate autoosu.timing.estimate_timing")
    ap.add_argument("--out", default="")
    ap.add_argument("--cached-only", action="store_true", help="skip songs without cached tracker output")
    ap.add_argument("--no-changes", action="store_true", help="force one constant tempo per song")
    ap.add_argument("--shard", default="0/1", help="i/n: evaluate every n-th song starting at i")
    args = ap.parse_args()
    rows = []
    dirs = [d for d in sorted(Path(args.corpus).iterdir()) if (d / "meta.json").exists()]
    if args.limit:
        dirs = dirs[:args.limit]
    si, sn = (int(x) for x in args.shard.split("/"))
    dirs = dirs[si::sn]
    for d in dirs:
        ref = reference(d)
        if ref is None:
            continue
        if args.cached_only and not (Path(args.cache) / f"{d.name}.beat.npz").exists():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        t0 = time.time()
        y, sr = load_audio(d / meta["audio"], sr=22050)
        if args.baseline:
            from autoosu.audio import analyze
            from autoosu.timing import estimate_timing as baseline_timing

            tm = baseline_timing(analyze(y, sr))
            reds = [RedLine(tm.offset_ms, tm.beat_length)]
            kind = "base"
        else:
            cache = Path(args.cache) / f"{d.name}.beat.npz"
            if cache.exists():
                z = np.load(cache)
                beat, down = z["beat"].astype(np.float32), z["down"].astype(np.float32)
            else:
                beat, down = tracker_activations(y, sr)
                np.savez_compressed(cache, beat=beat.astype(np.float16), down=down.astype(np.float16))
            env = onset_envelopes(y, sr)
            try:
                res = estimate_timing(env, beat, down, first_note_ms=ref.notes[0].time + args.shift,
                                      allow_changes=not args.no_changes)
                reds, kind = res.red_lines, f"{res.kind[:5]} {res.snap_rate:.2f}"
                beats = pick_beats(beat)
                devs = dict(dev_ours=res.tracker_dev_ms, dev_ref=tracker_deviation(
                    beats, [RedLine(r.time + args.shift, r.beat_ms, r.meter) for r in ref.red_lines]))
            except ValueError as exc:
                print(d.name, "FAILED", exc, flush=True)
                continue
        rep = timing_report(ref, reds, args.shift)
        rows.append(dict(set=d.name, ref_reds=len(ref.red_lines), our_reds=len(reds), kind=kind, **rep.__dict__,
                         **(devs if not args.baseline else {})))
        print(f"{d.name:>8} ref {rep.bpm_ref:7.2f}x{len(ref.red_lines):<3} ours {rep.bpm_ours:7.2f}x{len(reds):<3}"
              f" consistent {rep.consistent_5ms:6.1%} <=5ms {rep.within_5ms:6.1%} <=10ms {rep.within_10ms:6.1%} med {rep.median_signed:+5.1f}"
              f" p95 {rep.p95_abs:5.1f}  {time.time() - t0:4.1f}s  {meta['artist'][:20]} - {meta['title'][:30]}",
              flush=True)
    w5 = np.array([r["consistent_5ms"] for r in rows])
    single = np.array([r["ref_reds"] == 1 for r in rows])
    print(f"\nsongs {len(rows)}  pass(>=98% consistent within 5ms) {np.mean(w5 >= 0.98):.1%}  "
          f"single-BPM pass {np.mean(w5[single] >= 0.98):.1%} of {single.sum()}  "
          f"multi-BPM pass {np.mean(w5[~single] >= 0.98) if (~single).any() else 0:.1%} of {(~single).sum()}  "
          f"median signed {np.median([r['median_signed'] for r in rows]):+.2f} ms")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
