"""Reference-free evaluation: does the generator behave the same on music no one ever mapped?

For every audio file (a folder of never-mapped tracks, or ranked test songs for comparison) the full
pipeline runs exactly like the CLI (tracker, timing, auto style, all difficulties), and per chart:

* hard constraints: verify_chart, star rating vs target, skipped difficulties;
* timing self-consistency: share of strong attacks on the 1/4|1/3 grid, median distance of the
  tracker's beats from the half-beat grid, red lines / tempo changes;
* structure: archetype, type shares, in-section purity, type/density change at music boundaries,
  density-loudness correlation, density range (same code as the structure harness);
* playability: notes/s, chord rows, LN share, jacks, same-hand runs, longest anchor.

    python scripts/mania4k_eval_unseen.py --audio-dir ~/data/unseen --out unseen.json
    python scripts/mania4k_eval_unseen.py --prepared ~/data/prepared --corpus ~/data/mania4k --split test --out test.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from autoosu.audio import load_audio                                             # noqa: E402
from autoosu.mania4k.generate import DIFFICULTIES, analyse_song, generate_chart, load_models, rules_for  # noqa: E402
from autoosu.mania4k.onsets import onset_peaks                                   # noqa: E402
from autoosu.mania4k.planner import choose_archetype                             # noqa: E402
from autoosu.mania4k.style import analysis_descriptors                           # noqa: E402
from autoosu.mania4k.structure import ARCHETYPES                                 # noqa: E402
from autoosu.mania4k.timing import pick_beats, snap_rate, tracker_activations    # noqa: E402
from autoosu.mania4k.verify import verify_chart                                  # noqa: E402
from mania4k_eval import pattern_stats, stats                                    # noqa: E402
from mania4k_eval_structure import structure_metrics                             # noqa: E402


def timing_self_check(an, beat_logit) -> dict:
    o, h = onset_peaks(an.features.env, min_height=0.25)
    w = np.minimum(h, 1.0) ** 1.5
    reds = an.timing.red_lines
    seg = [s for s in an.timing.segments]
    rates = [snap_rate(o[(o >= s.start_ms) & (o < s.end_ms)], w[(o >= s.start_ms) & (o < s.end_ms)], s.period, s.phase)
             for s in seg]
    beats = pick_beats(beat_logit)
    devs = []
    for t in beats:
        r = max((r for r in reds if r.time <= t + 1), key=lambda r: r.time, default=reds[0])
        u = (t - r.time) / (r.beat_ms / 2)
        devs.append(abs(u - round(u)) * r.beat_ms / 2)
    return dict(bpm=round(reds[0].bpm, 3), red_lines=len(reds), kind=an.timing.kind,
                snap_rate=float(np.average(rates, weights=[s.end_ms - s.start_ms for s in seg])) if seg else 0.0,
                tracker_dev_ms=float(np.median(devs)) if devs else 99.0)


def run_song(path: Path, label: str, models, rng, beat_logits=None, save_dir: str = "") -> dict:
    y, sr = load_audio(path, sr=22050)
    if beat_logits is None:
        beat_logits = tracker_activations(y, sr)
    an = analyse_song(path, y, sr, beat_logits)
    out = dict(song=label, file=str(path), duration_s=round(an.duration_ms / 1000, 1),
               timing=timing_self_check(an, beat_logits[0]), charts=[])
    mel, high = an.features.mel, an.features.env.high
    for name, target in DIFFICULTIES.items():
        a = choose_archetype(target, "auto", descriptors=analysis_descriptors(an, target))   # as the CLI does
        chart, rep = generate_chart(an, name, target, models=models, archetype=a)
        chk = verify_chart(chart, an.features.env, rules_for(target))
        notes = np.array([(n.time, n.lane, n.end) for n in chart.notes]) if chart.notes else np.zeros((0, 3))
        sm = structure_metrics(notes, chart.red_lines, mel, high, rng) if len(notes) else None
        st = stats(chart.notes) if chart.notes else dict(nps=0, chord=0, ln=0)
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            Path(save_dir, f"{label}_{name}.json").write_text(json.dumps(dict(
                song=label, audio=str(path), name=name, stars=rep.stars, archetype=ARCHETYPES[a],
                reds=[(r.time, r.beat_ms, r.meter) for r in chart.red_lines],
                notes=[(n.time, n.lane, n.end) for n in chart.notes])))
        out["charts"].append(dict(name=name, archetype=ARCHETYPES[a], target=target, stars=rep.stars, ok=chk.ok, problems=chk.problems,
                                  notes=len(chart.notes), nps=st["nps"], chord=st["chord"], ln=st["ln"],
                                  **pattern_stats(chart.notes), structure=sm, plan=rep.plan))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", default="")
    ap.add_argument("--prepared", default=os.environ.get("MANIA4K_PREPARED", str(Path.home() / "data" / "prepared")))
    ap.add_argument("--corpus", default=os.environ.get("MANIA4K_CORPUS", str(Path.home() / "data" / "mania4k")))
    ap.add_argument("--cache", default=os.environ.get("MANIA4K_CACHE", str(Path.home() / "data" / "cache")))
    ap.add_argument("--split", default="test")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-dir", default="", help="write charts as JSON (audio time) for the sync audit")
    args = ap.parse_args()
    models = load_models()
    rng = np.random.default_rng(0)
    jobs = []
    if args.audio_dir:
        jobs = [(p, p.stem, None) for p in sorted(Path(args.audio_dir).glob("*.mp3"))]
    else:
        for f in sorted(Path(args.prepared).glob("*.npz")):
            meta = json.loads(bytes(np.load(f)["meta"]))
            if meta["split"] != args.split:
                continue
            c = Path(args.cache) / f"{meta['set']}.beat.npz"
            logits = None
            if c.exists():
                z = np.load(c)
                logits = (z["beat"].astype(np.float32), z["down"].astype(np.float32))
            jobs.append((Path(args.corpus) / meta["set"] / meta["audio"], meta["set"], logits))
    si, sn = (int(x) for x in args.shard.split("/"))
    rows = []
    for path, label, logits in jobs[si::sn]:
        t0 = time.time()
        try:
            r = run_song(path, label, models, rng, logits, args.save_dir)
        except Exception as exc:                                      # noqa: BLE001
            print(f"{label}: FAILED {exc}", flush=True)
            rows.append(dict(song=label, failed=str(exc)))
            continue
        rows.append(r)
        t = r["timing"]
        print(f"{label:24s} {t['bpm']:7.2f} BPM x{t['red_lines']} snap {t['snap_rate']:.2f} dev {t['tracker_dev_ms']:4.1f}ms "
              + " ".join(f"{c['name'][0]}{c['archetype']}{c['stars']:.1f}{'' if c['ok'] else '!'}" for c in r["charts"])
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
