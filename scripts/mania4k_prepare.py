"""Turn the ranked 4K corpus into training tensors for the note and pattern models.

Per beatmap set: log-mel + onset envelopes of the audio. Per chart: the chart's own beat grid
(1/8 + 1/6 positions) in audio time, and per-tick labels (notes, LN heads and lengths, lane masks),
plus the chart's star rating (rosu-pp, identical to osu!), OD and HP.

    python scripts/mania4k_prepare.py --corpus ~/data/mania4k --out ~/data/prepared
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoosu.audio import load_audio                                      # noqa: E402
from autoosu.mania4k.chart import RedLine, load_osu                       # noqa: E402
from autoosu.mania4k.features import build_grid, env_matrix, local_loudness, song_features  # noqa: E402

DEFAULT_SHIFT = 28.0


def split_of(set_id: str) -> str:
    h = zlib.crc32(set_id.encode()) % 100
    return "test" if h < 12 else ("val" if h < 18 else "train")


def chart_shift(env, heads_ms: np.ndarray) -> float:
    """Audio-minus-chart offset (ms) that best puts the chart's note heads on onsets."""
    lags = np.arange(10.0, 50.01, 0.5)
    sc = [env.at(env.full, heads_ms + l).mean() for l in lags]
    best = float(lags[int(np.argmax(sc))])
    return best if 14.0 <= best <= 46.0 else DEFAULT_SHIFT


def prepare_set(args) -> str:
    d, out = args
    target = out / f"{d.name}.npz"
    if target.exists():
        return f"{d.name} cached"
    import rosu_pp_py as rosu

    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    try:
        y, sr = load_audio(d / meta["audio"], sr=22050)
    except Exception as exc:                                         # noqa: BLE001
        return f"{d.name} audio failed: {exc}"
    feat = song_features(y, sr)
    arrays = {"mel": feat.mel,
              "env": np.stack([feat.env.full, feat.env.low, feat.env.mid, feat.env.high]).astype(np.float16)}
    charts = []
    for n, f in enumerate(sorted(d.glob("*.osu"))):
        c = load_osu(f)
        if c is None or len(c.notes) < 50:
            continue
        stars = float(rosu.Difficulty().calculate(rosu.Beatmap(path=str(f))).stars)
        heads = np.array(c.heads)
        shift = chart_shift(feat.env, heads)
        reds = [RedLine(r.time + shift, r.beat_ms, r.meter) for r in c.red_lines]
        grid = build_grid(reds, max(0.0, heads[0] + shift - 4000.0), min(feat.duration_ms, max(n_.end or n_.time for n_ in c.notes) + shift + 2000.0))
        times = grid.times
        count = np.zeros(len(times), np.int8)
        mask = np.zeros(len(times), np.int8)
        ln_head = np.zeros(len(times), np.int8)
        ln_beats = np.zeros(len(times), np.float32)
        missed = 0
        for note in c.notes:
            t = note.time + shift
            i = int(np.searchsorted(times, t))
            cand = [j for j in (i - 1, i) if 0 <= j < len(times) and abs(times[j] - t) <= 2.5]
            if not cand:
                missed += 1
                continue
            j = min(cand, key=lambda j: abs(times[j] - t))
            if mask[j] & (1 << note.lane):
                continue
            mask[j] |= 1 << note.lane
            count[j] += 1
            if note.is_hold:
                ln_head[j] = 1
                ln_beats[j] = max(ln_beats[j], (note.end - note.time) / grid.beat_ms[j])
        key = f"c{n}"
        arrays[f"{key}_times"] = times.astype(np.float64)
        arrays[f"{key}_pos"] = grid.pos.astype(np.int8)
        arrays[f"{key}_div"] = grid.div.astype(np.int8)
        arrays[f"{key}_bim"] = grid.beat_in_measure.astype(np.int8)
        arrays[f"{key}_beatms"] = grid.beat_ms
        arrays[f"{key}_beat"] = grid.beat
        arrays[f"{key}_env"] = env_matrix(feat.env, times).astype(np.float16)
        arrays[f"{key}_loud"] = local_loudness(feat.mel, times)
        arrays[f"{key}_count"] = count
        arrays[f"{key}_mask"] = mask
        arrays[f"{key}_lnhead"] = ln_head
        arrays[f"{key}_lnbeats"] = ln_beats
        notes = np.array([(nn.time + shift, nn.lane, (nn.end + shift) if nn.is_hold else 0.0) for nn in c.notes])
        arrays[f"{key}_notes"] = notes
        charts.append(dict(key=key, file=f.name, version=c.version, stars=stars, od=c.od, hp=c.hp,
                           shift=shift, n_notes=len(c.notes), off_grid=missed,
                           reds=[(r.time + shift, r.beat_ms, r.meter) for r in c.red_lines]))
    if not charts:
        return f"{d.name} no charts"
    arrays["meta"] = np.frombuffer(json.dumps(dict(set=d.name, split=split_of(d.name), audio=meta["audio"],
                                                   title=meta["title"], artist=meta["artist"],
                                                   charts=charts)).encode(), np.uint8)
    tmp = target.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, target)
    return f"{d.name} {len(charts)} charts, shifts {[round(c['shift'], 1) for c in charts]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.environ.get("MANIA4K_CORPUS", str(Path.home() / "data" / "mania4k")))
    ap.add_argument("--out", default=os.environ.get("MANIA4K_PREPARED", str(Path.home() / "data" / "prepared")))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = [d for d in sorted(Path(args.corpus).iterdir()) if (d / "meta.json").exists()]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(prepare_set, [(d, out) for d in dirs]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
