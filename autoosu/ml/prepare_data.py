"""Build the training corpus from the HuggingFace `project-riz/osu-beatmaps` shards.

    python -m autoosu.ml.prepare_data --shards data/hf/compressed --out data/prep --workers 8

Per track  : data/prep/tracks/<shard>_<seq>.npy   uint8 log-mel, shape (T, 64), 10 ms frames at 16 kHz
Per beatmap: data/prep/maps/<beatmap_id>.npz      objects (N,4), red lines (R,3), sv points, settings
Index      : data/prep/index.jsonl                 one line per std beatmap with metadata

Only osu!standard (mode 0) ranked/approved/loved maps are kept.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .osu_parse import parse_osu

SR = 16000
HOP = 160            # 10 ms
N_MELS = 64
N_FFT = 1024
MEL_LO, MEL_HI = -12.0, 4.0     # log-mel range mapped to uint8


def logmel_uint8(opus_bytes: bytes) -> Tuple[np.ndarray, float]:
    import librosa
    import soundfile as sf

    try:
        y, sr = sf.read(io.BytesIO(opus_bytes), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    except Exception:
        # libsndfile rejects a few files; ffmpeg decodes them fine (system binary, else the one
        # bundled with the imageio-ffmpeg wheel so no system install is needed)
        import shutil
        import subprocess
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        raw = subprocess.run([ffmpeg, "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                              "-ar", str(SR), "pipe:1"], input=opus_bytes, capture_output=True, check=True).stdout
        y = np.frombuffer(raw, dtype=np.float32).copy()
    peak = float(np.abs(y).max()) or 1.0
    y = y / peak
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=20, fmax=8000)
    lm = np.log(mel + 1e-6).T                       # (T, 64)
    q = np.clip((lm - MEL_LO) / (MEL_HI - MEL_LO), 0, 1) * 255
    return q.astype(np.uint8), len(y) / SR


def _keep(b: dict) -> bool:
    return (b.get("mode") == 0 and b.get("approved") in (1, 2, 4)
            and 0.5 <= float(b.get("difficultyrating") or 0) <= 10.0)


def process_track(job: Tuple[str, str, bytes, dict, str]) -> Optional[List[dict]]:
    shard, seq, opus, meta, out_dir = job
    out = Path(out_dir)
    try:
        maps = [b for b in meta["beatmaps"] if _keep(b)]
        if not maps or float(meta.get("audio_length", 0)) > 600:
            return None
        parsed = []
        for b in maps:
            pb = parse_osu(b["content"])
            if pb.mode != 0 or pb.n_objects < 30 or not pb.red_lines:
                continue
            parsed.append((b, pb))
        if not parsed:
            return None
        feat, dur = logmel_uint8(opus)
        track_name = f"{shard}_{seq}"
        np.save(out / "tracks" / f"{track_name}.npy", feat)
        rows = []
        for b, pb in parsed:
            bid = int(b["beatmap_id"])
            np.savez_compressed(
                out / "maps" / f"{bid}.npz",
                objects=pb.objects.astype(np.float32),
                red_lines=np.array([(r.time, r.beat_length, r.meter) for r in pb.red_lines], dtype=np.float32),
                sv_points=np.array(pb.sv_points, dtype=np.float32).reshape(-1, 2),
                slider_multiplier=np.float32(pb.slider_multiplier),
            )
            year = int(str(b.get("approved_date") or b.get("submit_date") or "2015")[:4])
            rows.append(dict(
                beatmap_id=bid, beatmapset_id=int(b.get("beatmapset_id") or 0), track=track_name,
                sr=float(b["difficultyrating"]), cs=float(b.get("diff_size") or pb.settings.get("CircleSize", 4)),
                ar=float(b.get("diff_approach") or pb.settings.get("ApproachRate", 8)),
                od=float(b.get("diff_overall") or pb.settings.get("OverallDifficulty", 6)),
                hp=float(b.get("diff_drain") or pb.settings.get("HPDrainRate", 5)),
                year=year, creator_id=int(b.get("creator_id") or 0), bpm=float(b.get("bpm") or 0),
                n_objects=pb.n_objects, n_red=len(pb.red_lines), length_s=round(dur, 2),
                version=str(b.get("version", ""))[:40], approved=int(b.get("approved")),
            ))
        return rows
    except Exception as e:  # noqa: BLE001 - one bad track must not kill the run
        sys.stderr.write(f"[skip {shard}/{seq}] {type(e).__name__}: {e}\n")
        return None


def iter_jobs(shard_path: Path, out_dir: Path):
    tf = tarfile.open(shard_path)
    pending: Dict[str, dict] = {}
    for m in tf:
        if not m.isfile():
            continue
        seq, ext = m.name.rsplit(".", 1)
        entry = pending.setdefault(seq, {})
        if ext == "json":
            entry["meta"] = json.load(tf.extractfile(m))
        elif ext in ("opus", "mp3", "ogg", "wav"):
            entry["audio"] = tf.extractfile(m).read()
        if "meta" in entry and "audio" in entry:
            del pending[seq]
            yield (shard_path.stem, seq, entry["audio"], entry["meta"], str(out_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/hf/compressed")
    ap.add_argument("--out", default="data/prep")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many tracks (debug)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    index = out / "index.jsonl"
    done_tracks = set()
    if index.exists():
        for line in index.open(encoding="utf-8"):
            done_tracks.add(json.loads(line)["track"])

    shards = sorted(Path(args.shards).glob("*.tar"))
    t0 = time.perf_counter()
    n_tracks = n_maps = 0
    with Pool(args.workers) as pool, index.open("a", encoding="utf-8") as idx:
        for shard in shards:
            jobs = (j for j in iter_jobs(shard, out) if f"{j[0]}_{j[1]}" not in done_tracks)
            for rows in pool.imap_unordered(process_track, jobs, chunksize=2):
                if not rows:
                    continue
                for r in rows:
                    idx.write(json.dumps(r, ensure_ascii=False) + "\n")
                idx.flush()
                n_tracks += 1
                n_maps += len(rows)
                if n_tracks % 100 == 0:
                    print(f"{shard.name}: {n_tracks} tracks, {n_maps} maps, {time.perf_counter() - t0:.0f} s", flush=True)
                if args.limit and n_tracks >= args.limit:
                    print("limit reached")
                    return
    print(f"done: {n_tracks} tracks, {n_maps} std beatmaps in {time.perf_counter() - t0:.0f} s -> {index}")


if __name__ == "__main__":
    main()
