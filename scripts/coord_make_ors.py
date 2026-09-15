"""Convert the HuggingFace shards into the ORS layout osu-diffusion trains on:

    <out>/Track00000/beatmaps/<beatmap_id>.osu
    <out>/Track00000/metadata.json   {"Beatmaps": {"<beatmap_id>": {"StandardStarRating": {"0": sr, "64": sr_dt}}}}

Only osu!standard ranked/approved/loved maps; star ratings from rosu-pp (nomod and DoubleTime) so both
values come from the same calculator. One Track folder per audio track. Idempotent per shard.

    python scripts/coord_make_ors.py --shards data/hf/compressed --out data/ors --workers 24
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple


def _keep(b: dict) -> bool:
    return b.get("mode") == 0 and b.get("approved") in (1, 2, 4) and 0.5 <= float(b.get("difficultyrating") or 0) <= 10.0


def convert_track(job: Tuple[int, dict]) -> Optional[Tuple[int, int]]:
    """Write one Track folder. Returns (track_index, n_maps) or None if nothing usable."""
    track_index, meta = job
    import rosu_pp_py as rosu
    from slider import Beatmap

    maps = {}
    for b in meta["beatmaps"]:
        if not _keep(b):
            continue
        content = b["content"]
        try:
            Beatmap.parse(content)                     # must be readable by the trainer's parser
            bm = rosu.Beatmap(content=content)
            sr0 = float(rosu.Difficulty().calculate(bm).stars)
            sr64 = float(rosu.Difficulty(mods=64).calculate(bm).stars)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[skip beatmap {b.get('beatmap_id')}] {type(e).__name__}: {e}\n")
            continue
        if not (0.3 <= sr0 <= 12.0):
            continue
        maps[str(b["beatmap_id"])] = (content, sr0, sr64)
    if not maps:
        return None
    out = Path(OUT_DIR) / f"Track{track_index:05d}"
    (out / "beatmaps").mkdir(parents=True, exist_ok=True)
    for bid, (content, sr0, sr64) in maps.items():
        (out / "beatmaps" / f"{bid}.osu").write_text(content, encoding="utf-8")
    entries = {bid: {"StandardStarRating": {"0": sr0, "64": sr64}} for bid, (_, sr0, sr64) in maps.items()}
    (out / "metadata.json").write_text(json.dumps({"Beatmaps": entries}), encoding="utf-8")
    return track_index, len(maps)


OUT_DIR = ""


def _init(out_dir: str) -> None:
    global OUT_DIR
    OUT_DIR = out_dir


def iter_meta(shard: Path):
    tf = tarfile.open(shard)
    for m in tf:
        if m.isfile() and m.name.endswith(".json"):
            yield json.load(tf.extractfile(m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/hf/compressed")
    ap.add_argument("--out", default="data/ors")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many tracks (debug)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_file = out / "_progress.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"next_track": 0, "done_shards": []}
    shards = sorted(Path(args.shards).glob("*.tar"))
    t0 = time.perf_counter()
    total_maps = 0
    with Pool(args.workers, initializer=_init, initargs=(str(out.resolve()),)) as pool:
        for shard in shards:
            if shard.name in state["done_shards"]:
                continue
            start = state["next_track"]
            jobs = ((start + i, meta) for i, meta in enumerate(iter_meta(shard)))
            n_tracks = 0
            for res in pool.imap(convert_track, jobs, chunksize=4):
                n_tracks += 1
                if res:
                    total_maps += res[1]
            # tracks without usable maps leave gaps in the numbering; fill them with empty folders so
            # the trainer's contiguous Track index range stays valid
            for i in range(start, start + n_tracks):
                d = out / f"Track{i:05d}" / "beatmaps"
                if not d.exists():
                    d.mkdir(parents=True)
                    (out / f"Track{i:05d}" / "metadata.json").write_text('{"Beatmaps": {}}')
            state["next_track"] = start + n_tracks
            state["done_shards"].append(shard.name)
            state_file.write_text(json.dumps(state))
            print(f"{shard.name}: tracks -> {state['next_track']}, maps so far {total_maps}, {time.perf_counter() - t0:.0f} s", flush=True)
            if args.limit and state["next_track"] >= args.limit:
                break
    n_maps = sum(1 for _ in out.glob("Track*/beatmaps/*.osu"))
    print(f"done: {state['next_track']} tracks, {n_maps} std beatmaps -> {out}  (set data.end={state['next_track']} in the config)")


if __name__ == "__main__":
    main()
