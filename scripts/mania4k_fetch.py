"""Fetch ranked osu!mania 4K beatmapsets as a local reference corpus.

Only the audio file and the 4K .osu difficulties of each set are kept (no storyboards, videos or
backgrounds), together with the mirror's metadata (star ratings, play counts) in ``meta.json``.
The corpus is used to learn chart statistics and to audit generated charts against human charts
of the same audio; it is never redistributed.

    python scripts/mania4k_fetch.py --out ~/data/mania4k --sets 400
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

MIRROR = "https://osu.direct/api"
UA = {"User-Agent": "AUTO-OSU corpus fetcher (research; github.com/kanze1/AUTO-OSU)"}


def _get(url: str, timeout: float = 120.0, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:                      # noqa: BLE001 - network errors of every flavour
            if k == tries - 1:
                raise
            time.sleep(2 ** (k + 1))
    raise RuntimeError("unreachable")


def search(offset: int, amount: int = 100, sort: str = "ranked_date:desc", query: str = "") -> list[dict]:
    params = {"mode": 3, "status": 1, "amount": amount, "offset": offset, "sort": sort}
    if query:
        params["q"] = query
    return json.loads(_get(f"{MIRROR}/v2/search?{urllib.parse.urlencode(params)}", timeout=60))


def four_key(bm: dict) -> bool:
    return bm.get("mode_int") == 3 and abs(float(bm.get("cs", 0)) - 4) < 0.01 and not bm.get("convert")


def _audio_name(osu_text: str) -> str:
    m = re.search(r"^AudioFilename\s*:\s*(.+?)\s*$", osu_text, re.M)
    return m.group(1).replace("\\", "/") if m else ""


def _mode_cs(osu_text: str) -> tuple[int, float]:
    mode = re.search(r"^Mode\s*:\s*(\d+)", osu_text, re.M)
    cs = re.search(r"^CircleSize\s*:\s*([\d.]+)", osu_text, re.M)
    return (int(mode.group(1)) if mode else 0), (float(cs.group(1)) if cs else 0.0)


def fetch_set(meta: dict, out: Path) -> bool:
    target = out / str(meta["id"])
    if (target / "meta.json").exists():
        return True
    blob = _get(f"{MIRROR}/d/{meta['id']}?noVideo=1", timeout=300)
    z = zipfile.ZipFile(io.BytesIO(blob))
    charts: dict[str, str] = {}
    for name in z.namelist():
        if name.lower().endswith(".osu"):
            text = z.read(name).decode("utf-8-sig", errors="replace")
            mode, cs = _mode_cs(text)
            if mode == 3 and abs(cs - 4) < 0.01:
                charts[Path(name).name] = text
    audio = {_audio_name(t) for t in charts.values()} - {""}
    lookup = {n.lower(): n for n in z.namelist()}
    if not charts or len(audio) != 1 or next(iter(audio)).lower() not in lookup:
        return False
    audio_name = next(iter(audio))
    target.mkdir(parents=True, exist_ok=True)
    (target / Path(audio_name).name).write_bytes(z.read(lookup[audio_name.lower()]))
    for name, text in charts.items():
        (target / name).write_text(text, encoding="utf-8")
    keep = {k: meta.get(k) for k in ("id", "title", "artist", "creator", "bpm", "play_count",
                                     "favourite_count", "ranked_date", "tags", "genre_id")}
    keep["audio"] = Path(audio_name).name
    keep["beatmaps"] = [{k: b.get(k) for k in ("id", "version", "difficulty_rating", "accuracy", "drain",
                                                "bpm", "count_circles", "count_sliders", "total_length")}
                        for b in meta["beatmaps"] if four_key(b)]
    (target / "meta.json").write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path.home() / "data" / "mania4k"))
    ap.add_argument("--sets", type=int, default=400)
    ap.add_argument("--sort", default="ranked_date:desc")
    ap.add_argument("--max-length", type=int, default=330, help="skip sets longer than this (s)")
    ap.add_argument("--exclude", default="", help="corpus folder whose sets must not be fetched again")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    have = sum((p / "meta.json").exists() for p in out.iterdir() if p.is_dir())
    offset, seen = 0, set()
    if args.exclude:
        seen |= {int(p.name) for p in Path(args.exclude).iterdir() if p.name.isdigit()}
    while have < args.sets:
        page = search(offset, sort=args.sort)
        if not page:
            break
        offset += len(page)
        for meta in page:
            if have >= args.sets or meta["id"] in seen:
                continue
            seen.add(meta["id"])
            bms = [b for b in meta.get("beatmaps", []) if four_key(b)]
            if not bms or max(b.get("total_length", 0) for b in bms) > args.max_length:
                continue
            if (out / str(meta["id"]) / "meta.json").exists():
                continue
            try:
                ok = fetch_set(meta, out)
            except Exception as exc:          # noqa: BLE001
                print(f"  ! {meta['id']}: {exc}", flush=True)
                continue
            have += ok
            print(f"[{have}/{args.sets}] {meta['id']} {meta['artist']} - {meta['title']} "
                  f"({len(bms)} 4K diffs){'' if ok else ' skipped'}", flush=True)


if __name__ == "__main__":
    main()
