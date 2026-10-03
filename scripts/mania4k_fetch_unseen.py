"""Fetch never-mapped music for generalisation tests: Creative Commons tracks from Internet Archive
netlabel collections, across genres, skipping anything whose title shows up on the osu! mirror.

    python scripts/mania4k_fetch_unseen.py --out ~/data/unseen --per-genre 2
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

GENRES = ("rock", "metal", "pop", "electronic", "techno", "hip hop", "jazz", "classical", "folk",
          "ambient", "funk", "punk")
UA = {"User-Agent": "AUTO-OSU generalisation test (research)"}


def get(url: str, timeout: float = 60.0) -> bytes:
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:                      # noqa: BLE001
            if k == 3:
                raise
            time.sleep(2 ** (k + 1))
    return b""


def seconds(v) -> float:
    """Archive lengths come as '289.3' or '04:49'."""
    try:
        if isinstance(v, str) and ":" in v:
            parts = [float(x) for x in v.split(":")]
            return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
        return float(v or 0)
    except ValueError:
        return 0.0


def on_osu(title: str) -> bool:
    q = urllib.parse.urlencode({"q": title, "amount": 5})
    try:
        hits = json.loads(get(f"https://osu.direct/api/v2/search?{q}", 30))
    except Exception:                          # noqa: BLE001
        return True                            # unknown: be conservative
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())     # noqa: E731
    return any(norm(h.get("title", "")) == norm(title) for h in hits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-genre", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    prev = json.loads((out / "index.json").read_text()) if (out / "index.json").exists() else []
    for genre in GENRES:
        q = f'collection:(netlabels) AND mediatype:(audio) AND subject:("{genre}")'
        url = ("https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(
            {"q": q, "fl[]": "identifier", "rows": 40, "output": "json", "sort[]": "downloads desc"}))
        ids = [d["identifier"] for d in json.loads(get(url))["response"]["docs"]]
        got = len(list(out.glob(f"{genre.replace(' ', '_')}_*.mp3")))
        index += [e for e in prev if e["genre"] == genre]
        for ident in ids:
            if got >= args.per_genre:
                break
            if ident in {e["identifier"] for e in index}:
                continue                               # one track per album, across genres too
            meta = json.loads(get(f"https://archive.org/metadata/{ident}"))
            files = [f for f in meta.get("files", []) if f.get("name", "").lower().endswith(".mp3")
                     and 120 <= seconds(f.get("length")) <= 360]
            if not files:
                continue
            f = files[min(1, len(files) - 1)]                  # skip intros: take the 2nd track
            title = f.get("title") or Path(f["name"]).stem
            if on_osu(title):
                continue
            dest = out / f"{genre.replace(' ', '_')}_{got}.mp3"
            dest.write_bytes(get(f"https://archive.org/download/{ident}/{urllib.parse.quote(f['name'])}", 300))
            index.append(dict(file=dest.name, genre=genre, identifier=ident, title=title,
                              artist=meta.get("metadata", {}).get("creator", ""),
                              license=meta.get("metadata", {}).get("licenseurl", ""), length=f.get("length")))
            got += 1
            print(f"{genre:10s} {dest.name}: {title} ({f.get('length')} s)", flush=True)
            (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
