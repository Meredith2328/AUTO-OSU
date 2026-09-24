"""Local, song-grouped training data for the dedicated osu!mania 4K model.

Only real mode-3, four-key beatmaps and their referenced audio are accepted.
The original music stays in the input directory; prepared data contains log-mel
features and note labels, never a copy of the audio stream.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .osu_parse import _kv, _sections
from .prepare_data import HOP, SR, logmel_uint8

GRID = 8
FRAME_MS = HOP * 1000.0 / SR
N_CLASSES = 5  # empty, tap, LN head, LN body, LN tail
PATCH = 16  # Matches the shared gather_patches implementation.
LANES = 4


@dataclass
class Chart:
    artist: str
    title: str
    version: str
    audio_name: str
    od: float
    red_lines: list[tuple[float, float, int]]
    notes: list[tuple[float, int, float]]  # start ms, lane 0..3, end ms


def parse_mania(text: str) -> Chart | None:
    sec = _sections(text)
    general = _kv(sec.get("General", []))
    diff = _kv(sec.get("Difficulty", []))
    meta = _kv(sec.get("Metadata", []))
    try:
        if int(float(general.get("Mode", "0"))) != 3 or abs(float(diff.get("CircleSize", "0")) - 4) > 0.01:
            return None
        od = float(diff.get("OverallDifficulty", "6"))
    except ValueError:
        return None
    audio_name = general.get("AudioFilename", "").replace("\\", "/").strip()
    if not audio_name:
        return None

    red = []
    for line in sec.get("TimingPoints", []):
        p = line.split(",")
        try:
            t, beat_ms = float(p[0]), float(p[1])
            meter = int(float(p[2])) if len(p) > 2 else 4
            inherited = len(p) > 6 and p[6].strip() == "0"
        except (ValueError, IndexError):
            continue
        if not inherited and math.isfinite(t) and 100 <= beat_ms <= 2000:
            red.append((t, beat_ms, max(1, min(meter, 16))))
    red.sort()
    if not red:
        return None

    notes = []
    for line in sec.get("HitObjects", []):
        p = line.split(",")
        try:
            x, t, typ = int(p[0]), float(p[2]), int(p[3])
            if not (0 <= x < 512 and math.isfinite(t)):
                continue
            lane = min(3, x * 4 // 512)
            if typ & 128:
                end = float(p[5].split(":", 1)[0])
                if not math.isfinite(end) or end <= t:
                    continue
            elif typ & 1:
                end = t
            else:
                continue
        except (ValueError, IndexError):
            continue
        notes.append((t, lane, end))
    notes.sort()
    if len(notes) < 20:
        return None
    return Chart(meta.get("ArtistUnicode") or meta.get("Artist", ""),
                 meta.get("TitleUnicode") or meta.get("Title", ""),
                 meta.get("Version", ""), audio_name, od, red, notes)


def grid_from_redlines(red: list[tuple[float, float, int]], duration_ms: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, phase, beats = [], [], []
    for i, (t0, beat_ms, meter) in enumerate(red):
        t1 = min(duration_ms, red[i + 1][0] if i + 1 < len(red) else duration_ms)
        step = beat_ms / GRID
        first = max(0, math.ceil((max(0.0, t0) - t0) / step - 1e-7))
        count = max(0, math.ceil((t1 - t0) / step - 1e-7))
        for k in range(first, count):
            t = t0 + k * step
            if t >= duration_ms or (times and t <= times[-1] + 0.5):
                continue
            times.append(t)
            phase.append(k % (meter * GRID))
            beats.append(beat_ms)
    return (np.asarray(times, np.float32), np.asarray(phase, np.int64),
            np.asarray(beats, np.float32))


def label_chart(chart: Chart, duration_ms: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    times, phase, beats = grid_from_redlines(chart.red_lines, duration_ms)
    labels = np.zeros((len(times), LANES), dtype=np.uint8)
    accepted = 0
    if len(times) < 2:
        return times, phase, beats, labels, accepted
    for start, lane, end in chart.notes:
        if start < 0 or start >= duration_ms:
            continue
        i = int(np.searchsorted(times, start))
        candidates = [j for j in (i - 1, i) if 0 <= j < len(times)]
        if not candidates:
            continue
        j = min(candidates, key=lambda n: abs(float(times[n]) - start))
        left = float(times[j] - times[j - 1]) if j else float(times[1] - times[0])
        right = float(times[j + 1] - times[j]) if j + 1 < len(times) else left
        if abs(float(times[j]) - start) > max(18.0, 0.46 * min(left, right)):
            continue
        if end > start:
            k0 = int(np.searchsorted(times, end))
            ends = [n for n in (k0 - 1, k0) if j < n < len(times)]
            if not ends:
                continue
            k = min(ends, key=lambda n: abs(float(times[n]) - end))
            if abs(float(times[k]) - end) > max(18.0, 0.46 * float(times[k] - times[k - 1])):
                continue
            if np.any(labels[j:k + 1, lane]):
                continue
            labels[j, lane], labels[k, lane] = 2, 4
            labels[j + 1:k, lane] = 3
        else:
            if labels[j, lane]:
                continue
            labels[j, lane] = 1
        accepted += 1
    return times, phase, beats, labels, accepted


def _song_key(chart: Chart) -> str:
    def norm(s: str) -> str:
        return re.sub(r"\W+", "", s.casefold(), flags=re.UNICODE)
    return norm(chart.artist) + "|" + norm(chart.title)


def _split_groups(rows: list[dict], seed: int) -> dict[str, str]:
    """Union charts sharing metadata OR audio bytes before assigning song splits."""
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        for key in (("song", row["song_key"]), ("audio", row["audio_hash"])):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) < 3:
        raise ValueError(f"Need at least 3 distinct song groups for train/val/test; found {len(groups)}")
    order = sorted(groups, key=lambda root: sorted(rows[i]["song_key"] for i in groups[root])[0])
    random.Random(seed).shuffle(order)
    n = len(order)
    n_test = max(1, round(n * 0.1))
    n_val = max(1, round(n * 0.1))
    if n_test + n_val >= n:
        n_test = n_val = 1
    split_for_root = {root: ("test" if rank < n_test else "val" if rank < n_test + n_val else "train")
                      for rank, root in enumerate(order)}
    for root, indices in groups.items():
        for i in indices:
            rows[i]["group"] = str(root)
            rows[i]["split"] = split_for_root[root]
    return {str(root): split_for_root[root] for root in groups}


def prepare(input_dir: Path, out: Path, seed: int = 42, limit: int = 0) -> dict:
    paths = sorted(input_dir.rglob("*.osz"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No .osz files under {input_dir}")
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    skipped: list[str] = []
    seen_maps: set[str] = set()
    for path in paths:
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                audio_lookup = {name.replace("\\", "/").casefold(): name for name in names}
                charts = []
                for name in names:
                    if not name.lower().endswith(".osu"):
                        continue
                    raw = archive.read(name)
                    chart = parse_mania(raw.decode("utf-8-sig", "replace"))
                    if chart is None:
                        continue
                    wanted = chart.audio_name.casefold()
                    audio_name = audio_lookup.get(wanted)
                    if audio_name is None:
                        matches = [n for n in names if Path(n).name.casefold() == Path(wanted).name.casefold()]
                        if len(matches) != 1:
                            skipped.append(f"{path.name}:{name}: referenced audio missing or ambiguous")
                            continue
                        audio_name = matches[0]
                    charts.append((name, raw, chart, audio_name))
                for name, raw, chart, audio_name in charts:
                    audio_bytes = archive.read(audio_name)
                    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
                    track = out / "tracks" / f"{audio_hash}.npy"
                    track_meta = out / "tracks" / f"{audio_hash}.json"
                    if track.exists():
                        mel = np.load(track, mmap_mode="r")
                        if track_meta.exists():
                            duration_ms = json.loads(track_meta.read_text())["duration_ms"]
                        else:
                            import soundfile as sf

                            try:
                                duration_ms = sf.info(io.BytesIO(audio_bytes)).duration * 1000
                            except RuntimeError:
                                _, duration = logmel_uint8(audio_bytes)
                                duration_ms = duration * 1000
                            track_meta.write_text(json.dumps({"duration_ms": duration_ms}), encoding="utf-8")
                    else:
                        mel, duration = logmel_uint8(audio_bytes)
                        if not 10 <= duration <= 600:
                            skipped.append(f"{path.name}:{name}: duration {duration:.1f}s outside 10..600")
                            continue
                        np.save(track, mel)
                        duration_ms = float(duration * 1000)
                        track_meta.write_text(json.dumps({"duration_ms": duration_ms}), encoding="utf-8")
                    times, phase, beats, labels, accepted = label_chart(chart, duration_ms)
                    if len(times) < 32 or accepted < 20 or accepted / len(chart.notes) < 0.7:
                        skipped.append(f"{path.name}:{name}: {accepted}/{len(chart.notes)} notes on 1/8 grid")
                        continue
                    chart_hash = hashlib.sha256(raw + audio_hash.encode()).hexdigest()
                    if chart_hash in seen_maps:
                        continue
                    seen_maps.add(chart_hash)
                    np.savez_compressed(out / "maps" / f"{chart_hash}.npz", times=times, phase=phase,
                                        beat_ms=beats, labels=labels, audio_hash=audio_hash,
                                        duration_ms=np.float32(duration_ms),
                                        raw_heads=np.asarray([(t, lane) for t, lane, _ in chart.notes],
                                                             dtype=np.float32).reshape(-1, 2))
                    nps = accepted / max(1.0, (times[-1] - times[0]) / 1000)
                    rows.append(dict(map_id=chart_hash, audio_hash=audio_hash, song_key=_song_key(chart),
                                     source=str(path), osu_name=name, artist=chart.artist, title=chart.title,
                                     version=chart.version, od=float(chart.od), target_nps=round(float(nps), 3),
                                     notes=int(accepted), total_notes=len(chart.notes), duration_s=round(float(duration_ms) / 1000, 2)))
        except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
            skipped.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if not rows:
        raise ValueError(f"No usable real mania 4K maps; first skips: {skipped[:5]}")
    groups = _split_groups(rows, seed)
    with (out / "index.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = dict(seed=seed, source_packages=len(paths), maps=len(rows), song_groups=len(groups),
                   accepted_notes=sum(r["notes"] for r in rows),
                   total_notes=sum(r["total_notes"] for r in rows),
                   splits={s: sum(v == s for v in groups.values()) for s in ("train", "val", "test")},
                   skipped=skipped)
    (out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real osu!mania 4K .osz files for local model training")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.out, args.seed, args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
