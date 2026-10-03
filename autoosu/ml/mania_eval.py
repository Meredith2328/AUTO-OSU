"""Evaluate held-out real charts by note-head and onset F1, never by empty-tick accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .mania_infer import decode_predictions, predict_logits
from .mania_model import ManiaNet
from .mania_train import Corpus, bounded_runtime


def match(pred, truth, tolerance: float, lane: bool) -> tuple[int, int, int]:
    def grouped(items):
        buckets = {}
        for t, key in items:
            buckets.setdefault(key if lane else 0, []).append(float(t))
        return {key: sorted(set(round(t, 1) for t in times)) for key, times in buckets.items()}
    p, t = grouped(pred), grouped(truth)
    hits = 0
    for key in p.keys() | t.keys():
        pp, tt = p.get(key, []), t.get(key, [])
        i = j = 0
        while i < len(pp) and j < len(tt):
            if abs(pp[i] - tt[j]) <= tolerance:
                hits += 1
                i += 1
                j += 1
            elif pp[i] < tt[j]:
                i += 1
            else:
                j += 1
    return hits, sum(map(len, p.values())), sum(map(len, t.values()))


def score(counts):
    hit, pred, true = counts
    precision = hit / pred if pred else 0.0
    recall = hit / true if true else 0.0
    return dict(precision=round(precision, 4), recall=round(recall, 4),
                f1=round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
                matched=hit, predicted=pred, reference=true)


def evaluate(corpus, model, ckpt, split, device, thresholds, audio_shift_frames=0,
             hold_threshold=0.45, mask_temperature=1.0):
    caches = []
    for row in corpus.by_split[split]:
        m = corpus.map(row)
        mel = corpus.mel(row)
        if audio_shift_frames:
            mel = np.roll(mel, audio_shift_frames, axis=0)
        lane_logits, chord_logits, duration_logits = predict_logits(
            model, mel, m["times"], m["phase"], m["beat_ms"],
            row["target_nps"], row["od"], device)
        quant = [(float(m["times"][i]), lane) for i, lane in np.argwhere((m["labels"] == 1) | (m["labels"] == 2))]
        raw = [(float(t), int(lane)) for t, lane in m["raw_heads"]]
        caches.append((row, m, lane_logits, chord_logits, duration_logits, quant, raw))
    results = []
    for threshold in thresholds:
        totals = {name: [0, 0, 0] for name in ("raw_lane_35", "raw_lane_50", "raw_onset_50", "grid_lane_50")}
        predicted_ln = predicted_notes = reference_ln = reference_notes = 0
        no_conflicts = True
        quad_ticks = chord_ticks = 0
        lane_counts = [0, 0, 0, 0]
        for row, m, lane_logits, chord_logits, duration_logits, quant, raw in caches:
            decoded = decode_predictions(lane_logits, chord_logits, duration_logits, m["times"], threshold=threshold,
                                         duration_ms=float(m["duration_ms"]),
                                         hold_threshold=hold_threshold,
                                         seed=int(row["map_id"][:8], 16),
                                         mask_temperature=mask_temperature)
            pred = [(start, lane) for start, lane, _, _ in decoded]
            predicted_notes += len(decoded)
            predicted_ln += sum(end > start for start, _, end, _ in decoded)
            reference_notes += len(raw)
            reference_ln += int(np.sum(m["labels"] == 2))
            lane_ends = [-1] * 4
            for start, lane, end, _ in decoded:
                lane_counts[lane] += 1
                if start <= lane_ends[lane] or end > float(m["duration_ms"]):
                    no_conflicts = False
                lane_ends[lane] = max(start, end)
            counts = np.unique([start for start, _, _, _ in decoded], return_counts=True)[1]
            chord_ticks += int(np.sum(counts >= 2))
            quad_ticks += int(np.sum(counts == 4))
            pairs = {
                "raw_lane_35": match(pred, raw, 35, True),
                "raw_lane_50": match(pred, raw, 50, True),
                "raw_onset_50": match(pred, raw, 50, False),
                "grid_lane_50": match(pred, quant, 50, True),
            }
            for name, triple in pairs.items():
                totals[name] = [x + y for x, y in zip(totals[name], triple)]
        metrics = {name: score(v) for name, v in totals.items()}
        metrics.update(threshold=float(threshold), hold_threshold=float(hold_threshold),
                       mask_temperature=float(mask_temperature), maps=len(caches),
                       density_ratio=round(predicted_notes / reference_notes, 4) if reference_notes else 0,
                       predicted_ln_ratio=round(predicted_ln / predicted_notes, 4) if predicted_notes else 0,
                       reference_ln_ratio=round(reference_ln / reference_notes, 4) if reference_notes else 0,
                       chord_ticks=chord_ticks, quad_ticks=quad_ticks,
                       lane_counts=lane_counts, legal=no_conflicts)
        results.append(metrics)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--device", choices=("cpu", "mps"), default="mps")
    p.add_argument("--threshold", type=float)
    p.add_argument("--hold-threshold", type=float)
    p.add_argument("--mask-temperature", type=float, default=1.0)
    p.add_argument("--random-baseline", action="store_true",
                   help="also evaluate an untrained same-architecture model at fixed thresholds")
    p.add_argument("--audio-shift-frames", type=int, default=0,
                   help="validation ablation: roll audio features by this many 10 ms frames")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.mask_temperature <= 0:
        p.error("mask-temperature must be positive")
    bounded_runtime(args.device)
    model, ckpt = ManiaNet.from_checkpoint(args.checkpoint, args.device)
    if args.split == "test" and args.audio_shift_frames:
        p.error("audio ablation is validation-only")
    thresholds = ([args.threshold] if args.threshold is not None else
                  [float(ckpt["threshold"])] if args.split == "test" else
                  np.arange(0.25, 0.76, 0.05))
    corpus = Corpus(args.data)
    result = evaluate(corpus, model, ckpt, args.split, args.device, thresholds,
                      args.audio_shift_frames,
                      args.hold_threshold if args.hold_threshold is not None else ckpt.get("hold_threshold", 0.45),
                      args.mask_temperature)
    if args.random_baseline:
        torch.manual_seed(42)
        random_model = ManiaNet(model.cfg).to(args.device).eval()
        random_result = evaluate(corpus, random_model, ckpt, args.split, args.device, thresholds,
                                 hold_threshold=args.hold_threshold if args.hold_threshold is not None
                                 else ckpt.get("hold_threshold", 0.45),
                                 mask_temperature=args.mask_temperature)
        result = {"trained": result, "untrained_random": random_result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
