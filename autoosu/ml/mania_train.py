"""Train the dedicated 4K model from song-disjoint, real mania charts."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import gather_patches
from .mania_data import FRAME_MS, PATCH
from .mania_model import ManiaNet


def bounded_runtime(device: str) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if device == "mps":
        limit = min(1.0, (1024 ** 3) / torch.mps.recommended_max_memory())
        torch.mps.set_per_process_memory_fraction(limit)


class Corpus:
    def __init__(self, root: Path):
        self.root = root
        self.rows = [json.loads(s) for s in (root / "index.jsonl").read_text().splitlines()]
        self.by_split = {s: [r for r in self.rows if r["split"] == s] for s in ("train", "val", "test")}
        self.maps = {}
        self.mels = {}

    def map(self, row):
        key = row["map_id"]
        if key not in self.maps:
            with np.load(self.root / "maps" / f"{key}.npz") as data:
                self.maps[key] = {name: data[name] for name in data.files}
            labels = self.maps[key]["labels"]
            durations = np.full(labels.shape, -100, dtype=np.int64)
            durations[labels == 1] = 0
            for lane in range(4):
                tails = np.flatnonzero(labels[:, lane] == 4)
                for head in np.flatnonzero(labels[:, lane] == 2):
                    later = tails[tails > head]
                    if len(later):
                        durations[head, lane] = min(32, int(later[0] - head))
            self.maps[key]["durations"] = durations
        return self.maps[key]

    def mel(self, row):
        key = row["audio_hash"]
        if key not in self.mels:
            self.mels[key] = np.load(self.root / "tracks" / f"{key}.npy", mmap_mode="r")
        return self.mels[key]

    def batch(self, rows, starts, length: int, device: str):
        patches, phases, beats, conds, labels, durations = [], [], [], [], [], []
        for row, start in zip(rows, starts):
            m = self.map(row)
            end = min(len(m["times"]), start + length)
            frames = np.rint(m["times"][start:end] / FRAME_MS).astype(np.int64)
            p = gather_patches(self.mel(row), frames).astype(np.float32) / 255.0
            assert p.shape[1] == PATCH
            valid = len(p)
            patches.append(np.pad(p, ((0, length - valid), (0, 0), (0, 0))))
            phases.append(np.pad(m["phase"][start:end], (0, length - valid)))
            beats.append(np.pad(m["beat_ms"][start:end], (0, length - valid), constant_values=500))
            labels.append(np.pad(m["labels"][start:end].astype(np.int64),
                                 ((0, length - valid), (0, 0)), constant_values=-100))
            durations.append(np.pad(m["durations"][start:end],
                                    ((0, length - valid), (0, 0)), constant_values=-100))
            conds.append((row["target_nps"] / 12, row["od"] / 10))
        def tensor(items, dtype):
            return torch.as_tensor(np.asarray(items), dtype=dtype, device=device)
        return (tensor(patches, torch.float32), tensor(phases, torch.long),
                tensor(beats, torch.float32), tensor(conds, torch.float32),
                tensor(labels, torch.long), tensor(durations, torch.long))


def loss_on_split(model, corpus, split, length, device, weights):
    model.eval()
    losses = []
    with torch.inference_mode():
        for row in corpus.by_split[split]:
            n = len(corpus.map(row)["times"])
            for start in range(0, n, length):
                batch = corpus.batch([row], [start], length, device)
                lane_logits, chord_logits, duration_logits = model(*batch[:4])
                losses.append(float(objective(lane_logits, chord_logits, duration_logits,
                                              batch[4], batch[5], weights)))
    return float(np.mean(losses))


def objective(lane_logits, chord_logits, duration_logits, labels, durations, weights):
    lane_loss = (nn.functional.cross_entropy(lane_logits.reshape(-1, 5), labels.reshape(-1),
                                              weight=weights, ignore_index=-100)
                 if torch.any(labels != -100) else lane_logits.sum() * 0)
    chord_targets = (((labels == 1) | (labels == 2)).long() *
                     torch.tensor([1, 2, 4, 8], device=labels.device)).sum(-1)
    chord_targets[labels[:, :, 0] == -100] = -100
    chord_weights = torch.ones(16, device=labels.device)
    chord_weights[0] = 0.4
    chord_loss = (nn.functional.cross_entropy(chord_logits.reshape(-1, 16),
                                               chord_targets.reshape(-1),
                                               weight=chord_weights, ignore_index=-100)
                  if torch.any(chord_targets != -100) else chord_logits.sum() * 0)
    duration_weights = torch.ones(33, device=labels.device)
    duration_weights[0] = 0.6
    duration_loss = (nn.functional.cross_entropy(duration_logits.reshape(-1, 33),
                                                  durations.reshape(-1), weight=duration_weights,
                                                  ignore_index=-100)
                     if torch.any(durations != -100) else duration_logits.sum() * 0)
    return 0.35 * lane_loss + chord_loss + 0.5 * duration_loss


def train(args):
    if min(args.steps, args.batch, args.length, args.eval_every, args.patience) <= 0:
        raise ValueError("steps, batch, length, eval-every and patience must be positive")
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    bounded_runtime(device)
    corpus = Corpus(args.data)
    model = ManiaNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    weights = torch.tensor([0.5, 1.0, 4.0, 1.0, 1.5], device=device)
    rng = random.Random(args.seed)
    best = float("inf")
    stale = 0
    started = time.monotonic()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    train_rows = corpus.by_split["train"]
    nps = np.asarray([r["target_nps"] for r in train_rows])
    difficulty_nps = dict(zip(("Easy", "Normal", "Hard", "Insane"),
                              [float(x) for x in np.quantile(nps, [0.2, 0.4, 0.65, 0.85])]))
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        rows = rng.choices(train_rows, k=args.batch)
        starts = [rng.randrange(max(1, len(corpus.map(row)["times"]) - args.length + 1)) for row in rows]
        batch = corpus.batch(rows, starts, args.length, device)
        optimizer.zero_grad(set_to_none=True)
        lane_logits, chord_logits, duration_logits = model(*batch[:4])
        loss = objective(lane_logits, chord_logits, duration_logits, batch[4], batch[5], weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = loss_on_split(model, corpus, "val", args.length, device, weights)
            if not math.isfinite(val):
                raise RuntimeError(f"non-finite validation loss at step {step}")
            elapsed = time.monotonic() - started
            record = dict(step=step, train_loss=float(loss.detach()), val_loss=val,
                          elapsed_s=round(elapsed, 2), steps_per_s=round(step / elapsed, 3))
            history.append(record)
            print(json.dumps(record), flush=True)
            if val < best - 1e-4:
                best, stale = val, 0
                torch.save(model.checkpoint(seed=args.seed, step=step, val_loss=val,
                                            difficulty_nps=difficulty_nps, threshold=0.35,
                                            hold_threshold=0.35, mask_temperature=1.0,
                                            source_manifest=json.loads((args.data / "manifest.json").read_text())),
                           args.out)
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early_stop step={step}", flush=True)
                    break
    report = dict(device=device, parameters=sum(p.numel() for p in model.parameters()),
                  best_val_loss=best, difficulty_nps=difficulty_nps, history=history)
    args.out.with_suffix(".train.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--patience", type=int, default=4)
    args = p.parse_args()
    if min(args.steps, args.batch, args.length, args.eval_every, args.patience) <= 0:
        p.error("steps, batch, length, eval-every and patience must be positive")
    if args.lr <= 0:
        p.error("learning rate must be positive")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
