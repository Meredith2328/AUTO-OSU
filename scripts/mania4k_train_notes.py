"""Train the 4K note model (autoosu.mania4k.model.NoteNet) on prepared ranked charts.

    python scripts/mania4k_train_notes.py --data ~/data/prepared --out models/mania4k_notes.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoosu.mania4k.features import MEL_HOP, PATCH                  # noqa: E402
from autoosu.mania4k.model import NoteNet, ln_class                  # noqa: E402
from autoosu.mania4k.onsets import SR                                # noqa: E402

CROP = 768          # ticks per training crop
PAD_FRAMES = 24     # mel context outside the crop


class Corpus:
    def __init__(self, root: Path):
        self.songs, self.charts = {}, {"train": [], "val": [], "test": []}
        for f in sorted(root.glob("*.npz")):
            if f.name.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            meta = json.loads(bytes(z["meta"]))
            self.songs[meta["set"]] = z["mel"]
            for c in meta["charts"]:
                k = c["key"]
                if c["off_grid"] > 0.08 * c["n_notes"]:
                    continue                     # mostly unsnapped / exotic timing: unreliable labels
                arr = {n: z[f"{k}_{n}"] for n in ("times", "pos", "div", "bim", "beatms", "env", "loud",
                                                   "count", "lnhead", "lnbeats")}
                self.charts[meta["split"]].append(dict(set=meta["set"], stars=c["stars"], **arr))
        print({k: len(v) for k, v in self.charts.items()}, "charts;", len(self.songs), "songs", flush=True)


def make_batch(corpus: Corpus, items, crop: int | None):
    """items: list of (chart, start). Returns padded tensors."""
    B = len(items)
    lens = [min(crop, len(c["times"]) - s) if crop else len(c["times"]) for c, s in items]
    T = max(lens)
    fps = SR / MEL_HOP
    frame_ranges = []
    for (c, s), L in zip(items, lens):
        t = c["times"][s:s + L]
        f0 = max(0, int(t[0] / 1000 * fps) - PAD_FRAMES)
        f1 = int(t[-1] / 1000 * fps) + PAD_FRAMES + 1
        frame_ranges.append((f0, f1))
    Fm = max(b - a for a, b in frame_ranges)
    mel = torch.zeros(B, Fm, 64)
    out = {k: torch.zeros(B, T, dtype=torch.long) for k in ("frame", "pos", "div", "bim", "count", "lnlen")}
    for k in ("env",):
        out[k] = torch.zeros(B, T, items[0][0]["env"].shape[1])
    for k in ("loud", "beatms", "lnhead", "mask"):
        out[k] = torch.zeros(B, T)
    stars = torch.zeros(B)
    for b, ((c, s), L, (f0, f1)) in enumerate(zip(items, lens, frame_ranges)):
        m = corpus.songs[c["set"]][f0:f1]
        mel[b, :len(m)] = torch.from_numpy(m.astype(np.float32) / 255.0)
        sl = slice(s, s + L)
        out["frame"][b, :L] = torch.from_numpy(np.round(c["times"][sl] / 1000 * fps).astype(np.int64) - f0)
        for k in ("pos", "div", "bim", "count"):
            out[k][b, :L] = torch.from_numpy(c[k][sl].astype(np.int64))
        out["env"][b, :L] = torch.from_numpy(c["env"][sl].astype(np.float32))
        out["loud"][b, :L] = torch.from_numpy(c["loud"][sl])
        out["beatms"][b, :L] = torch.from_numpy(c["beatms"][sl])
        out["lnhead"][b, :L] = torch.from_numpy(c["lnhead"][sl].astype(np.float32))
        out["lnlen"][b, :L] = ln_class(torch.from_numpy(c["lnbeats"][sl]))
        out["mask"][b, :L] = 1.0
        stars[b] = c["stars"]
    return mel, out, stars


def forward(net, mel, o, stars):
    return net(mel, o["frame"], o["env"], o["pos"], o["div"], o["bim"], o["loud"], o["beatms"], stars)


def losses(pred, o):
    m = o["mask"]
    count = o["count"].clamp(0, 4)
    lc = (F.cross_entropy(pred["count"].transpose(1, 2), count, reduction="none") * m).sum() / m.sum()
    has = (count > 0).float() * m
    lh = (F.binary_cross_entropy_with_logits(pred["lnhead"], o["lnhead"], reduction="none") * has).sum() / has.sum().clamp(min=1)
    lnm = o["lnhead"] * m
    ll = (F.cross_entropy(pred["lnlen"].transpose(1, 2), o["lnlen"], reduction="none") * lnm).sum() / lnm.sum().clamp(min=1)
    return lc, lh, ll


@torch.no_grad()
def evaluate(net, corpus, split: str, max_charts: int = 60):
    net.eval()
    charts = corpus.charts[split][:max_charts]
    tot = np.zeros(3)
    tp = fp = fn = 0
    for c in charts:
        mel, o, stars = make_batch(corpus, [(c, 0)], None)
        pred = forward(net, mel, o, stars)
        lc, lh, ll = losses(pred, o)
        tot += [lc.item(), lh.item(), ll.item()]
        p_note = 1 - torch.softmax(pred["count"], -1)[..., 0]
        y = o["count"] > 0
        hat = p_note > 0.5
        tp += int((hat & y).sum()); fp += int((hat & ~y).sum()); fn += int((~hat & y).sum())
    net.train()
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return dict(count=tot[0] / len(charts), lnhead=tot[1] / len(charts), lnlen=tot[2] / len(charts),
                precision=prec, recall=rec, f1=2 * prec * rec / max(1e-9, prec + rec))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MANIA4K_PREPARED", str(Path.home() / "data" / "prepared")))
    ap.add_argument("--out", default="models/mania4k_notes.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    random.seed(0)
    corpus = Corpus(Path(args.data))
    net = NoteNet(hidden=args.hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    train = corpus.charts["train"]
    crops_per_epoch = sum(max(1, len(c["times"]) // CROP) for c in train)
    steps = args.epochs * crops_per_epoch // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M params, {steps} steps", flush=True)
    best = 1e9
    step = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        items = []
        for c in train:
            n = len(c["times"])
            for _ in range(max(1, n // CROP)):
                items.append((c, random.randint(0, max(0, n - CROP))))
        random.shuffle(items)
        t0 = time.time()
        run = np.zeros(3)
        for b in range(0, len(items) - args.batch + 1, args.batch):
            if step >= steps:
                break
            mel, o, stars = make_batch(corpus, items[b:b + args.batch], CROP)
            stars = stars + torch.randn_like(stars) * 0.05          # star labels are noisy anyway
            lc, lh, ll = losses(forward(net, mel, o, stars), o)
            loss = lc + 0.3 * lh + 0.2 * ll
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run += [lc.item(), lh.item(), ll.item()]
        nb = max(1, len(items) // args.batch)
        ev = evaluate(net, corpus, "val")
        print(f"epoch {epoch:2d} train {run[0] / nb:.4f}/{run[1] / nb:.3f}/{run[2] / nb:.3f} "
              f"val {ev['count']:.4f}/{ev['lnhead']:.3f}/{ev['lnlen']:.3f} "
              f"P {ev['precision']:.3f} R {ev['recall']:.3f} F1 {ev['f1']:.3f}  {time.time() - t0:.0f}s", flush=True)
        score = ev["count"] + 0.3 * ev["lnhead"] + 0.2 * ev["lnlen"]
        if score < best:
            best = score
            torch.save({"model": net.state_dict(), "config": net.config, "epoch": epoch,
                        "val": ev}, args.out)
    print("best val", best)


if __name__ == "__main__":
    main()
