"""Train the 4K lane-pattern model (autoosu.mania4k.patterns.PatternNet) on ranked charts.

    python scripts/mania4k_train_patterns.py --data ~/data/prepared --out models/mania4k_patterns.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoosu.mania4k.chart import RedLine, red_line_at                          # noqa: E402
from autoosu.mania4k.patterns import (POPCOUNT, PatternNet, RowState, advance,   # noqa: E402
                                      row_features)
from autoosu.mania4k.structure import ARCHETYPES, TYPES, chart_archetype, chart_profile  # noqa: E402


def chart_rows(notes: np.ndarray, reds, stars: float, mirror: bool):
    """Features and mask labels of every row of one chart, conditioned on the human chart's own
    window types (the section style) and archetype."""
    prof = chart_profile(notes, reds)
    arch = ARCHETYPES.index(chart_archetype(prof["types"]))
    w_start = np.array([a for a, _ in prof["windows"]]) if prof["windows"] else np.zeros(1)
    w_type = [TYPES.index(t) for t in prof["types"]] or [1]
    t = np.round(notes[:, 0], 1)
    lanes = notes[:, 1].astype(int)
    if mirror:
        lanes = 3 - lanes
    ends = notes[:, 2]
    X, Y = [], []
    st = RowState.fresh()
    order = np.argsort(t, kind="stable")
    t, lanes, ends = t[order], lanes[order], ends[order]
    i = 0
    while i < len(t):
        j = i
        while j < len(t) and t[j] == t[i]:
            j += 1
        mask, ln_end = 0, [0.0] * 4
        for l, e in zip(lanes[i:j], ends[i:j]):
            mask |= 1 << l
            if e > t[i]:
                ln_end[l] = e
        k = bin(mask).count("1")
        wi = int(np.clip(np.searchsorted(w_start, t[i], side="right") - 1, 0, len(w_type) - 1))
        X.append(row_features(st, float(t[i]), k, any(ln_end), red_line_at(reds, t[i]).beat_ms, stars,
                              w_type[wi], arch))
        Y.append(mask)
        advance(st, float(t[i]), mask, ln_end)
        i = j
    return X, Y


def load(data: Path):
    sets = {"train": ([], []), "val": ([], [])}
    for f in sorted(data.glob("*.npz")):
        z = np.load(f)
        meta = json.loads(bytes(z["meta"]))
        split = "train" if meta["split"] == "train" else ("val" if meta["split"] == "val" else None)
        if split is None:
            continue
        for c in meta["charts"]:
            reds = [RedLine(*r) for r in c["reds"]]
            for mirror in ((False, True) if split == "train" else (False,)):
                X, Y = chart_rows(z[f"{c['key']}_notes"], reds, c["stars"], mirror)
                sets[split][0].extend(X)
                sets[split][1].extend(Y)
    return {k: (torch.from_numpy(np.array(x, np.float32)), torch.tensor(y)) for k, (x, y) in sets.items()}


def masked_logits(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    k = x[:, :4].argmax(1) + 1
    pc = torch.from_numpy(POPCOUNT)[None, :].to(logits.device)
    return logits.masked_fill(pc != k[:, None], -1e9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MANIA4K_PREPARED", str(Path.home() / "data" / "prepared")))
    ap.add_argument("--out", default="models/mania4k_patterns.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    t0 = time.time()
    d = load(Path(args.data))
    Xtr, Ytr = d["train"]
    Xva, Yva = d["val"]
    print(f"rows train {len(Ytr)} val {len(Yva)} ({time.time() - t0:.0f}s)", flush=True)
    # reference: frequency of each mask given chord size only
    net = PatternNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * (len(Ytr) // 2048)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    best = 1e9
    for ep in range(args.epochs):
        perm = torch.randperm(len(Ytr))
        net.train()
        tot = 0.0
        for b in range(0, len(perm) - 2047, 2048):
            idx = perm[b:b + 2048]
            loss = F.cross_entropy(masked_logits(net(Xtr[idx]), Xtr[idx]), Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        net.eval()
        with torch.no_grad():
            lv = masked_logits(net(Xva), Xva)
            nll = F.cross_entropy(lv, Yva).item()
            acc = (lv.argmax(1) == Yva).float().mean().item()
            k = Xva[:, :4].argmax(1) + 1
            single = k == 1
            acc1 = (lv.argmax(1) == Yva)[single].float().mean().item()
        print(f"epoch {ep} train {tot / max(1, len(perm) // 2048):.4f} val nll {nll:.4f} acc {acc:.3f} "
              f"(single-note rows {acc1:.3f}, chance 0.25)", flush=True)
        if nll < best:
            best = nll
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": net.state_dict(), "config": net.config, "val_nll": nll, "val_acc": acc}, args.out)


if __name__ == "__main__":
    main()
