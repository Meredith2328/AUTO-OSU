"""Train the tick rhythm model (autoregressive or masked) with optional Weights & Biases logging.

Single GPU:
    python -m autoosu.ml.train --mode ar --prep data/prep --out runs/ar_v0 --wandb autoosu
Multi GPU (DDP, one process per card):
    torchrun --standalone --nnodes=1 --nproc-per-node=2 -m autoosu.ml.train --mode ar ...

Every eval reports teacher-forced metrics and, more importantly, *generation* metrics: a few
validation sequences are generated from scratch and compared with the human map (onset F1,
density error, rhythm-gap mix) so the two paradigms can be compared on equal terms.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import GRID, N_CLASSES, TickDataset, load_index, object_starts
from .model import ModelConfig, TickTransformer, ar_inputs, masked_inputs, random_mask, sample_ar, sample_masked

CLASS_NAMES = ["none", "circle", "head", "body", "end", "spin"]
GAP_EDGES = [0.375, 0.75, 1.5, 2.5, np.inf]      # 1/4, 1/2, 1/1, 2, 3+ beats


def setup_distributed() -> Tuple[int, int, int]:
    """(rank, world_size, local_rank); plain single-process run when not launched by torchrun."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0


def is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def gap_hist(labels: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(object_starts(labels))
    if len(starts) < 2:
        return np.ones(len(GAP_EDGES)) / len(GAP_EDGES)
    gaps = np.diff(starts) / GRID
    h = np.array([np.sum((gaps <= e) & (gaps > (GAP_EDGES[i - 1] if i else 0))) for i, e in enumerate(GAP_EDGES)], float)
    return (h + 0.5) / (h.sum() + 0.5 * len(h))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare_sequences(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    ps, ts = object_starts(pred).astype(bool), object_starts(true).astype(bool)
    hit = int((ps & ts).sum())
    prec = hit / max(int(ps.sum()), 1)
    rec = hit / max(int(ts.sum()), 1)
    measures = max(len(true) / (4 * GRID), 1e-6)
    return {
        "gen_onset_f1": 2 * prec * rec / max(prec + rec, 1e-9), "gen_onset_precision": prec, "gen_onset_recall": rec,
        "gen_density_true": float(ts.sum()) / measures, "gen_density_pred": float(ps.sum()) / measures,
        "gen_gap_js": js_divergence(gap_hist(pred), gap_hist(true)),
        "gen_slider_share_true": float(np.mean(true[ts] == 2)) if ts.any() else 0.0,
        "gen_slider_share_pred": float(np.mean(pred[ps] == 2)) if ps.any() else 0.0,
    }


def model_loss(model, batch: dict, mode: str, device: str):
    labels = batch["labels"]
    if mode == "ar":
        tokens = ar_inputs(labels, batch["prev_label"])
        target = labels
    else:
        mask = random_mask(labels)
        tokens = masked_inputs(labels, mask)
        target = torch.where(mask, labels, torch.full_like(labels, -100))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=is_cuda(device)):
        logits = model(batch["audio"], tokens, batch["metrical"], batch["extra"], batch["cond"])
    loss = F.cross_entropy(logits.float().flatten(0, 1), target.flatten(), ignore_index=-100, label_smoothing=0.05)
    return loss, logits, target


@torch.no_grad()
def evaluate(model: TickTransformer, loader: DataLoader, device: str, mode: str, max_batches: int = 40,
             gen_sequences: int = 6, gen_len: int = 512, steps: int = 12) -> Dict:
    model.eval()
    tot_loss = tot_n = 0.0
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    gen_stats: List[Dict[str, float]] = []
    examples = []
    gen_rng = torch.Generator(device=device).manual_seed(0)
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        loss, logits, target = model_loss(model, batch, mode, device)
        valid = target != -100
        tot_loss += float(loss) * int(valid.sum())
        tot_n += int(valid.sum())
        pred = logits.argmax(-1)[valid].cpu().numpy()
        np.add.at(conf, (target[valid].cpu().numpy(), pred), 1)
        if len(gen_stats) < gen_sequences:
            j = 0
            true = batch["labels"][j].cpu().numpy()
            n = int((true != -100).sum())
            if n < 64:
                continue
            n = min(n, gen_len)
            args = (batch["audio"][j, :n], batch["metrical"][j, :n], batch["extra"][j, :n], batch["cond"][j])
            if mode == "ar":
                gen = sample_ar(model, *args, temperature=0.9, generator=gen_rng, prev0=int(batch["prev_label"][j]))
            else:
                gen = sample_masked(model, *args, steps=steps, temperature=0.9, generator=gen_rng)
            gen_stats.append(compare_sequences(gen, true[:n]))
            if len(examples) < 2:
                examples.append((true[:n], gen))
    model.train()
    tp = np.diag(conf).astype(float)
    prec = tp / np.maximum(conf.sum(0), 1)
    rec = tp / np.maximum(conf.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    out = {"val_loss": tot_loss / max(tot_n, 1), "acc": float(tp.sum() / max(conf.sum(), 1)),
           "f1": {n: round(float(v), 3) for n, v in zip(CLASS_NAMES, f1)}}
    if gen_stats:
        for k in gen_stats[0]:
            out[k] = float(np.mean([g[k] for g in gen_stats]))
        out["gen_density_mae"] = float(np.mean([abs(g["gen_density_pred"] - g["gen_density_true"]) for g in gen_stats]))
    out["_examples"] = examples
    return out


def render_examples(examples, path: Path, title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(examples), 1, figsize=(16, 2.2 * len(examples)), squeeze=False)
    colors = {1: "#e34948", 2: "#1baf7a", 3: "#9ed8c0", 4: "#1baf7a", 5: "#4a3aa7"}
    for ax, (true, gen) in zip(axes[:, 0], examples):
        for row, seq, name in ((1.0, true, "human"), (0.0, gen, "model")):
            for c, col in colors.items():
                xs = np.flatnonzero(seq == c)
                ax.scatter(xs, np.full(len(xs), row), s=14 if c in (1, 2, 5) else 5, color=col)
            ax.text(-2, row, name, ha="right", va="center", fontsize=8)
        for m in range(0, len(true), 4 * GRID):
            ax.axvline(m, color="#e6e5e1", lw=0.6)
        ax.set_ylim(-0.6, 1.6); ax.set_yticks([]); ax.set_xlim(-8, len(true))
    axes[0, 0].set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ar", "masked"], default="ar")
    ap.add_argument("--prep", default="data/prep")
    ap.add_argument("--out", default="runs/rhythm_v0")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64, help="per-GPU batch size")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--audio-ctx-layers", type=int, default=0,
                    help="bidirectional label-free layers over the audio first (lets the ar decoder hear ahead)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--gen-sequences", type=int, default=6)
    ap.add_argument("--decode-steps", type=int, default=12, help="masked mode: decoding rounds in generation eval")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", default="", help="W&B project name (set WANDB_API_KEY / WANDB_ENTITY in the env)")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    main_proc = rank == 0
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = None
    if args.wandb and main_proc:
        import wandb

        run = wandb.init(project=args.wandb, name=args.run_name or f"{args.mode}-{out.name}",
                         config={**vars(args), "world_size": world, "effective_batch": args.batch * world})

    train_rows, val_rows = load_index(args.prep)
    if main_proc:
        print(f"train maps {len(train_rows)}  val maps {len(val_rows)}  device {device}  mode {args.mode}  "
              f"world {world}  effective batch {args.batch * world}")
    train_ds = TickDataset(train_rows, args.prep, seq_len=args.seq, train=True)
    val_ds = TickDataset(val_rows, args.prep, seq_len=args.seq, train=True, density_dropout=0.0)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=sampler is None, sampler=sampler,
                          num_workers=args.workers, drop_last=True, pin_memory=is_cuda(device),
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=min(2, args.workers),
                        pin_memory=is_cuda(device))

    cfg = ModelConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, dropout=args.dropout,
                      max_len=max(2048, args.seq), mode=args.mode, audio_ctx_layers=args.audio_ctx_layers)
    model = TickTransformer(cfg).to(device)
    if main_proc:
        print(f"model params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        state = dict(ck["state_dict"])
        if "dec_pos_emb.weight" not in state:      # v0 checkpoints predate the two-stage forward
            state["dec_pos_emb.weight"] = torch.zeros_like(model.dec_pos_emb.weight)
        model.load_state_dict(state)
        step = ck.get("step", 0)
    train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    fwd_model = torch.compile(train_model) if args.compile else train_model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)

    def lr_at(s: int) -> float:
        if s < args.warmup:
            return args.lr * s / max(1, args.warmup)
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    log = (out / "log.jsonl").open("a", encoding="utf-8") if main_proc else None
    best = float("inf")
    if args.resume and (out / "best.pt").exists():
        # keep the earlier best checkpoint unless the resumed run beats it
        prev = torch.load(out / "best.pt", map_location="cpu", weights_only=False).get("eval", {})
        if "gen_onset_f1" in prev:
            best = -float(prev["gen_onset_f1"])
            if main_proc:
                print(f"resuming: previous best gen onset F1 {-best:.4f} at step {prev.get('step')}")
    t0 = time.perf_counter()
    train_model.train()
    run_loss, run_n = 0.0, 0
    epoch = 0
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_dl:
            if step >= args.steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, _, _ = model_loss(fwd_model, batch, args.mode, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            run_loss += float(loss.detach()); run_n += 1
            if step % 50 == 0 and main_proc:
                msg = {"step": step, "loss": round(run_loss / run_n, 4), "lr": lr_at(step),
                       "elapsed_s": round(time.perf_counter() - t0), "epoch": epoch}
                print(msg, flush=True)
                log.write(json.dumps(msg) + "\n"); log.flush()
                if run:
                    run.log(msg, step=step)
                run_loss, run_n = 0.0, 0
            if step % args.eval_every == 0 or step == args.steps:
                if main_proc:
                    ev = evaluate(model, val_dl, device, args.mode, gen_sequences=args.gen_sequences, steps=args.decode_steps)
                    examples = ev.pop("_examples")
                    ev["step"] = step
                    print("EVAL", json.dumps(ev), flush=True)
                    log.write(json.dumps(ev) + "\n"); log.flush()
                    model.save(str(out / "last.pt"), {"step": step, "args": vars(args)})
                    score = ev.get("gen_onset_f1", -ev["val_loss"])
                    if -score < best:
                        best = -score
                        model.save(str(out / "best.pt"), {"step": step, "args": vars(args), "eval": ev})
                    scalars = {k: v for k, v in ev.items() if not isinstance(v, dict)}
                    scalars.update({f"f1/{k}": v for k, v in ev["f1"].items()})
                    if examples:
                        png = render_examples(examples, out / "samples" / f"step{step:07d}.png",
                                              f"{args.mode} step {step}  onset F1 {ev.get('gen_onset_f1', 0):.3f}")
                        if run:
                            import wandb
                            scalars["samples"] = wandb.Image(str(png))
                    if run:
                        run.log(scalars, step=step)
                    train_model.train()
                if world > 1:
                    dist.barrier()
    if main_proc:
        print(f"done in {(time.perf_counter() - t0) / 60:.1f} min, best gen onset F1 {-best:.4f} -> {out / 'best.pt'}")
        if run:
            run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
