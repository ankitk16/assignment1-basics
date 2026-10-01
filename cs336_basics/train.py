"""Training loop for CS336 Assignment 1.
 
Run:
    python -m cs336_basics.train \
        --train-data artifacts/tinystories_train.npy \
        --val-data   artifacts/tinystories_valid.npy \
        --run-name   ts-baseline
 
Resume:
    python -m cs336_basics.train ... --resume checkpoints/ts-baseline/latest.pt
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cs336_basics.nn_modules import (
    AdamW,
    Transformer,
    cross_entropy,
    get_batch,
    gradient_clipping,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
)

# --- adjust these imports to match your module layout -------------------------

# ------------------------------------------------------------------------------


def get_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train-data", type=str, required=True)
    p.add_argument("--val-data", type=str, required=True)

    # model
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10_000.0)

    # optimizer / schedule
    p.add_argument("--lr-max", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--cosine-iters", type=int, default=5_000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=5_000)
    p.add_argument("--device", type=str, default=None, help="cpu | mps | cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true", help="torch.compile (cuda only)")

    # logging / checkpointing
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=1_000)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)

    return p.parse_args(argv)


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def estimate_loss(model: nn.Module, data, args, device: torch.device):
    """Average loss over several batches. Cheap proxy for full-set evaluation."""
    model.eval()  # set model in eval mode
    losses = []
    for _ in range(args.eval_batches):
        x, y = get_batch(data, args.batch_size, args.context_length, device)
        logits = model(x)
        losses.append(cross_entropy(logits, y).item())
    model.train()
    return float(np.mean(losses))


def main(argv=None):

    # this is when we have GPU
    torch.set_float32_matmul_precision("high")

    args = get_args(argv)
    if args.cosine_iters is None:
        args.cosine_iters = args.max_iters

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    run_name = args.run_name or time.strftime("run-%Y%m%d-%H%M%S")
    ckpt_dir = Path(args.ckpt_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  run={run_name}  ckpt_dir={ckpt_dir}")

    # ----data: mmap never fully loaded-------------------
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")
    print(f"train tokens: {len(train_data):,}   val tokens: {len(val_data):,}")

    # -------model / optimizer -------

    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.1f}M")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    if args.compile and device.type == "cude":
        model = torch.compile(model)

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed from {args.resume} at iteration {start_iter}")

    # -----log experiment results--------
    log_path = ckpt_dir / "log.josnl"
    log_f = open(log_path, "a")  # noqa: SIM115

    def log(**kw):
        """
        **kw collects all keyword arguments passed to log() into a single dict named kw
        """
        log_f.write(json.dumps(kw) + "\n")
        log_f.flush()

    log(event="config", **vars(args), n_params=n_params, device=str(device))

    # ---------------training loop------------------------------
    model.train()
    t0 = time.perf_counter()
    tokens_per_iter = args.batch_size * args.context_length

    for it in range(start_iter, args.max_iters):
        lr = learning_rate_schedule(
            it, args.lr_max, args.lr_min, args.warmup_iters, args.cosine_iters
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # if A100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = cross_entropy(logits, y)

        # if loss goes to infinite, stop
        if not math.isfinite(loss.item()):
            log(event="diverged", iter=it, lr=lr)
            print(f"diverged at iter {it}, lr {lr:.2e}")
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if it % args.log_every == 0:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            steps = max(1, it - start_iter + 1)
            tok_sec = tokens_per_iter * steps / dt
            print(
                f"iter {it:>6}  loss {loss.item():.4f}  lr {lr:.2e}  "
                f"{tok_sec / 1e3:.1f}k tok/s  {dt / 60:.1f} min"
            )

            # logging for experiments
            log(
                event="train",
                iter=it,
                loss=loss.item(),
                lr=lr,
                wallclock=dt,
                tokens=tokens_per_iter * (it + 1),
            )

        if it > 0 and it % args.ckpt_every == 0:
            save_checkpoint(model, optimizer, it, ckpt_dir / "latest.pt")

        # ----------------- final ---------------
        val_loss = estimate_loss(model, val_data, args, device)
        print(f"final VAL loss {val_loss:.4f}  ppl {math.exp(val_loss):.1f}")
        save_checkpoint(model, optimizer, args.max_iters, ckpt_dir / "final.pt")
        log(event="val", iter=it, val_loss=val_loss, wallclock=time.perf_counter() - t0)


if __name__ == "__main__":
    main()
