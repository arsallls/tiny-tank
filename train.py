"""Train the fish. Single stage - the whole run is minutes on any GPU.

  python3 train.py                      # defaults
  python3 train.py --max-iters 4000     # longer
"""
import argparse, math, os, pickle, time

import numpy as np
import torch

from model import GPT, GPTConfig


def get_batch(data, split, bs, block, device):
    d = data[split]
    ix = torch.randint(len(d) - block - 1, (bs,))
    x = torch.stack([torch.from_numpy(d[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(it, a, warmup):
    if it < warmup:
        return a.lr * (it + 1) / warmup
    r = (it - warmup) / max(1, a.max_iters - warmup)
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * r))


@torch.no_grad()
def estimate(model, data, a, device, ctx, batches=40):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(batches)
        for i in range(batches):
            x, y = get_batch(data, split, a.batch_size, a.block_size, device)
            with ctx:
                _, loss = model(x, targets=y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main(a):
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    with open(os.path.join(a.data, "meta.pkl"), "rb") as f:
        vocab = pickle.load(f)["vocab"]
    data = {s: np.memmap(os.path.join(a.data, f"{s}.bin"), dtype=np.uint16, mode="r")
            for s in ("train", "val")}
    print(f"vocab {len(vocab)}  train {len(data['train']):,}  val {len(data['val']):,}")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        # T4 (sm_75) has no bf16 -> fp16 + GradScaler; L4/A100 -> bf16, no scaler
        use_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        ctx = torch.amp.autocast("cuda", dtype=dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
        print(f"{torch.cuda.get_device_name(0)}  {dtype}")
    else:
        import contextlib
        ctx, scaler = contextlib.nullcontext(), torch.amp.GradScaler(enabled=False)
        print("cpu")

    cfg = GPTConfig(block_size=a.block_size, vocab_size=len(vocab),
                    n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd)
    model = GPT(cfg).to(device)
    print(f"params {model.num_params() / 1e6:.2f}M")
    opt = model.configure_optimizers(0.1, a.lr, (0.9, 0.95), device)

    raw = model
    if a.compile and device == "cuda":
        model = torch.compile(model)

    warmup = min(a.warmup, max(1, a.max_iters // 10))
    best, stale, t0 = float("inf"), 0, time.time()
    for it in range(a.max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, a, warmup)

        if it % a.eval_interval == 0:
            m = estimate(model, data, a, device, ctx)
            el = time.time() - t0
            # losses are nats (cross_entropy); eval.py reports bits
            print(f"eval {it}: train {m['train']:.4f} | val {m['val']:.4f} nats "
                  f"| {el:.0f}s | {el / max(1, it):.3f}s/iter", flush=True)
            if m["val"] < best:
                best, stale = m["val"], 0
                # vars(cfg), not the dataclass: torch.load defaults to
                # weights_only=True since 2.6 and refuses a custom class
                torch.save({"cfg": vars(cfg), "model": raw.state_dict(), "iter": it,
                            "val": best, "vocab": vocab},
                           os.path.join(a.out_dir, "tank.pt"))
            else:
                stale += 1
                if stale >= a.patience:
                    print(f"early stop at {it} (best val {best:.4f})")
                    break

        x, y = get_batch(data, "train", a.batch_size, a.block_size, device)
        with ctx:
            _, loss = model(x, targets=y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    print(f"done: best val {best:.4f} nats = {best / math.log(2):.4f} bits/token"
          f"  ({time.time() - t0:.0f}s)")
    print(f"saved {a.out_dir}/tank.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out-dir", default="out")
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    main(p.parse_args())
