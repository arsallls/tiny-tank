"""Checkpoint -> flat float32 binary + config + a parity fixture for the browser.

Writes three files into docs/:
  weights.bin   every tensor concatenated, float32, little-endian
  config.json   model dims, vocab, and a manifest of {name, shape, offset}
  parity.json   a fixed input and the logits PyTorch produced for it

The manifest means model.js never hardcodes the layout, and parity.json means
the JS port is checked against PyTorch in the browser itself - a transposed
RoPE cannot slip through by passing in one toolchain and failing in the other.

  python3 export.py --ckpt out/m/tank.pt
"""
import argparse, json, os

import numpy as np
import torch

from model import GPT, GPTConfig


def tensors(model, cfg):
    """(name, tensor) in a fixed order. head.weight is tied to tok.weight."""
    out = [("tok.weight", model.tok.weight)]
    for i, b in enumerate(model.blocks):
        out += [(f"h{i}.n1.weight", b.n1.weight),
                (f"h{i}.attn.qkv.weight", b.attn.qkv.weight),
                (f"h{i}.attn.proj.weight", b.attn.proj.weight),
                (f"h{i}.n2.weight", b.n2.weight),
                (f"h{i}.mlp.gate.weight", b.mlp.gate.weight),
                (f"h{i}.mlp.up.weight", b.mlp.up.weight),
                (f"h{i}.mlp.down.weight", b.mlp.down.weight)]
    out.append(("norm.weight", model.norm.weight))
    return out


def main(a):
    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).eval()
    model.load_state_dict(ck["model"])
    vocab = ck["vocab"]
    os.makedirs(a.out, exist_ok=True)

    manifest, blobs, off = [], [], 0
    for name, t in tensors(model, cfg):
        arr = t.detach().contiguous().float().numpy().astype("<f4")
        manifest.append({"name": name, "shape": list(arr.shape), "offset": off})
        blobs.append(arr.ravel())
        off += arr.size
    np.concatenate(blobs).tofile(os.path.join(a.out, "weights.bin"))

    hidden = model.blocks[0].mlp.gate.weight.shape[0]
    cfgjs = {"block_size": cfg.block_size, "vocab_size": cfg.vocab_size,
             "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
             "head_dim": cfg.n_embd // cfg.n_head, "hidden": hidden,
             "rope_base": 10000.0, "norm_eps": 1e-6,
             "n_params": off, "vocab": vocab, "tensors": manifest}
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfgjs, f)

    # logits at EVERY position, not just the last: a RoPE bug is invisible at
    # position 0 and grows with index, so one trailing check can miss it
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, a.parity_len), generator=g)
    with torch.no_grad():
        x = model.tok(ids)
        for blk in model.blocks:
            x, _ = blk(x, model.cos, model.sin, None)
        logits = model.head(model.norm(x))[0]
    with open(os.path.join(a.out, "parity.json"), "w") as f:
        json.dump({"ids": ids[0].tolist(),
                   "logits": [round(v, 5) for v in logits.flatten().tolist()]}, f)

    mb = off * 4 / 1e6
    print(f"{len(manifest)} tensors, {off:,} params -> {mb:.2f} MB")
    print(f"wrote {a.out}/weights.bin  {a.out}/config.json  {a.out}/parity.json")
    print(f"cfg: {cfg.n_layer}L {cfg.n_embd}d {cfg.n_head}h  hidden {hidden}  "
          f"block {cfg.block_size}  vocab {cfg.vocab_size}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/m/tank.pt")
    p.add_argument("--out", default="docs")
    p.add_argument("--parity-len", type=int, default=64)
    main(p.parse_args())
