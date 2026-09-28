"""N-gram baselines + the two behavioral probes.

Run this BEFORE training anything. If a 5-gram already tracks the habituation
and hunger curves, the simulator has no long-range structure and a transformer
is pointless - fix sim.py, not the model.

Every model is a function (ctx, pos) -> probs(32,) - pos lets an array-backed
oracle index itself; count-based models ignore it - so the
transformer plugs into the same probes later.

  python3 eval.py                      # n-gram baselines
  python3 eval.py --ckpt out/tank.pt   # + the transformer
"""
import argparse, os, pickle, time
from collections import defaultdict

import numpy as np

FEAR = ["HIDE", "STARTLED", "SPLASH", "DART_D"]
TAP_WINDOW = 300          # must match sim.py


def load(d):
    with open(os.path.join(d, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    vocab = meta["vocab"]
    data = {s: np.fromfile(os.path.join(d, f"{s}.bin"), dtype=np.uint16)
            for s in ("train", "val")}
    return vocab, {t: i for i, t in enumerate(vocab)}, data


class NGram:
    """Interpolated n-gram. p_k(x|ctx) = (c[ctx][x] + a*p_{k-1}(x)) / (n[ctx] + a).

    Contexts are packed into an int (5 bits/token, vocab is 32) so the tables
    are plain dicts instead of a 32^k dense array.
    """

    def __init__(self, order, vocab_size, alpha=2.0):
        self.order, self.V, self.alpha = order, vocab_size, alpha
        self.tab = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        self.tot = [defaultdict(int) for _ in range(order)]

    def fit(self, data):
        d = data.astype(np.int64)
        for k in range(self.order):
            if k == 0:
                self.uni = np.bincount(d, minlength=self.V).astype(np.float64)
                self.uni /= self.uni.sum()
                continue
            # rolling base-32 key over the k tokens before each position
            key = np.zeros(len(d) - k, dtype=np.int64)
            for j in range(k):
                key = (key << 5) | d[j:len(d) - k + j]
            nxt = d[k:]
            for c, x in zip(key.tolist(), nxt.tolist()):
                self.tab[k][c][x] += 1
                self.tot[k][c] += 1
        return self

    def predict(self, ctx):
        p = self.uni
        for k in range(1, self.order):
            if len(ctx) < k:
                break
            c = 0
            for t in ctx[-k:]:
                c = (c << 5) | int(t)
            n = self.tot[k].get(c, 0)
            if not n:
                break
            row = np.zeros(self.V)
            for x, v in self.tab[k][c].items():
                row[x] = v
            p = (row + self.alpha * p) / (n + self.alpha)
        return p


def nll(predict, data, order, n=20000, seed=0):
    """Mean negative log-likelihood in bits/token over random val positions."""
    rng = np.random.default_rng(seed)
    lo = max(order, 512)
    pos = rng.integers(lo, len(data) - 1, size=n)
    tot = 0.0
    for i in pos:
        p = predict(data[i - lo:i].tolist(), int(i))
        tot -= np.log2(max(p[int(data[i])], 1e-12))
    return tot / n


def probe_habituation(predict, data, stoi, itos_fear, n_per=400, seed=0):
    """P(fear response | the context ends in <TAP>), bucketed by how many taps
    already landed in the trailing window. Real val contexts, no synthetics."""
    tap = stoi["<TAP>"]
    hits = np.flatnonzero(data[512:] == tap) + 512
    rng = np.random.default_rng(seed)
    rng.shuffle(hits)
    buckets = defaultdict(list)
    for i in hits:
        w = data[max(0, i - TAP_WINDOW):i]
        k = int((w == tap).sum()) + 1
        b = 1 if k <= 1 else (2 if k <= 3 else (3 if k <= 8 else (4 if k <= 15 else 5)))
        if len(buckets[b]) >= n_per:
            continue
        p = predict(data[i - 512:i + 1].tolist(), int(i) + 1)
        buckets[b].append(float(p[itos_fear].sum()))
        if all(len(buckets[j]) >= n_per for j in range(1, 6)):
            break
    return {b: float(np.mean(v)) for b, v in sorted(buckets.items()) if v}


def probe_hunger(predict, data, stoi, n_per=400, seed=0):
    """P(CHOMP | the context ends in <FOOD>), bucketed by ticks since the last
    CHOMP - the observable proxy for hunger. A hungry fish commits to the bite."""
    food, chomp = stoi["<FOOD>"], stoi["CHOMP"]
    hits = np.flatnonzero(data[512:] == food) + 512
    rng = np.random.default_rng(seed)
    rng.shuffle(hits)
    buckets = defaultdict(list)
    for i in hits:
        w = data[max(0, i - 800):i]
        last = np.flatnonzero(w == chomp)
        gap = 999 if len(last) == 0 else len(w) - 1 - int(last[-1])
        b = 1 if gap < 100 else (2 if gap < 250 else (3 if gap < 500 else 4))
        if len(buckets[b]) >= n_per:
            continue
        p = predict(data[i - 512:i + 1].tolist(), int(i) + 1)
        buckets[b].append(float(p[chomp]))
        if all(len(buckets[j]) >= n_per for j in range(1, 5)):
            break
    return {b: float(np.mean(v)) for b, v in sorted(buckets.items()) if v}


HAB_LBL = {1: "1 tap", 2: "2-3", 3: "4-8", 4: "9-15", 5: "16+"}
HUN_LBL = {1: "<100", 2: "100-250", 3: "250-500", 4: "500+"}


def report(name, predict, data, stoi, fear_ids, order):
    b = nll(predict, data["val"], order)
    h = probe_habituation(predict, data["val"], stoi, fear_ids)
    g = probe_hunger(predict, data["val"], stoi)
    print(f"\n{name}")
    print(f"  val  {b:.4f} bits/token")
    print("  habituation  " + "  ".join(f"{HAB_LBL[k]}={v:.3f}" for k, v in h.items()))
    print("  hunger       " + "  ".join(f"{HUN_LBL[k]}={v:.3f}" for k, v in g.items()))
    return {"nll": b, "hab": h, "hunger": g}


def main(a):
    vocab, stoi, data = load(a.data)
    fear_ids = [stoi[t] for t in FEAR]
    print(f"vocab {len(vocab)}  train {len(data['train']):,}  val {len(data['val']):,}")

    out = {}
    for order in a.orders:
        t0 = time.time()
        m = NGram(order, len(vocab)).fit(data["train"])
        print(f"fit {order}-gram in {time.time() - t0:.0f}s", flush=True)
        out[f"{order}-gram"] = report(f"{order}-gram", lambda c, _p, m=m: m.predict(c),
                                     data, stoi, fear_ids, order)

    if a.ckpt:
        import torch
        from model import GPT, GPTConfig
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(a.ckpt, map_location=dev)
        m = GPT(GPTConfig(**ck["cfg"])).to(dev).eval()
        m.load_state_dict(ck["model"])

        @torch.no_grad()
        def predict(ctx, _pos=None):
            x = torch.tensor(ctx[-m.cfg.block_size:], device=dev)[None]
            return torch.softmax(m(x)[0][0, -1].float(), -1).cpu().numpy()

        out["transformer"] = report("transformer", predict, data, stoi, fear_ids, 512)

    with open(os.path.join(a.data, "eval.pkl"), "wb") as f:
        pickle.dump(out, f)
    print(f"\nwrote {a.data}/eval.pkl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--ckpt")
    p.add_argument("--orders", type=int, nargs="+", default=[2, 3, 5])
    main(p.parse_args())
