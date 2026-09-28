"""Fish behavior simulator -> token sequences, plus its own oracle.

The fish has hidden state (hunger, startle, habituation, boredom, circadian)
that evolves over hundreds of ticks. The token stream never states that state
directly - moods fire only on the tick they CHANGE - so recovering it needs
long-range context. That gap is the point: an n-gram cannot do it.

dist() returns the exact next-token distribution and step() samples from it,
so the oracle is the generating process itself and cannot drift from the
behavior the way a hand-written copy of the policy would.

  python3 sim.py --inspect        # read ticks, with state annotations
  python3 sim.py --oracle         # irreducible entropy + oracle probe curves
  python3 sim.py                  # write data/{train,val}.bin + meta.pkl
"""
import argparse, os, pickle
from collections import deque

import numpy as np

STIMULI = ["<FOOD>", "<TAP>", "<HAND>", "<LIGHT_ON>", "<LIGHT_OFF>"]
MOTION = ["DART_L", "DART_R", "DART_U", "DART_D",
          "DRIFT_L", "DRIFT_R", "DRIFT_U", "DRIFT_D",
          "HOVER", "SINK", "TURN", "CIRCLE"]
ACTIONS = ["CHOMP", "NIBBLE", "SCAN", "ORIENT", "HIDE", "SURFACE"]
SOUNDS = ["BUBBLE", "GULP", "SPLASH"]
MOODS = ["HUNGRY", "CONTENT", "STARTLED", "BORED", "SLEEPY"]
VOCAB = ["<BOS>"] + STIMULI + MOTION + ACTIONS + SOUNDS + MOODS
STOI = {t: i for i, t in enumerate(VOCAB)}
V = len(VOCAB)

TAP_WINDOW = 300        # ticks a tap still counts toward habituation
DAY = 4000              # circadian period
P_STIM = 0.007          # per-tick chance any stimulus arrives


class Tank:
    def __init__(self, rng):
        self.rng = rng
        self.hunger = rng.random() * 0.3
        self.startle = 0.0
        self.boredom = 0
        self.light = True
        self.food = 0                        # ticks of food still floating
        self.burst = 0                       # remaining ticks of a tap burst
        self.taps = deque()
        self.t = int(rng.integers(0, DAY))   # start anywhere in the cycle
        self.mood = None

    @property
    def fear(self):
        # habituation: each recent tap makes the next one land less.
        # 0.15 not 0.45 - a steeper decay saturates to zero fear by the 4th
        # tap, which an n-gram matches by just detecting a recent tap. The
        # graded curve is what forces the model to actually count.
        return self.startle * float(np.exp(-0.15 * len(self.taps)))

    def sleepy(self):
        return np.sin(2 * np.pi * self.t / DAY) < -0.7

    def current_mood(self):
        if self.fear > 0.50:
            return "STARTLED"
        if not self.light or self.sleepy():
            return "SLEEPY"
        if self.hunger > 0.6:
            return "HUNGRY"
        if self.boredom > 250:
            return "BORED"
        if self.hunger < 0.15:
            return "CONTENT"
        return None

    FEAR_MENU = {"HIDE": 4, "DART_L": 2, "DART_R": 2, "DART_D": 3,
                 "SINK": 1, "SPLASH": 0.5}
    IDLE_MENU = {"DRIFT_L": 3, "DRIFT_R": 3, "DRIFT_U": 1.5, "DRIFT_D": 1.5,
                 "HOVER": 3, "TURN": 1.5, "BUBBLE": 0.8, "CIRCLE": 0.7}

    def menu(self):
        """Weights over behavior tokens for the un-startled fish."""
        if self.food:
            # three tiers, because a full fish that ignores food entirely reads
            # as a broken demo: clicking Feed must always do something visible
            if self.hunger > 0.25:
                return {"ORIENT": 2, "DART_U": 3, "DART_L": 1, "DART_R": 1,
                        "CHOMP": 4, "NIBBLE": 2, "GULP": 1.5}
            if self.hunger > 0.08:
                return {"ORIENT": 3, "NIBBLE": 3, "DRIFT_U": 2,
                        "CHOMP": 1, "TURN": 1, "BUBBLE": 0.5}
            return {**self.IDLE_MENU, "ORIENT": 2, "DRIFT_U": 2.5, "TURN": 2.5}
        if self.hunger > 0.6:
            return {"SCAN": 4, "SURFACE": 2, "DRIFT_U": 2, "DRIFT_L": 1.5,
                    "DRIFT_R": 1.5, "TURN": 1, "BUBBLE": 0.6}
        if not self.light or self.sleepy():
            return {"HOVER": 5, "SINK": 3, "DRIFT_D": 2, "BUBBLE": 0.4}
        if self.boredom > 250:
            return {"CIRCLE": 4, "TURN": 3, "BUBBLE": 1.5,
                    "DRIFT_L": 1.5, "DRIFT_R": 1.5}
        return self.IDLE_MENU

    def dist(self):
        """Exact next-token distribution. Pure - mutates nothing."""
        p = np.zeros(V)
        if self.burst:
            p[STOI["<TAP>"]] = 0.6
            rest = 0.4
        else:
            p[STOI["<FOOD>"]] = P_STIM * 0.45
            p[STOI["<TAP>"]] = P_STIM * 0.25
            p[STOI["<HAND>"]] = P_STIM * 0.15
            p[STOI["<LIGHT_OFF>" if self.light else "<LIGHT_ON>"]] = P_STIM * 0.15
            rest = 1.0 - P_STIM

        m = self.current_mood()
        if m is not None and m != self.mood:
            p[STOI[m]] += rest
            return p

        f = min(1.0, self.fear)
        for menu, w in ((self.FEAR_MENU, rest * f), (self.menu(), rest * (1 - f))):
            if w <= 0:
                continue
            tot = sum(menu.values())
            for tok, v in menu.items():
                p[STOI[tok]] += w * v / tot
        return p

    def apply(self, tok):
        if self.burst:
            self.burst -= 1
        if tok in STIMULI:
            self.boredom = 0
            if tok == "<FOOD>":
                self.food = 60
            elif tok == "<TAP>":
                self.taps.append(self.t)
                self.startle = 1.0
                if not self.burst and self.rng.random() < 0.55:
                    self.burst = int(self.rng.integers(5, 20))
            elif tok == "<HAND>":
                self.startle = 1.0
            elif tok == "<LIGHT_OFF>":
                self.light = False
            elif tok == "<LIGHT_ON>":
                self.light = True
            return                     # a stimulus tick does not latch mood
        self.mood = self.current_mood()
        if tok == "CHOMP":
            self.hunger = max(0.0, self.hunger - 0.25)
            self.food = max(0, self.food - 22)

    def tick(self):
        """Advance time and decay. Call before dist()."""
        self.t += 1
        while self.taps and self.t - self.taps[0] > TAP_WINDOW:
            self.taps.popleft()
        self.hunger = min(1.0, self.hunger + 0.0012)
        self.startle *= 0.97
        self.boredom += 1
        if self.food:
            self.food -= 1

    def step(self, want_dist=False):
        self.tick()
        p = self.dist()
        i = int(np.searchsorted(np.cumsum(p), self.rng.random() * p.sum()))
        tok = VOCAB[min(i, V - 1)]
        self.apply(tok)
        return (tok, p) if want_dist else tok


def episode(rng, n):
    tk = Tank(rng)
    return [STOI["<BOS>"]] + [STOI[tk.step()] for _ in range(n)]


def main(a):
    rng = np.random.default_rng(a.seed)
    eps = [episode(rng, a.ep_len) for _ in range(a.episodes)]
    # hold out whole episodes: a mid-episode split would leak the fish's state
    # across the boundary, the same way a mid-movie split leaks context
    cut = max(1, int(len(eps) * 0.1))
    os.makedirs(a.out, exist_ok=True)
    for name, part in (("val", eps[:cut]), ("train", eps[cut:])):
        arr = np.array([t for e in part for t in e], dtype=np.uint16)
        arr.tofile(os.path.join(a.out, f"{name}.bin"))
        print(f"{name}: {len(part)} episodes, {len(arr):,} tokens")
    with open(os.path.join(a.out, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab": VOCAB, "ep_len": a.ep_len, "seed": a.seed}, f)
    print(f"vocab: {V}")


def oracle(a):
    """The generating distribution's own score: the floor no model can beat.

    Regenerates val with the same seed as main() - the rng is consumed in the
    same order, so these are the exact sequences eval.py scores.
    """
    import eval as ev
    rng = np.random.default_rng(a.seed)
    n_val = max(1, int(a.episodes * 0.1))
    toks, dists = [], []
    for _ in range(n_val):
        tk = Tank(rng)
        toks.append(STOI["<BOS>"])
        dists.append(np.full(V, 1.0 / V))          # <BOS> is unconditioned
        for _ in range(a.ep_len):
            tok, p = tk.step(want_dist=True)
            toks.append(STOI[tok])
            dists.append(p / p.sum())
    data = np.array(toks, dtype=np.uint16)
    D = np.asarray(dists, dtype=np.float32)

    bits = -np.log2(np.maximum(D[np.arange(len(data)), data], 1e-12))[512:].mean()
    predict = lambda _ctx, pos: D[pos]             # D[j] generated token j
    fear = [STOI[t] for t in ev.FEAR]
    h = ev.probe_habituation(predict, data, STOI, fear)
    g = ev.probe_hunger(predict, data, STOI)
    print(f"\noracle (irreducible)\n  val  {bits:.4f} bits/token")
    print("  habituation  " + "  ".join(f"{ev.HAB_LBL[k]}={v:.3f}" for k, v in h.items()))
    print("  hunger       " + "  ".join(f"{ev.HUN_LBL[k]}={v:.3f}" for k, v in g.items()))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "oracle.pkl"), "wb") as f:
        pickle.dump({"nll": float(bits), "hab": h, "hunger": g}, f)
    print(f"\nwrote {a.out}/oracle.pkl")


def inspect(a):
    tk = Tank(np.random.default_rng(a.seed))
    for i in range(a.n):
        tok = tk.step()
        mark = "  <<<" if tok in STIMULI else ""
        print(f"{i:4d}  {tok:<12} hunger={tk.hunger:.2f} fear={tk.fear:.2f} "
              f"taps={len(tk.taps):2d} food={tk.food:3d} light={int(tk.light)}{mark}")


def check():
    # dist() is a real distribution and step() samples from exactly it
    tk = Tank(np.random.default_rng(3))
    for _ in range(50):
        tk.tick()
        p = tk.dist()
        assert abs(p.sum() - 1.0) < 1e-9, p.sum()
        assert (p >= 0).all()
        tk.apply(VOCAB[int(np.argmax(p))])

    # habituation: 15 recent taps must blunt the response vs a single tap
    a = Tank(np.random.default_rng(0)); a.t = 1000
    a.taps.append(a.t); a.startle = 1.0
    b = Tank(np.random.default_rng(0)); b.t = 1000
    for i in range(15):
        b.taps.append(b.t - i)
    b.startle = 1.0
    assert b.fear < 0.25 * a.fear, (a.fear, b.fear)

    # hunger surfaces on its own, and eating clears it
    f = Tank(np.random.default_rng(1))
    f.hunger, f.light, f.t = 0.0, True, 0
    f.dist = lambda _s=f: _drop_stimuli(_s)
    toks = [f.step() for _ in range(700)]
    assert "HUNGRY" in toks, "hunger never surfaced"
    f.food = 300
    toks = [f.step() for _ in range(200)]
    assert "CHOMP" in toks and f.hunger < 0.6, (f.hunger, set(toks))
    print("sim self-check OK")


def _drop_stimuli(tk):
    """dist() with the stimulus mass removed and renormalized - lets the
    self-check isolate hunger without random feeding interfering."""
    p = Tank.dist(tk).copy()
    for s in STIMULI:
        p[STOI[s]] = 0.0
    return p / p.sum()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--ep-len", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default="data")
    a = p.parse_args()
    if a.check:
        check()
    elif a.oracle:
        oracle(a)
    elif a.inspect:
        inspect(a)
    else:
        main(a)
