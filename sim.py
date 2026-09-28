"""Fish behavior simulator -> token sequences.

The fish has hidden state (hunger, startle, habituation, boredom, circadian)
that evolves over hundreds of ticks. The token stream never states that state
directly - moods fire only on the tick they CHANGE - so recovering it needs
long-range context. That gap is the point: an n-gram cannot do it.

  python3 sim.py --inspect        # read 200 ticks, with state annotations
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

TAP_WINDOW = 300      # ticks a tap still counts toward habituation
DAY = 4000            # circadian period in ticks


class Tank:
    def __init__(self, rng):
        self.rng = rng
        self.hunger = rng.random() * 0.3
        self.startle = 0.0
        self.boredom = 0
        self.light = True
        self.food = 0                 # ticks of food still floating
        self.burst = 0                # remaining ticks of a tap burst
        self.taps = deque()
        self.t = int(rng.integers(0, DAY))   # start anywhere in the cycle
        self.mood = None

    @property
    def fear(self):
        # habituation: each recent tap makes the next one land less.
        # 0.15 not 0.45 - a steeper decay saturates to zero fear by the 4th
        # tap, which an n-gram can match by just detecting a recent tap.
        # The graded curve is what forces the model to actually count.
        return self.startle * float(np.exp(-0.15 * len(self.taps)))

    def sleepy(self):
        return np.sin(2 * np.pi * self.t / DAY) < -0.7

    def pick(self, d):
        toks = list(d)
        c = np.cumsum(np.fromiter(d.values(), float))
        return toks[int(np.searchsorted(c, self.rng.random() * c[-1]))]

    def stimulus(self):
        """A stimulus token or None.

        Bursty on purpose: habituation only appears in the data if taps
        sometimes arrive in rapid runs. Uniform arrivals would never teach it.
        """
        if self.burst:
            self.burst -= 1
            return "<TAP>" if self.rng.random() < 0.6 else None
        if self.rng.random() > 0.007:
            return None
        r = self.rng.random()
        if r < 0.45:
            return "<FOOD>"
        if r < 0.70:
            self.burst = int(self.rng.integers(5, 20))
            return "<TAP>"
        if r < 0.85:
            return "<HAND>"
        return "<LIGHT_OFF>" if self.light else "<LIGHT_ON>"

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

    def behave(self):
        if self.rng.random() < self.fear:
            return self.pick({"HIDE": 4, "DART_L": 2, "DART_R": 2, "DART_D": 3,
                              "SINK": 1, "SPLASH": 0.5})
        if self.food:
            # three tiers, because a full fish that ignores food entirely reads
            # as a broken demo: clicking Feed must always do something visible
            if self.hunger > 0.25:
                return self.pick({"ORIENT": 2, "DART_U": 3, "DART_L": 1, "DART_R": 1,
                                  "CHOMP": 4, "NIBBLE": 2, "GULP": 1.5})
            if self.hunger > 0.08:
                return self.pick({"ORIENT": 3, "NIBBLE": 3, "DRIFT_U": 2,
                                  "CHOMP": 1, "TURN": 1, "BUBBLE": 0.5})
            if self.rng.random() < 0.35:
                return self.pick({"ORIENT": 2, "DRIFT_U": 1, "TURN": 1})
        if self.hunger > 0.6:
            return self.pick({"SCAN": 4, "SURFACE": 2, "DRIFT_U": 2, "DRIFT_L": 1.5,
                              "DRIFT_R": 1.5, "TURN": 1, "BUBBLE": 0.6})
        if not self.light or self.sleepy():
            return self.pick({"HOVER": 5, "SINK": 3, "DRIFT_D": 2, "BUBBLE": 0.4})
        if self.boredom > 250:
            return self.pick({"CIRCLE": 4, "TURN": 3, "BUBBLE": 1.5,
                              "DRIFT_L": 1.5, "DRIFT_R": 1.5})
        return self.pick({"DRIFT_L": 3, "DRIFT_R": 3, "DRIFT_U": 1.5, "DRIFT_D": 1.5,
                          "HOVER": 3, "TURN": 1.5, "BUBBLE": 0.8, "CIRCLE": 0.7})

    def step(self):
        self.t += 1
        while self.taps and self.t - self.taps[0] > TAP_WINDOW:
            self.taps.popleft()
        self.hunger = min(1.0, self.hunger + 0.0012)
        self.startle *= 0.97
        self.boredom += 1
        if self.food:
            self.food -= 1

        s = self.stimulus()
        if s:
            self.boredom = 0
            if s == "<FOOD>":
                self.food = 60
            elif s == "<TAP>":
                self.taps.append(self.t)
                self.startle = 1.0
            elif s == "<HAND>":
                self.startle = 1.0
            elif s == "<LIGHT_OFF>":
                self.light = False
            elif s == "<LIGHT_ON>":
                self.light = True
            return s

        # moods fire on transition only - emitting one every tick would collapse
        # the task to "copy the previous mood", learnable by a bigram
        m = self.current_mood()
        if m != self.mood:
            self.mood = m
            if m:
                return m

        tok = self.behave()
        if tok == "CHOMP":
            self.hunger = max(0.0, self.hunger - 0.25)
            self.food = max(0, self.food - 22)
        return tok


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
        pickle.dump({"vocab": VOCAB, "ep_len": a.ep_len}, f)
    print(f"vocab: {len(VOCAB)}")


def inspect(a):
    rng = np.random.default_rng(a.seed)
    tk = Tank(rng)
    for i in range(a.n):
        tok = tk.step()
        mark = "  <<<" if tok in STIMULI else ""
        print(f"{i:4d}  {tok:<12} hunger={tk.hunger:.2f} fear={tk.fear:.2f} "
              f"taps={len(tk.taps):2d} food={tk.food:3d} light={int(tk.light)}{mark}")


def check():
    # habituation: 15 recent taps must blunt the response vs a single tap
    a = Tank(np.random.default_rng(0)); a.t = 1000
    a.taps.append(a.t); a.startle = 1.0
    b = Tank(np.random.default_rng(0)); b.t = 1000
    for i in range(15):
        b.taps.append(b.t - i)
    b.startle = 1.0
    assert b.fear < 0.25 * a.fear, (a.fear, b.fear)

    # hunger crosses the mood threshold on its own, and eating clears it
    f = Tank(np.random.default_rng(1))
    f.hunger, f.light = 0.0, True
    f.stimulus = lambda: None
    f.t = 0                              # awake: sin(0)=0 > -0.7
    toks = [f.step() for _ in range(700)]
    assert "HUNGRY" in toks, "hunger never surfaced"
    f.food = 300
    toks = [f.step() for _ in range(200)]
    assert "CHOMP" in toks and f.hunger < 0.6, (f.hunger, set(toks))
    print("sim self-check OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--ep-len", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default="data")
    a = p.parse_args()
    if a.check:
        check()
    elif a.inspect:
        inspect(a)
    else:
        main(a)
