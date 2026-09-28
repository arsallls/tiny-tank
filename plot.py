"""Probe-curve figures for the README.

Reads data/eval.pkl and data/oracle.pkl so the numbers can never drift from
what eval.py measured. Two panels, two measures, TWO Y-SCALES - which is why
they are two charts and not one with a second axis.

  python3 plot.py                 # figures/probes-light.png + probes-dark.png
"""
import argparse, os, pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots 1 and 2, both modes validated with the dataviz skill's
# validate_palette.js (light DE 24.7 protan / 33.6 normal; dark 23.5 / 29.0).
# The oracle is a REFERENCE, not a third series, so it is neutral + dashed +
# directly labelled rather than taking a categorical hue.
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e5e4e0",
                  s1="#2a78d6", s2="#eb6834", ref="#52514e"),
    "dark":  dict(surface="#1a1a19", ink="#f0efec", ink2="#a3a19c", grid="#2e2d2b",
                  s1="#3f86d8", s2="#d96e34", ref="#a3a19c"),
}
HAB = ["1", "2–3", "4–8", "9–15", "16+"]
HUN = ["<40", "40–90", "90–170", "170+"]


def series(d, key, n):
    return [d[key][i] for i in range(1, n + 1)]


def panel(ax, T, xs, curves, title, sub, ylab, ymax, label_side):
    ax.set_facecolor(T["surface"])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=T["grid"], lw=1)
    ax.xaxis.grid(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(T["grid"])
    ax.tick_params(colors=T["ink2"], length=0, labelsize=9.5)

    x = range(len(xs))
    for name, ys, col, dash in curves:
        ax.plot(x, ys, color=col, lw=2, ls=dash, marker="o", ms=5.5,
                mec=T["surface"], mew=1.6, zorder=3, label=name, clip_on=False)
    # direct labels only where the lines actually separate. On the habituation
    # panel oracle (0.640) and transformer (0.626) sit 0.014 apart and the labels
    # collide, so that panel leans on the shared legend instead.
    if label_side:
        i = len(xs) - 1
        # the transformer tracks the oracle to within 0.003 here, so the labels
        # overlap unless they are pushed apart to a readable minimum gap
        gap = 0.055 * ymax
        rows = sorted(((ys[i], name, col) for name, ys, col, _ in curves))
        placed = []
        for y, name, col in rows:
            if placed and y - placed[-1][0] < gap:
                y = placed[-1][0] + gap
            placed.append((y, name, col))
        for y, name, col in placed:
            ax.text(i + 0.14, y, name, color=col, fontsize=9.5, va="center",
                    ha="left", fontweight="medium")
        ax.set_xlim(-0.5, len(xs) - 0.5 + 1.15)

    ax.set_xticks(list(x)); ax.set_xticklabels(xs)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylab, color=T["ink2"], fontsize=9.5, labelpad=8)
    ax.set_title(title, color=T["ink"], fontsize=12.5, loc="left", pad=16,
                 fontweight="semibold")
    ax.text(0, 1.035, sub, transform=ax.transAxes, color=T["ink2"], fontsize=9.5)


def main(a):
    ev = pickle.load(open(os.path.join(a.data, "eval.pkl"), "rb"))
    orc = pickle.load(open(os.path.join(a.data, "oracle.pkl"), "rb"))
    ng = next(k for k in ev if k.endswith("-gram"))
    os.makedirs(a.out, exist_ok=True)

    for mode, T in THEME.items():
        fig, (l, r) = plt.subplots(1, 2, figsize=(11.6, 4.9), facecolor=T["surface"])
        panel(l, T, HAB,
              [("oracle", series(orc, "hab", 5), T["ref"], (0, (4, 2.5))),
               ("transformer", series(ev["transformer"], "hab", 5), T["s1"], "-"),
               (ng, series(ev[ng], "hab", 5), T["s2"], "-")],
              "Habituation", "P(fear response)  vs  taps already in the window",
              "probability", 0.70, None)
        panel(r, T, HUN,
              [("oracle", series(orc, "hunger", 4), T["ref"], (0, (4, 2.5))),
               ("transformer", series(ev["transformer"], "hunger", 4), T["s1"], "-"),
               (ng, series(ev[ng], "hunger", 4), T["s2"], "-")],
              "Hunger", "P(CHOMP)  vs  ticks since the last CHOMP",
              "probability", 0.33, "right")
        h, lb = l.get_legend_handles_labels()
        leg = fig.legend(h, lb, loc="lower left", bbox_to_anchor=(0.043, 0.905),
                         ncol=3, frameon=False, handlelength=2.2, fontsize=9.5,
                         columnspacing=1.8)
        for t in leg.get_texts():
            t.set_color(T["ink2"])
        fig.text(0.5, 0.015,
                 "the oracle is the simulator's own distribution — the floor no model can beat",
                 ha="center", color=T["ink2"], fontsize=9, style="italic")
        fig.tight_layout(rect=(0, 0.045, 0.995, 0.90))
        p = os.path.join(a.out, f"probes-{mode}.png")
        fig.savefig(p, dpi=170, facecolor=T["surface"])
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="figures")
    main(p.parse_args())
