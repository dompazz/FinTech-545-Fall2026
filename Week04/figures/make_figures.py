"""Regenerate the VaR distribution figure for Week 04.

    python3 make_figures.py

Writes var-distribution.png into this directory. No data is involved -- the curve
is a scaled Student's t chosen so the mean sits at 1 and the 5% quantile at -8,
which are the numbers quoted in the text.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

INK = "#1f2933"
MUTED = "#7b8794"
BODY = "#cfe0f7"      # bulk of the distribution
TAIL = "#b54708"      # the alpha% tail
LINE = "#1f6feb"

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 9, "axes.titleweight": "normal", "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "grid.color": "#e4e7eb", "grid.linewidth": 0.6,
})

DF = 5
MU = 1.0
ALPHA = 0.05
Q = -8.0                                   # target 5% quantile
S = (MU - Q) / abs(stats.t.ppf(ALPHA, DF))  # scale that puts the quantile at -8


def main():
    x = np.linspace(-26, 26, 1200)
    y = stats.t.pdf((x - MU) / S, DF) / S
    peak = y.max()

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    ax.fill_between(x, y, color=BODY, lw=0)
    xt = x[x <= Q]
    ax.fill_between(xt, stats.t.pdf((xt - MU) / S, DF) / S, color=TAIL, lw=0, alpha=0.9)
    ax.plot(x, y, color=LINE, lw=1.6)

    # the alpha% cut
    yq = stats.t.pdf((Q - MU) / S, DF) / S
    ax.vlines(Q, 0, yq, color=TAIL, lw=1.6)
    ax.annotate("$F^{-1}(0.05) = -8$\n5% of outcomes fall left of here",
                (Q - 0.6, yq), textcoords="offset points", xytext=(-10, 30),
                ha="right", va="bottom", fontsize=8, color=INK, linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=0, shrinkB=2))

    # the mean
    ax.vlines(MU, 0, peak, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    ax.annotate(r"mean $\approx 1$", (MU, peak), textcoords="offset points",
                xytext=(7, -3), fontsize=8, color=INK)

    # the two reporting conventions, drawn below the baseline
    u = peak * 0.115
    for k, (x1, lab) in enumerate(((0.0, "absolute:  VaR = 8"),
                                   (MU, "relative to the mean:  VaR = 9"))):
        y0 = -(1.6 + 0.95 * k) * u
        ax.annotate("", (Q, y0), (x1, y0),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
        ax.annotate(lab, (Q - 0.8, y0), textcoords="offset points", xytext=(0, -3),
                    ha="right", va="center", fontsize=7.5, color=INK)

    ax.set_title("Profit and loss distribution", loc="left", pad=6)
    ax.set_xlabel("profit / loss", labelpad=40)
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.set_xlim(-27, 27)
    ax.set_ylim(-3.4 * u, peak * 1.16)
    ax.spines["bottom"].set_position(("data", 0))
    ax.grid(False)

    fig.savefig("var-distribution.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote var-distribution.png")
    print(f"  scale={S:.4f}, 5% quantile={MU + S*stats.t.ppf(ALPHA, DF):.3f}")


if __name__ == "__main__":
    main()
