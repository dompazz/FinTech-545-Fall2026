"""Regenerate the exponential weight figure for Week 03.

    python3 make_figures.py

Writes ew-weights.png and ess-lambda.png into this directory. No data is involved --
these are plots of the weight function w_{t-i} = (1-lambda) * lambda^(i-1) itself and
of two summary statistics derived from it in closed form.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#1f2933"
MUTED = "#7b8794"
C_DAILY = "#1f6feb"    # lambda = 0.94
C_MONTHLY = "#b54708"  # lambda = 0.97
C_EQUAL = "#5f6b7a"    # equal weighting

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 9, "axes.titleweight": "normal", "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e4e7eb", "grid.linewidth": 0.6,
})

N = 260  # one trading year


def weights(lam, n=N):
    w = (1 - lam) * lam ** np.arange(n)
    return w / w.sum()


def main():
    i = np.arange(1, N + 1)
    w94, w97 = weights(0.94), weights(0.97)
    weq = np.full(N, 1.0 / N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    for w, c, lab in ((w94, C_DAILY, r"$\lambda = 0.94$"),
                      (w97, C_MONTHLY, r"$\lambda = 0.97$"),
                      (weq, C_EQUAL, "equal weight")):
        ax1.plot(i, w, color=c, lw=2.0, label=lab,
                 ls="-" if c != C_EQUAL else (0, (4, 2)))
        ax2.plot(i, np.cumsum(w), color=c, lw=2.0, label=lab,
                 ls="-" if c != C_EQUAL else (0, (4, 2)))

    ax1.set_title("Weight on the observation $i$ days ago", loc="left", pad=6)
    ax1.set_xlabel("$i$ (days ago)")
    ax1.set_ylabel("normalized weight")
    ax1.set_xlim(0, N)
    ax1.grid(axis="y")
    ax1.legend(frameon=False, fontsize=7.5)

    ax2.set_title("Cumulative weight", loc="left", pad=6)
    ax2.set_xlabel("$i$ (days ago)")
    ax2.set_xlim(0, N)
    ax2.set_ylim(0, 1.02)
    ax2.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax2.grid(axis="y")

    # half life markers, offset in opposite directions so the labels do not collide
    for w, c, off in ((w94, C_DAILY, (-6, 9)), (w97, C_MONTHLY, (7, -13))):
        k = int(np.searchsorted(np.cumsum(w), 0.5)) + 1
        ax2.plot([k], [0.5], "o", ms=5, color=c, mec="white", mew=0.8, zorder=5)
        ax2.annotate(f"{k} days", (k, 0.5), textcoords="offset points",
                     xytext=off, fontsize=7.5, color=INK,
                     ha="right" if off[0] < 0 else "left")

    fig.tight_layout()
    fig.savefig("ew-weights.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote ew-weights.png")
    for lam in (0.94, 0.97):
        w = weights(lam)
        print(f"lambda={lam}: half of the weight in the last "
              f"{int(np.searchsorted(np.cumsum(w), 0.5)) + 1} days, "
              f"effective n = {1/np.sum(w**2):.0f}")


def ess_figure():
    """Effective sample size and half life as functions of lambda.

    Both have closed forms on the infinite horizon:
        ESS       = (1 + lam) / (1 - lam)
        half life = ln(0.5) / ln(lam)
    """
    lam = np.linspace(0.80, 0.995, 400)
    ess = (1 + lam) / (1 - lam)
    hl = np.log(0.5) / np.log(lam)
    marks = (0.94, 0.97)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    for ax, y, lab, title in (
        (ax1, ess, "observations", r"Effective sample size  $(1+\lambda)/(1-\lambda)$"),
        (ax2, hl, "days", r"Half life  $\ln(0.5)/\ln(\lambda)$"),
    ):
        ax.plot(lam, y, color=C_DAILY, lw=2.0)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", pad=6)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(lab)
        ax.set_xlim(0.80, 0.995)
        ax.grid(axis="y")
        for m, c, off in zip(marks, (C_DAILY, C_MONTHLY), ((-9, -4), (-9, 4))):
            v = (1 + m) / (1 - m) if ax is ax1 else np.log(0.5) / np.log(m)
            ax.plot([m], [v], "o", ms=5.5, color=c, mec="white", mew=0.8, zorder=5)
            ax.annotate(rf"$\lambda={m}$: {v:.0f}", (m, v), textcoords="offset points",
                        xytext=off, fontsize=7.5, color=INK,
                        ha="right" if off[0] < 0 else "left")

    fig.tight_layout()
    fig.savefig("ess-lambda.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote ess-lambda.png")
    for m in (0.90, 0.94, 0.97, 0.99):
        print(f"  lambda={m}: ESS={(1+m)/(1-m):.0f}, "
              f"half life={np.log(0.5)/np.log(m):.1f}d, "
              f"weight on newest obs={(1-m)*100:.0f}%")


if __name__ == "__main__":
    main()
    ess_figure()
