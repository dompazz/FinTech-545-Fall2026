"""Regenerate the Week 05 figures.

    python3 make_figures.py

Writes es-vs-var.png and joint-exceedance.png. No data is involved -- both are
plots of closed-form functions.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

INK = "#1f2933"
MUTED = "#7b8794"
BODY = "#cfe0f7"
C1 = "#1f6feb"    # first series
C2 = "#b54708"    # second series
C3 = "#087443"    # third series
C4 = "#6b3fa0"    # fourth series

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 9, "axes.titleweight": "normal", "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e4e7eb", "grid.linewidth": 0.6,
})

ALPHA = 0.05


def norm_var(a=ALPHA):
    return -stats.norm.ppf(a)


def norm_es(a=ALPHA):
    return stats.norm.pdf(stats.norm.ppf(a)) / a


def t_var(nu, a=ALPHA):
    """VaR of a t scaled to unit variance."""
    return -stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2))


def t_es(nu, a=ALPHA):
    """ES of a t scaled to unit variance. Closed form, checked numerically."""
    q = stats.t.ppf(a, nu)
    es_raw = stats.t.pdf(q, nu) / a * (nu + q ** 2) / (nu - 1)
    return es_raw / np.sqrt(nu / (nu - 2))


def es_vs_var():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # --- left: where the two numbers sit on a density ---
    x = np.linspace(-4.2, 4.2, 900)
    y = stats.norm.pdf(x)
    v, e = norm_var(), norm_es()
    ax1.fill_between(x, y, color=BODY, lw=0)
    xt = x[x <= -v]
    ax1.fill_between(xt, stats.norm.pdf(xt), color=C2, lw=0, alpha=0.85)
    ax1.plot(x, y, color=C1, lw=1.6)

    hv, he = 0.235, 0.325
    ax1.vlines(-v, 0, hv, color=C1, lw=1.4)
    ax1.vlines(-e, 0, he, color=C2, lw=1.4)
    ax1.annotate(f"VaR = {v:.2f}$\\sigma$", (-v, hv), textcoords="offset points",
                 xytext=(4, -1), fontsize=8, color=C1)
    ax1.annotate(f"ES = {e:.2f}$\\sigma$", (-e, he), textcoords="offset points",
                 xytext=(4, -1), fontsize=8, color=C2)
    ax1.annotate("mean of the shaded tail", (-e, he), textcoords="offset points",
                 xytext=(4, -12), fontsize=7, color=MUTED)
    ax1.set_title("Standard normal, $\\alpha = 5\\%$", loc="left", pad=6)
    ax1.set_xlabel("standard deviations")
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.set_xlim(-4.2, 4.2)
    ax1.set_ylim(0, 0.46)

    # --- right: ES/VaR ratio as the tail gets heavier ---
    nus = np.linspace(2.6, 30, 400)
    ratio = np.array([t_es(n) / t_var(n) for n in nus])
    ax2.plot(nus, ratio, color=C1, lw=2.0, label="$t_\\nu$, unit variance")
    ax2.axhline(norm_es() / norm_var(), color=C2, lw=1.6, ls=(0, (4, 2)),
                label="normal, 1.25")
    for n in (3, 5, 10):
        r = t_es(n) / t_var(n)
        ax2.plot([n], [r], "o", ms=5, color=C1, mec="white", mew=0.8, zorder=5)
        ax2.annotate(f"$\\nu={n}$: {r:.2f}", (n, r), textcoords="offset points",
                     xytext=(7, 3), fontsize=7.5, color=INK)
    ax2.set_title("ES / VaR at $\\alpha = 5\\%$", loc="left", pad=6)
    ax2.set_xlabel("degrees of freedom $\\nu$")
    ax2.set_xlim(2.6, 30)
    ax2.grid(axis="y")
    ax2.legend(frameon=False, fontsize=7.5, loc="upper right")

    fig.tight_layout()
    fig.savefig("es-vs-var.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote es-vs-var.png   normal VaR={v:.4f} ES={e:.4f} ratio={e/v:.4f}")
    for n in (3, 5, 10, 30):
        print(f"  t_{n}: VaR={t_var(n):.3f} ES={t_es(n):.3f} ratio={t_es(n)/t_var(n):.3f}")


# rho is an ordered magnitude, so this is a sequential ramp, not categorical hues
RAMP = ["#6ba3f2", "#2f7ce8", "#1553b0", "#08306b"]


def joint_exceedance():
    """P(U1 <= u | U2 <= u) for the Gaussian copula, as u goes to 0.

    The limit is the lower tail dependence coefficient, which is 0 for every
    rho < 1. Nothing here needs a copula we do not teach.
    """
    us = np.logspace(-6, -1, 160)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    for rho, c in zip((0.3, 0.5, 0.7, 0.9), RAMP):
        y = [stats.multivariate_normal.cdf([stats.norm.ppf(u)] * 2, mean=[0, 0],
                                           cov=[[1, rho], [rho, 1]]) / u for u in us]
        ax.plot(us, y, color=c, lw=2.0)
        ax.annotate(rf"$\rho = {rho}$", (us[-1], y[-1]), textcoords="offset points",
                    xytext=(6, -2), fontsize=8, color=c)

    ax.set_xscale("log")
    ax.set_title("Gaussian copula: $P(U_1 \\leq u \\mid U_2 \\leq u)$", loc="left", pad=6)
    ax.set_xlabel("$u$")
    ax.set_ylabel("conditional probability")
    ax.set_xlim(1e-6, 1.6e-1)
    ax.set_ylim(0, 0.78)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig("joint-exceedance.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote joint-exceedance.png")
    for rho in (0.3, 0.5, 0.7, 0.9):
        vals = []
        for u in (0.05, 0.001, 1e-5):
            c = stats.multivariate_normal.cdf([stats.norm.ppf(u)] * 2, mean=[0, 0],
                                              cov=[[1, rho], [rho, 1]]) / u
            vals.append(f"u={u:g}: {c:.3f}")
        print("  rho=%.1f  " % rho + "  ".join(vals))


if __name__ == "__main__":
    es_vs_var()
    joint_exceedance()
