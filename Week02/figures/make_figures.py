"""Regenerate the MA(1) and AR(1) example figures for Week 02.

    python3 make_figures.py

Writes ma1-example.png and ar1-example.png into this directory.
No statsmodels dependency -- the PACF is computed by Durbin-Levinson.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

SEED = 545
N = 500
NLAGS = 20

INK = "#1f2933"      # primary text
MUTED = "#7b8794"    # axes, grid, reference lines
ACCENT = "#1f6feb"   # the single data hue
BAND = "#c9d5e3"     # significance band fill

mpl.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 9,
    "axes.titleweight": "normal",
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#e4e7eb",
    "grid.linewidth": 0.6,
})


def acf(x, nlags):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    return np.array([np.dot(x[k:], x[: n - k]) / denom for k in range(nlags + 1)])


def pacf(x, nlags):
    """Durbin-Levinson recursion on the sample ACF."""
    r = acf(x, nlags)
    out = np.zeros(nlags + 1)
    out[0] = 1.0
    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = r[1]
    out[1] = r[1]
    v = 1 - r[1] ** 2
    for k in range(2, nlags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        phi[k, k] = num / v
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        v *= 1 - phi[k, k] ** 2
        out[k] = phi[k, k]
    return out


def corr_panel(ax, vals, n, title, theory=None):
    lags = np.arange(1, len(vals))
    v = vals[1:]
    bound = 1.96 / np.sqrt(n)
    ax.axhspan(-bound, bound, color=BAND, alpha=0.55, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.vlines(lags, 0, v, color=ACCENT, lw=2.0, zorder=2)
    ax.plot(lags, v, "o", ms=3.6, color=ACCENT, mec="white", mew=0.7, zorder=3)
    if theory is not None:
        ax.plot(lags, theory[: len(lags)], color=INK, lw=1.0, ls=(0, (3, 2)),
                zorder=4, label="theoretical")
        ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    ax.set_title(title, loc="left", pad=6)
    ax.set_xlabel("lag")
    ax.set_xlim(0.2, len(vals) - 0.2)
    ax.set_ylim(-0.45, 1.0)
    ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75])
    ax.grid(axis="y", zorder=0)


SHOW = 200   # points drawn in the series panel; the ACF still uses all N


def series_panel(ax, y, title):
    y = y[:SHOW]
    ax.plot(np.arange(len(y)), y, color=ACCENT, lw=1.0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(title, loc="left", pad=6)
    ax.set_xlabel("t  (first %d of %d simulated)" % (SHOW, N))
    ax.set_xlim(0, len(y))
    ax.grid(axis="y")


def build(kind, coef, fname):
    rng = np.random.default_rng(SEED)
    eps = rng.standard_normal(N + 1)
    if kind == "ma":
        y = eps[1:] + coef * eps[:-1]
        head = rf"MA(1):  $y_t = \epsilon_t + {coef}\,\epsilon_{{t-1}}$"
        th_acf = np.zeros(NLAGS)
        th_acf[0] = coef / (1 + coef ** 2)
        th_pacf = None
    else:
        y = np.zeros(N)
        for t in range(1, N):
            y[t] = coef * y[t - 1] + eps[t]
        y = y[50:]                      # drop the burn-in
        head = rf"AR(1):  $y_t = {coef}\,y_{{t-1}} + \epsilon_t$"
        th_acf = coef ** np.arange(1, NLAGS + 1)
        th_pacf = np.zeros(NLAGS)
        th_pacf[0] = coef

    n = len(y)
    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.62, wspace=0.22,
                          left=0.075, right=0.985, top=0.9, bottom=0.11)
    series_panel(fig.add_subplot(gs[0, :]), y, head)
    corr_panel(fig.add_subplot(gs[1, 0]), acf(y, NLAGS), n, "ACF", th_acf)
    corr_panel(fig.add_subplot(gs[1, 1]), pacf(y, NLAGS), n, "PACF", th_pacf)
    fig.savefig(fname, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", fname)


if __name__ == "__main__":
    build("ma", 0.6, "ma1-example.png")
    build("ar", 0.7, "ar1-example.png")
