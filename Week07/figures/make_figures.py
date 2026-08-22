"""Regenerate the Week 07 plot figures.

    python3 make_figures.py

Writes finite-difference.png, frontier-generic.png, efficient-frontier.png,
sharpe-line.png, and capital-market-line.png. The two binomial trees are TikZ -- see amput.tex and
divtree.tex, built by the Makefile's `figures` target.

No data is involved. Everything here is a closed-form function or the solution
of the three asset optimization stated in the notes.

Note on axes: risk is plotted as STANDARD DEVIATION, not variance. The capital
market line is a straight line in (sigma, mu) space and a square root curve in
(sigma^2, mu) space, so the tangency argument only reads correctly against sigma.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize

INK = "#1f2933"
MUTED = "#7b8794"
C1 = "#1f6feb"     # frontier / primary curve
C2 = "#b54708"     # capital market line / secondary
C3 = "#087443"     # third series
FILL = "#dbe8fb"

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 9, "axes.titleweight": "normal", "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e4e7eb", "grid.linewidth": 0.6,
})

# --- the three asset example from the notes -------------------------------
CORR = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]])
SD = np.array([0.20, 0.10, 0.05])
ER = np.array([0.05, 0.04, 0.03])
COV = np.outer(SD, SD) * CORR
RF = 0.035
SUM_TO_ONE = {"type": "eq", "fun": lambda w: w.sum() - 1}
BOUNDS = [(0.0, 1.0)] * 3


def min_var(target):
    cons = [SUM_TO_ONE, {"type": "eq", "fun": lambda w: w @ ER - target}]
    res = minimize(lambda w: w @ COV @ w, np.ones(3) / 3, bounds=BOUNDS,
                   constraints=cons, tol=1e-14)
    return res.x


def frontier(n=160):
    lo = minimize(lambda w: w @ COV @ w, np.ones(3) / 3, bounds=BOUNDS,
                  constraints=[SUM_TO_ONE], tol=1e-14).x
    targets = np.linspace(lo @ ER, ER.max(), n)
    pts = [(np.sqrt(w @ COV @ w), t) for t in targets for w in [min_var(t)]]
    return np.array(pts)


def max_sharpe():
    res = minimize(lambda w: -(w @ ER - RF) / np.sqrt(w @ COV @ w),
                   np.ones(3) / 3, bounds=BOUNDS, constraints=[SUM_TO_ONE], tol=1e-14)
    w = res.x
    return w, np.sqrt(w @ COV @ w), w @ ER


def finite_difference():
    """Each approximation is drawn as the chord between the points it uses.

    f is a sine, so the central difference is not exact and the picture is
    honest. At x=0.9 with Delta=0.55 the slopes are backward 0.801, central
    0.591, forward 0.381, against a true derivative of 0.622.
    """
    f, fp = np.sin, np.cos
    x0, h = 0.9, 0.55
    xs = np.linspace(0.0, 2.75, 500)

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(xs, f(xs), color=INK, lw=2.0, zorder=3)
    ax.annotate("$f(x)$", (2.6, f(2.6)), textcoords="offset points",
                xytext=(6, 0), fontsize=8.5, color=INK)

    tx = np.array([x0 - 0.72, x0 + 0.72])
    ax.plot(tx, f(x0) + fp(x0) * (tx - x0), color=MUTED, lw=1.1, ls=(0, (4, 3)),
            zorder=2)
    ax.annotate("true tangent at $x$", (0.5, f(x0) + fp(x0) * (0.5 - x0)),
                textcoords="offset points", xytext=(-2, 10), ha="right",
                fontsize=7.5, color=MUTED)

    chords = (
        ("backward difference", x0 - h, x0, C2, (-3, 7), "right"),
        ("central difference", x0 - h, x0 + h, C1, (4, 4), "left"),
        ("forward difference", x0, x0 + h, C3, (4, -6), "left"),
    )
    ext = 0.42   # draw each line past its two points so the slope is visible
    for name, xa, xb, c, off, ha in chords:
        m = (f(xb) - f(xa)) / (xb - xa)
        xe = np.array([xa - ext, xb + ext])
        ax.plot(xe, f(xa) + m * (xe - xa), color=c, lw=1.9, zorder=4)
        # always label at the upper right end of the drawn line
        ax.annotate(name, (xb + ext, f(xb) + m * ext), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=8, color=c)

    for x, lab in ((x0 - h, "$x-\\Delta$"), (x0, "$x$"), (x0 + h, "$x+\\Delta$")):
        ax.vlines(x, 0.02, f(x), color=MUTED, lw=0.9, ls=(0, (2, 2)), zorder=1)
        ax.plot([x], [f(x)], "o", ms=5, color=INK, mec="white", mew=0.9, zorder=6)
        ax.annotate(lab, (x, 0.02), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=8, color=INK)

    ax.set_xlim(0.02, 2.95)
    ax.set_ylim(0.02, 1.40)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0.02))
    ax.set_title("Each line runs through the two points its formula uses",
                 loc="left", pad=6)
    fig.tight_layout()
    fig.savefig("finite-difference.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote finite-difference.png")
    print(f"  true {fp(x0):.4f}  central {(f(x0+h)-f(x0-h))/(2*h):.4f} "
          f"forward {(f(x0+h)-f(x0))/h:.4f}  backward {(f(x0)-f(x0-h))/h:.4f}")


def frontier_generic():
    """A schematic frontier. No numbers, no units -- risk against reward."""
    t = np.linspace(-1.0, 1.0, 400)
    risk = 0.30 + 0.62 * t ** 2
    ret = 0.50 + 0.40 * t
    upper = t >= 0

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    # the achievable set is everything to the right of the frontier
    ax.fill_betweenx(ret, risk, 1.06, color=FILL, lw=0, zorder=1)
    ax.plot(risk[~upper], ret[~upper], color=MUTED, lw=2.0, ls=(0, (4, 3)), zorder=3)
    ax.plot(risk[upper], ret[upper], color=C1, lw=2.4, zorder=4)
    ax.plot([0.30], [0.50], "o", ms=6, color=C1, mec="white", mew=1.0, zorder=5)

    ax.annotate("efficient frontier", (risk[330], ret[330]),
                textcoords="offset points", xytext=(-8, 10), ha="right",
                fontsize=8.5, color=C1)
    ax.annotate("inefficient: same risk,\nless return", (risk[45], ret[45]),
                textcoords="offset points", xytext=(12, 6), fontsize=8,
                color=MUTED, linespacing=1.5)
    ax.annotate("minimum risk portfolio", (0.30, 0.50), textcoords="offset points",
                xytext=(16, 10), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=0, shrinkB=4))
    ax.annotate("achievable portfolios", (0.80, 0.50), textcoords="offset points",
                xytext=(0, 0), ha="center", fontsize=8, color=MUTED)

    ax.set_xlabel("risk")
    ax.set_ylabel("expected return")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0.18, 1.06)
    ax.set_ylim(0.14, 0.96)
    ax.set_title("The efficient frontier", loc="left", pad=6)
    fig.tight_layout()
    fig.savefig("frontier-generic.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote frontier-generic.png")


def efficient_frontier():
    pts = frontier()
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(pts[:, 0], pts[:, 1], color=C1, lw=2.2)
    for i, (nm, s, m) in enumerate(zip(["A1", "A2", "A3"], SD, ER)):
        ax.plot([s], [m], "o", ms=5, color=MUTED, mec="white", mew=0.8, zorder=5)
        ax.annotate(nm, (s, m), textcoords="offset points", xytext=(6, -3),
                    fontsize=8, color=MUTED)
    ax.annotate("efficient frontier", (pts[len(pts) // 2, 0], pts[len(pts) // 2, 1]),
                textcoords="offset points", xytext=(8, -14), fontsize=8, color=C1)
    _axes(ax, "Efficient frontier, three assets, no shorting")
    fig.tight_layout()
    fig.savefig("efficient-frontier.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote efficient-frontier.png")


def _axes(ax, title):
    ax.set_title(title, loc="left", pad=6)
    ax.set_xlabel("risk, standard deviation $\\sigma$")
    ax.set_ylabel("expected return")
    ax.set_xlim(0, 0.235)
    ax.set_ylim(0.028, 0.056)
    ax.grid(axis="y")


def sharpe_line():
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    rf = 0.05
    x = np.linspace(0, 0.26, 100)
    for nm, mu, sd, c, off, ha in (("Investment A", 0.10, 0.16, C1, (-8, 12), "right"),
                                   ("Investment B", 0.08, 0.10, C2, (8, -16), "left")):
        sr = (mu - rf) / sd
        ax.plot(x, rf + sr * x, color=c, lw=2.0)
        ax.plot([sd], [mu], "o", ms=6, color=c, mec="white", mew=1.0, zorder=6)
        ax.annotate(f"{nm}\nSR = {sr:.4f}", (sd, mu), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=8, color=c, linespacing=1.5)
    ax.vlines(0.08, rf, rf + ((0.10 - rf) / 0.16) * 0.08, color=MUTED, lw=0.9,
              ls=(0, (2, 2)))
    ax.annotate("8% risk budget", (0.08, 0.052), textcoords="offset points",
                xytext=(5, 0), fontsize=7.5, color=MUTED)
    ax.plot([0], [rf], "o", ms=5, color=MUTED, mec="white", mew=0.8, zorder=5)
    ax.annotate("$r_{rf} = 5\\%$", (0, rf), textcoords="offset points",
                xytext=(7, 4), fontsize=8, color=MUTED)
    ax.set_title("Each investment combined with the risk free asset", loc="left", pad=6)
    ax.set_xlabel("risk, standard deviation $\\sigma$")
    ax.set_ylabel("expected return")
    ax.set_xlim(0, 0.26)
    ax.set_ylim(0.046, 0.138)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig("sharpe-line.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote sharpe-line.png")


def capital_market_line():
    pts = frontier()
    w, sig, mu = max_sharpe()
    sr = (mu - RF) / sig
    x = np.linspace(0, 0.235, 100)

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(pts[:, 0], pts[:, 1], color=C1, lw=2.2)
    ax.plot(x, RF + sr * x, color=C2, lw=2.0)
    ax.plot([sig], [mu], "o", ms=6, color=C2, mec="white", mew=1.0, zorder=6)
    ax.plot([0], [RF], "o", ms=5, color=MUTED, mec="white", mew=0.8, zorder=5)

    ax.annotate("efficient frontier", (pts[70, 0], pts[70, 1]),
                textcoords="offset points", xytext=(8, -14), fontsize=8, color=C1)
    ax.annotate("capital market line", (0.215, RF + sr * 0.215),
                textcoords="offset points", xytext=(-4, 8), ha="right",
                fontsize=8, color=C2)
    ax.annotate(f"max Sharpe portfolio\n$\\sigma={sig:.3f}$, $E(r)={mu:.4f}$",
                (sig, mu), textcoords="offset points", xytext=(10, -26),
                fontsize=7.5, color=INK, linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=0, shrinkB=4))
    ax.annotate("$r_{rf} = 3.5\\%$", (0, RF), textcoords="offset points",
                xytext=(6, -12), fontsize=8, color=MUTED)
    _axes(ax, "The capital market line")
    fig.tight_layout()
    fig.savefig("capital-market-line.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote capital-market-line.png")
    print(f"  max Sharpe weights {np.round(w, 4)}  sigma={sig:.4f} "
          f"E(r)={mu:.4f} SR={sr:.4f}")


if __name__ == "__main__":
    finite_difference()
    frontier_generic()
    efficient_frontier()
    sharpe_line()
    capital_market_line()
