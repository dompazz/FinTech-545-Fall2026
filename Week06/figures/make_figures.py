"""Regenerate the Week 06 plot figures.

    python3 make_figures.py

Writes payoffs.png and value-vs-payoff.png. The binomial trees are TikZ --
see tree1.tex and tree2.tex, built by the Makefile's `figures` target.

No data is involved. Both plots are the payoff function and the generalized
Black Scholes Merton formula evaluated over a range of underlying prices.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

INK = "#1f2933"
MUTED = "#7b8794"
C_VAL = "#1f6feb"     # option value
C_PAY = "#b54708"     # payoff at expiry
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

X = 100.0


def gbsm(S, X, T, r, b, sig, call=True):
    S = np.asarray(S, float)
    d1 = (np.log(S / X) + (b + sig ** 2 / 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if call:
        return S * np.exp((b - r) * T) * stats.norm.cdf(d1) - X * np.exp(-r * T) * stats.norm.cdf(d2)
    return X * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp((b - r) * T) * stats.norm.cdf(-d1)


def payoffs():
    S = np.linspace(40, 160, 600)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)

    for ax, pay, name, ann in (
        (a1, np.maximum(0, S - X), "Call:  $\\pi = \\max(0,\\ S - X)$", (134, 16)),
        (a2, np.maximum(0, X - S), "Put:  $\\pi = \\max(0,\\ X - S)$", (66, 16)),
    ):
        ax.fill_between(S, pay, color=FILL, lw=0)
        ax.plot(S, pay, color=C_PAY, lw=2.2)
        ax.axvline(X, color=MUTED, lw=1.0, ls=(0, (3, 2)))
        ax.annotate("strike $X$", (X, 58), textcoords="offset points",
                    xytext=(4, 0), fontsize=7.5, color=MUTED)
        ax.annotate("in the money", ann, fontsize=7.5, color=C_PAY, ha="center")
        ax.set_title(name, loc="left", pad=6)
        ax.set_xlabel("underlying price $S$ at expiry")
        ax.set_xlim(40, 160)
        ax.set_ylim(-2, 64)
        ax.grid(axis="y")
    a1.set_ylabel("payoff $\\pi$")

    fig.tight_layout()
    fig.savefig("payoffs.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote payoffs.png")


def value_vs_payoff():
    """X=100, r=8%, T=1, sigma=30% -- the parameters quoted in the text."""
    r, T, sig = 0.08, 1.0, 0.30
    S = np.linspace(1, 200, 800)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))

    # call: value always above payoff
    a1.plot(S, np.maximum(0, S - X), color=C_PAY, lw=2.0, ls=(0, (4, 2)))
    a1.plot(S, gbsm(S, X, T, r, r, sig, True), color=C_VAL, lw=2.2)
    a1.annotate("value", (150, gbsm(150, X, T, r, r, sig, True)),
                textcoords="offset points", xytext=(-4, 10), ha="right",
                fontsize=8, color=C_VAL)
    a1.annotate("payoff", (150, 50), textcoords="offset points",
                xytext=(6, -10), fontsize=8, color=C_PAY)
    a1.set_title("European call", loc="left", pad=6)

    # put: value falls below payoff deep in the money
    pay = np.maximum(0, X - S)
    val = gbsm(S, X, T, r, r, sig, False)
    a2.fill_between(S, val, pay, where=(pay > val), color="#f3ddd0", lw=0)
    a2.plot(S, pay, color=C_PAY, lw=2.0, ls=(0, (4, 2)))
    a2.plot(S, val, color=C_VAL, lw=2.2)
    gap = X * (1 - np.exp(-r * T))
    a2.annotate(f"gap $\\to X(1-e^{{-rT}}) = {gap:.2f}$", (8, 88),
                textcoords="offset points", xytext=(52, -26), ha="left",
                fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=2, shrinkB=2))
    for s_ in (30, 50):
        a2.plot([s_], [gbsm(s_, X, T, r, r, sig, False)], "o", ms=4.5,
                color=C_VAL, mec="white", mew=0.8, zorder=5)
    a2.annotate("payoff", (22, X - 22), textcoords="offset points",
                xytext=(10, 8), fontsize=8, color=C_PAY)
    a2.annotate("value", (78, gbsm(78, X, T, r, r, sig, False)),
                textcoords="offset points", xytext=(-2, -14), ha="right",
                fontsize=8, color=C_VAL)
    a2.set_title("European put", loc="left", pad=6)

    for ax in (a1, a2):
        ax.axvline(X, color=MUTED, lw=1.0, ls=(0, (2, 2)))
        ax.set_xlabel("underlying price $S$")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 105)
        ax.grid(axis="y")
    a1.set_ylabel("value")

    fig.suptitle("$X=100$, $r=8\\%$, $T=1$, $\\sigma=30\\%$", x=0.012, y=1.04,
                 ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig("value-vs-payoff.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote value-vs-payoff.png")
    for s in (30, 50):
        print(f"  put at S={s}: value={gbsm(s, X, T, r, r, sig, False):.2f} "
              f"payoff={max(0, X - s):.2f}")
    print(f"  X(1-exp(-rT)) = {gap:.4f}")


if __name__ == "__main__":
    payoffs()
    value_vs_payoff()
