# Assignment 2 — Covariance, VaR, and Copulas

Covers Weeks 1 through 5, weighted toward Weeks 3 to 5. Ten points: eight written,
two code.

## Files

| File | What it is |
|:--|:--|
| `assignment.qmd` | The assignment source. Renders to `Assignment 2.pdf`. |
| `solutions.qmd` | The answer key. Renders to `Assignment 2 - Solutions.pdf`. |
| `generate_data.jl` | Writes the five problem CSVs. Each block seeds its own RNG. |
| `solution.jl` | Produces every number quoted in the answer key, and the figures. |
| `problem1.csv` … `problem5.csv` | Student data. |

## Running

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. generate_data.jl     # only if the CSVs need rebuilding
julia --project=. solution.jl
```

`solution.jl` includes `fitted_model.jl`, `simulate.jl`, `RiskStats.jl`,
`ewCov.jl`, `multivariate_t.jl`, and `copula.jl` from `../../library/`, so run it
from this directory.

Rendering:

```
quarto render assignment.qmd --to pdf
quarto render solutions.qmd --to pdf
```

## What is planted in the data

Regenerating with a different seed, or on a different Julia version, will break
the answer key. In Problem 1 it will likely break the question itself.

- **Problem 1** -- four stocks and an index built from them at weights 0.40, 0.30,
  0.20, 0.10, plus a tracking error of 0.05% a day. The true matrix is close to
  singular. Stocks trade on mismatched calendars and `D` has only the last 100
  days. The pairwise correlation matrix has an eigenvalue of -0.011, and the
  negative direction is the tracking portfolio. Only about 45% of seeds produce a
  negative eigenvalue with this design. Seed 5463 was chosen because it does.
- **Problem 2** -- 460 days at 1% daily volatility, then 40 at 2.5%. Normal within
  each regime. The pooled excess kurtosis of 2.1 comes entirely from the mixture.
- **Problem 3** -- two independent bonds bought at 90, 4% default probability,
  recovery Beta(8, 12) times 100. Realized default rates 4.06% and 3.94%, 7.82% for
  at least one. VaR fails subadditivity at 5% and holds at 1%.
- **Problem 4** -- $t$ copula at $\nu = 4$, margins normal, $t_5$, $t_7$. Fitted
  copula $\hat{\nu} = 3.66$, $\Delta BIC = 196$. The joint tail counts at 2.5% in
  both tails land on the $t$ (48 in the data, 48.3 implied, 29.1 under the
  Gaussian). The lower tail alone at 5% does not separate the models, which is why
  the question uses 2.5% and both tails. `X2` has one -8.8 sd day that produces a
  sample skewness of -0.61. The key addresses it rather than hiding it.
- **Problem 5** -- betas 1.05 and 0.95, residual correlation 0.65 (0.70 realized).
  A diagonal residual covariance understates `P1` by 12% and overstates `P2` by 80%.
