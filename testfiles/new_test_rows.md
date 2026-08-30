# Rows to add to `Tests.xlsx`

Two new blocks. Test 7.5 extends the existing distribution-fitting block; test 13
is the Week 05 multivariate $t$ and $t$ copula material.

## Tests 7.5 and 7.6 — NIG fits

Same 1,000 observations, drawn from a NIG with $\mu=0.02$, $\alpha=40$,
$\beta=-8$, $\delta=0.05$. Two fits, two criteria.

| Test | Description | Input | Expected Output |
|:--|:--|:--|:--|
| 7.5 | Fit a Normal Inverse Gaussian by the method of moments. Returns mu, alpha, beta, delta. | `test7_5.csv` | `testout7_5.csv` |
| 7.6 | Fit the same NIG by maximum likelihood. Returns mu, alpha, beta, delta. | `test7_5.csv` | `testout7_6.csv` |

**Parameterization.** Both expected-output files use the
$(\mu, \alpha, \beta, \delta)$ convention. `scipy.stats.norminvgauss` uses
$(a, b, \mathrm{loc}, \mathrm{scale})$ with $a = \alpha\delta$ and
$b = \beta\delta$, so a student working in Python converts as

```
delta = scale;  alpha = a/delta;  beta = b/delta;  mu = loc
```

A direct `scipy.stats.norminvgauss.fit()` call on `test7_5.csv`, converted this
way, reproduces `testout7_6.csv` exactly — verified against scipy 1.18.1.

**One caution on 7.6.** `norminvgauss.fit` is a numerical optimizer, not a closed
form, so a different scipy version could move the trailing digits. Mark it on a
relative tolerance rather than on an exact match. 7.5 is closed form and does not
have this exposure.

## Test 13 — Multivariate t and the t copula

All nine cases read `test13_returns.csv`, 250 observations of 5 assets drawn from a
multivariate $t$ with $\nu = 6$. Case 13.9 also reads `test13_portfolio.csv`.

| Test | Description | Input | Expected Output |
|:--|:--|:--|:--|
| 13.1 | Correlation from Kendall's tau, rho = sin(pi*tau/2), repaired to positive definite | `test13_returns.csv` | `testout13_1.csv` |
| 13.2 | Multivariate t fit — the fitted mean vector | `test13_returns.csv` | `testout13_2.csv` |
| 13.3 | Multivariate t fit — the fitted scale matrix S (note cov = nu/(nu-2)*S) | `test13_returns.csv` | `testout13_3.csv` |
| 13.4 | Multivariate t fit — nu by profile, and the log likelihood | `test13_returns.csv` | `testout13_4.csv` |
| 13.5 | Gaussian copula log likelihood, generalized t margins | `test13_returns.csv` | `testout13_5.csv` |
| 13.6 | t copula — nu by profile, and the log likelihood | `test13_returns.csv` | `testout13_6.csv` |
| 13.7 | Choose between the copulas by AICc and BIC | `test13_returns.csv` | `testout13_7.csv` |
| 13.8 | Lower tail dependence for each pair under the fitted t copula | `test13_returns.csv` | `testout13_8.csv` |
| 13.9 | Portfolio VaR and ES under the t copula (same shape as 9.1, which is the Gaussian copula) | `test13_returns.csv`, `test13_portfolio.csv` | `testout13_9.csv` |

## What the answers come out to

The two NIG fits, against the truth they were drawn from. Each wins on its own
criterion, which is the comparison worth putting in front of the class:

| | true | 7.5 moments | 7.6 MLE |
|:--|--:|--:|--:|
| mu | 0.0200 | 0.0239 | 0.0197 |
| alpha | 40.000 | 52.947 | 46.924 |
| beta | -8.000 | -12.245 | -8.233 |
| delta | 0.0500 | 0.0575 | 0.0526 |
| log likelihood | | 1974.11 | **1974.54** |
| sample skewness -0.40329 | | **-0.40329** | -0.33763 |
| sample kurtosis 1.23044 | | **1.23044** | 1.38643 |

The MLE reaches the higher log likelihood. The method of moments reproduces the
sample's third and fourth moments exactly. Neither is wrong.

Worth knowing when marking, and a quick sanity check that a student's
implementation is in the right place rather than merely running.

- 13.4: nu = 5.82, log likelihood 3811.28. The data was generated at nu = 6.
- 13.6: t copula nu = 5.95, log likelihood 128.70.
- 13.5: Gaussian copula log likelihood 98.65.
- 13.7: BIC -251.88 for the t copula against -197.30 for the Gaussian, so the t
  copula wins by a wide margin. That is the point of the case.
- 13.8: lambda runs 0.08 to 0.13 across the pairs. Every one of these is exactly
  0 under the Gaussian copula, whatever the correlation.
- 13.9: total portfolio VaR95 163.00, ES95 228.57 on 10,000 of value.
