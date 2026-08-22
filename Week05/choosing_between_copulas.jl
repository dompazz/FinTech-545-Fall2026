# Choosing Between the Copulas -- Week 05
#
# Fit a Gaussian copula and a t copula to the same five assets, using the same
# margins and the same correlation matrix for both, then choose between them
# with AICc and BIC.
#
# The two models differ by exactly one free parameter, nu, which is what makes
# the comparison clean.

using DataFrames
using CSV
using Distributions
using StatsBase
using LinearAlgebra
using LoopVectorization
using JuMP
using Ipopt
using Printf

include("../Week04/return_calculate.jl")
include("fitted_model.jl")
include("simulate.jl")

stocks = ["SPY", "AAPL", "MSFT", "JPM", "XOM"]

prices = CSV.read("DailyPrices.csv", DataFrame)
returns = return_calculate(prices, dateColumn="Date")[!, stocks]

m = size(returns, 1)      # observations
n = length(stocks)        # variables

# --- Step 1: fit a generalized t to each margin, transform to U -----------
fittedModels = Dict{String,FittedModel}()
U = DataFrame()
for s in stocks
    fittedModels[s] = fit_general_t(returns[!, s])
    U[!, s] = fittedModels[s].u
end
Umat = Matrix(U)

println("\nMarginal fits")
for s in stocks
    @printf("  %-5s nu = %6.2f\n", s, dof(fittedModels[s].errorModel.ρ))
end

# --- Step 2: R from Kendall's tau ----------------------------------------
# rho = sin(pi*tau/2) is exact for any elliptical copula, so the same R serves
# both models. tau is rank based, so it does not matter whether it is computed
# on the returns or on U.
tau = corkendall(Umat)
R = sin.(pi .* tau ./ 2)
for i in 1:n
    R[i, i] = 1.0
end

# The tau matrix is PSD on complete data. The entrywise sin() transform is not
# guaranteed to preserve that, so check and repair. Higham returns a PSD matrix,
# which can still be singular, so nudge it to positive definite afterwards --
# the copula densities below need a Cholesky root.
function fix_correlation(R; tol=1e-8, ridge=1e-8)
    minEig = minimum(eigvals(R))
    if minEig < -tol
        @printf("R is not PSD (min eigenvalue %.6f). Repairing.\n", minEig)
        R = higham_nearestPSD(R)
    else
        @printf("R is PSD. Min eigenvalue %.6f\n", minEig)
    end
    # Higham can return a matrix that is asymmetric in the last bits and only
    # just PSD. Both break the Cholesky the copula densities need.
    R = (R + R') ./ 2
    if minimum(eigvals(R)) < ridge
        R = (R + ridge * I) ./ (1 + ridge)
    end
    return R
end

R = fix_correlation(R)

# --- Step 3: copula log likelihoods --------------------------------------
# A copula density is the joint density divided by the product of the marginal
# densities of the transformed variables.

function gaussian_copula_ll(U, R)
    Z = quantile.(Normal(), U)
    mvn = MvNormal(R)
    joint = sum(logpdf(mvn, Z[k, :]) for k in 1:size(Z, 1))
    margins = sum(logpdf.(Normal(), Z))
    return joint - margins
end

function t_copula_ll(U, R, nu)
    td = TDist(nu)
    T = quantile.(td, U)
    mvt = MvTDist(nu, Matrix(R))
    joint = sum(logpdf(mvt, T[k, :]) for k in 1:size(T, 1))
    margins = sum(logpdf.(td, T))
    return joint - margins
end

# --- Step 4: profile nu ---------------------------------------------------
# Search on theta = 1/nu so the grid resolution goes where it matters. The
# bounds put nu in [2, 100]; past 100 the t copula and the Gaussian are
# indistinguishable at any sample size we will have.
thetas = range(0.01, 0.49, length=200)
lls = [t_copula_ll(Umat, R, 1.0 / th) for th in thetas]
i = argmax(lls)

# refine between the neighbours of the grid maximum
lo = thetas[max(i - 1, 1)]
hi = thetas[min(i + 1, length(thetas))]
fine = range(lo, hi, length=200)
fineLls = [t_copula_ll(Umat, R, 1.0 / th) for th in fine]
j = argmax(fineLls)

nuhat = 1.0 / fine[j]
llT = fineLls[j]
llG = gaussian_copula_ll(Umat, R)

# --- Step 5: choose -------------------------------------------------------
aicc(ll, k, m) = -2 * ll + 2 * k + (2 * k^2 + 2 * k) / (m - k - 1)
bic(ll, k, m) = k * log(m) - 2 * ll

kG, kT = 0, 1
dBIC = 2 * (llT - llG) - log(m)

println("\nCopula comparison, m = $m observations, n = $n variables")
@printf("  %-22s %12s %12s\n", "", "Gaussian", "t")
@printf("  %-22s %12.2f %12.2f\n", "Log likelihood", llG, llT)
@printf("  %-22s %12d %12d\n", "Free parameters", kG, kT)
@printf("  %-22s %12s %12.1f\n", "Fitted nu", "--", nuhat)
@printf("  %-22s %12.2f %12.2f\n", "AICc", aicc(llG, kG, m), aicc(llT, kT, m))
@printf("  %-22s %12.2f %12.2f\n", "BIC", bic(llG, kG, m), bic(llT, kT, m))
@printf("\n  delta BIC = %.2f  -->  choose the %s copula\n",
        dBIC, dBIC > 0 ? "t" : "Gaussian")

# --- Tail dependence implied by the fit ----------------------------------
lower_tail_dep(nu, rho) = 2 * cdf(TDist(nu + 1), -sqrt((nu + 1) * (1 - rho) / (1 + rho)))

println("\nImplied lower tail dependence at nu = $(round(nuhat,digits=1))")
println("  (every one of these is 0 under the Gaussian copula)")
for i in 1:n, j in (i+1):n
    @printf("  %-5s %-5s rho = %6.3f   lambda = %.4f\n",
            stocks[i], stocks[j], R[i, j], lower_tail_dep(nuhat, R[i, j]))
end

# --- The profile is flat near its top ------------------------------------
println("\nProfile of the t copula log likelihood")
for nu in [4.0, 6.0, 8.0, 12.0, 14.0, 16.0, 20.0, 30.0, 50.0]
    @printf("  nu = %5.1f   ll = %8.3f\n", nu, t_copula_ll(Umat, R, nu))
end
