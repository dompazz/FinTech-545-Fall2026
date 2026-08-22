# A Worked Fit -- Week 05
#
# Fit a multivariate t to five assets by the rank based route: mu at the sample
# mean, R from Kendall's tau, the scales pinned to nu by the sample standard
# deviations, and nu itself by a one dimensional profile.
#
# This is the fit in the "Fitting the Multivariate t" section. The copula
# comparison that follows it is in choosing_between_copulas.jl.

using DataFrames
using CSV
using Distributions
using StatsBase
using LinearAlgebra
using LoopVectorization
using Random
using JuMP
using Ipopt
using Printf

include("../Week04/return_calculate.jl")
include("fitted_model.jl")
include("simulate.jl")

stocks = ["SPY", "AAPL", "MSFT", "JPM", "XOM"]

prices = CSV.read("DailyPrices.csv", DataFrame)
returns = return_calculate(prices, dateColumn="Date")[!, stocks]

X = Matrix(returns)
m, n = size(X)

# --- Step 1: the sample moments ------------------------------------------
mu = vec(mean(X, dims=1))
sd = vec(std(X, dims=1))

# A t fitted to each series on its own. Not used by the multivariate fit --
# it is there to show what one shared nu has to stand in for.
println("\nSample moments, and a t fitted to each series separately")
@printf("  %-6s %9s %9s %12s\n", "Asset", "Mean (%)", "Sd (%)", "Separate nu")
for (j, s) in enumerate(stocks)
    fm = fit_general_t(X[:, j])
    @printf("  %-6s %9.3f %9.3f %12.2f\n",
            s, 100 * mu[j], 100 * sd[j], dof(fm.errorModel.ρ))
end

# --- Step 2: R from Kendall's tau ----------------------------------------
# rho = sin(pi*tau/2) is exact for any elliptical distribution, and nu does not
# appear in it, so R can be estimated once and held fixed while nu is searched.
tau = corkendall(X)
R = sin.(pi .* tau ./ 2)
for i in 1:n
    R[i, i] = 1.0
end

# The tau matrix is PSD on complete data. The entrywise sin() transform is not
# guaranteed to preserve that, so check and repair. Higham returns a PSD matrix,
# which can still be singular, so nudge it to positive definite afterwards.
function fix_correlation(R; tol=1e-8, ridge=1e-8)
    minEig = minimum(eigvals(R))
    if minEig < -tol
        @printf("R is not PSD (min eigenvalue %.6f). Repairing.\n", minEig)
        R = higham_nearestPSD(R)
    else
        @printf("\nR is PSD. Min eigenvalue %.4f\n", minEig)
    end
    R = (R + R') ./ 2
    if minimum(eigvals(R)) < ridge
        R = (R + ridge * I) ./ (1 + ridge)
    end
    return R
end

R = fix_correlation(R)

println("\nR from Kendall's tau")
@printf("  %-6s", "")
for s in stocks
    @printf("%9s", s)
end
println()
for i in 1:n
    @printf("  %-6s", stocks[i])
    for j in 1:n
        j < i ? @printf("%9.3f", R[i, j]) : @printf("%9s", "")
    end
    println()
end

# --- Step 3: profile nu ---------------------------------------------------
# A multivariate t carries a scale matrix, not a covariance. cov = nu/(nu-2)*S,
# so at a given nu the scale of variable i has to be sd_i*sqrt((nu-2)/nu), and
# every parameter of the distribution becomes a function of nu alone.
function mvt_ll(nu)
    d = sd .* sqrt((nu - 2) / nu)
    S = (d * d') .* R
    dist = MvTDist(nu, mu, Matrix(S))
    return sum(logpdf(dist, X[k, :]) for k in 1:m)
end

# Search on theta = 1/nu so the resolution goes where it matters. The interval
# stops at 0.49 rather than 0.5 to keep nu off its lower bound, where the scale
# above collapses to zero and S stops being invertible.
thetas = range(0.01, 0.49, length=200)
lls = [mvt_ll(1.0 / th) for th in thetas]
i = argmax(lls)

lo = thetas[max(i - 1, 1)]
hi = thetas[min(i + 1, length(thetas))]
fine = range(lo, hi, length=200)
fineLls = [mvt_ll(1.0 / th) for th in fine]
j = argmax(fineLls)

nuhat = 1.0 / fine[j]

println("\nThe log likelihood as a function of nu")
for nu in [4.0, 6.0, nuhat, 10.0, 15.0, 20.0, 40.0]
    @printf("  nu = %5.1f   ll = %9.1f%s\n", nu, mvt_ll(nu), nu == nuhat ? "   <- max" : "")
end

@printf("\nnu = %.1f  (m = %d observations, n = %d variables)\n", nuhat, m, n)

# --- Simulating from the fit ---------------------------------------------
# The mixture representation: one chi-square draw scales the whole normal
# vector, and that common shock is what puts weight in the joint tail.
function simulate_mvt(mu, S, nu, nsim; seed=1234)
    Z = simulate_pca(S, nsim; seed=seed)
    Random.seed!(seed + 1)
    w = nu ./ rand(Chisq(nu), nsim)
    return sqrt.(w) .* Z .+ mu'
end

d = sd .* sqrt((nuhat - 2) / nuhat)
S = (d * d') .* R
sim = simulate_mvt(mu, S, nuhat, 100000)

# Compare against the package's own multivariate t. If these agree, use
# whichever you are more comfortable with.
Random.seed!(99)
sim2 = Matrix(rand(MvTDist(nuhat, mu, Matrix(S)), 100000)')

println("\nMixture simulation against the built in draw, 100,000 iterations")
@printf("  %-6s %10s %10s %10s %10s\n", "Asset", "sd mix", "sd built", "kurt mix", "kurt built")
for (jj, s) in enumerate(stocks)
    @printf("  %-6s %10.4f %10.4f %10.2f %10.2f\n", s,
            std(sim[:, jj]), std(sim2[:, jj]),
            kurtosis(sim[:, jj]), kurtosis(sim2[:, jj]))
end
