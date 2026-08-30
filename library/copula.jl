# Gaussian and t copulas -- densities, fitting, simulation, tail dependence.
#
# A copula density is the joint density divided by the product of the marginal
# densities of the transformed variables. The two models here differ by exactly
# one free parameter, nu, which is what makes the comparison clean.
#
# Requires fix_correlation, kendall_correlation, and profile_nu from
# multivariate_t.jl, and simulate_pca from simulate.jl.

using Distributions
using LinearAlgebra
using Random

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

# tau is rank based, so it does not matter whether it is computed on the
# returns or on U. The same R therefore serves both copulas.
function fit_gaussian_copula(U)
    R = kendall_correlation(U)
    return R, gaussian_copula_ll(U, R)
end

# The bounds put nu in [2, 100]; past 100 the t copula and the Gaussian are
# indistinguishable at any sample size we will have.
function fit_t_copula(U)
    R = kendall_correlation(U)
    nu, ll = profile_nu(nu -> t_copula_ll(U, R, nu))
    return R, nu, ll
end

function simulate_gaussian_copula(R, nsim; seed=1234)
    Z = simulate_pca(R, nsim; seed=seed)
    return cdf.(Normal(), Z)
end

# Same mixture shock as the multivariate t, then back through the t CDF. The
# common chi-square draw is what leaves dependence in the joint tail.
function simulate_t_copula(R, nu, nsim; seed=1234)
    Z = simulate_pca(R, nsim; seed=seed)
    Random.seed!(seed + 1)
    w = nu ./ rand(Chisq(nu), nsim)
    return cdf.(TDist(nu), sqrt.(w) .* Z)
end

# Every one of these is 0 under the Gaussian copula, whatever the correlation.
tail_dependence_t(rho, nu) = 2 * cdf(TDist(nu + 1), -sqrt((nu + 1) * (1 - rho) / (1 + rho)))

copula_aicc(ll, k, m) = -2 * ll + 2 * k + (2 * k^2 + 2 * k) / (m - k - 1)
copula_bic(ll, k, m) = k * log(m) - 2 * ll
