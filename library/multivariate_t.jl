# Multivariate t -- fitting and simulation.
#
# The rank based route from Week 05: mu at the sample mean, R from Kendall's
# tau, the scales pinned to nu by the sample standard deviations, and nu itself
# by a one dimensional profile.
#
# Requires higham_nearestPSD and simulate_pca from simulate.jl.

using Distributions
using LinearAlgebra
using Random
using StatsBase

# The tau matrix is PSD on complete data. The entrywise sin() transform is not
# guaranteed to preserve that, so check and repair. Higham returns a PSD matrix,
# which can still be singular, so nudge it to positive definite afterwards --
# a singular R breaks the Cholesky root the copula densities need.
function fix_correlation(R; tol=1e-8, ridge=1e-8)
    if minimum(eigvals(R)) < -tol
        R = higham_nearestPSD(R)
    end
    # Higham can return a matrix that is asymmetric in the last bits and only
    # just PSD. Both break the Cholesky.
    R = (R + R') ./ 2
    if minimum(eigvals(R)) < ridge
        R = (R + ridge * I) ./ (1 + ridge)
    end
    return R
end

min_eigenvalue(R) = minimum(eigvals(R))

# rho = sin(pi*tau/2) is exact for any elliptical distribution, and nu does not
# appear in it, so R can be estimated once and held fixed while nu is searched.
function kendall_correlation(X)
    n = size(X, 2)
    R = sin.(pi .* corkendall(X) ./ 2)
    for i in 1:n
        R[i, i] = 1.0
    end
    return fix_correlation(R)
end

# A multivariate t carries a scale matrix, not a covariance. cov = nu/(nu-2)*S,
# so at a given nu the scale of variable i has to be sd_i*sqrt((nu-2)/nu), and
# every parameter of the distribution becomes a function of nu alone.
function mvt_scale(sd, R, nu)
    d = sd .* sqrt((nu - 2) / nu)
    return (d * d') .* R
end

function mvt_loglikelihood(X, mu, sd, R, nu)
    dist = MvTDist(nu, mu, Matrix(mvt_scale(sd, R, nu)))
    return sum(logpdf(dist, X[k, :]) for k in 1:size(X, 1))
end

# Profile nu on theta = 1/nu so the grid resolution goes where it matters. The
# interval stops at 0.49 rather than 0.5 to keep nu off its lower bound, where
# the scale above collapses to zero and S stops being invertible.
function profile_nu(ll; lo=0.01, hi=0.49, n=200)
    thetas = range(lo, hi, length=n)
    lls = [ll(1.0 / th) for th in thetas]
    i = argmax(lls)

    # refine between the neighbours of the grid maximum
    fine = range(thetas[max(i - 1, 1)], thetas[min(i + 1, n)], length=n)
    fineLls = [ll(1.0 / th) for th in fine]
    j = argmax(fineLls)

    return 1.0 / fine[j], fineLls[j]
end

# Returns the fitted mean, scale matrix, degrees of freedom, and log likelihood.
function fit_multivariate_t(X)
    mu = vec(mean(X, dims=1))
    sd = vec(std(X, dims=1))
    R = kendall_correlation(X)

    nu, ll = profile_nu(nu -> mvt_loglikelihood(X, mu, sd, R, nu))

    return mu, mvt_scale(sd, R, nu), nu, ll
end

# The mixture representation: one chi-square draw scales the whole normal
# vector, and that common shock is what puts weight in the joint tail.
function simulate_mvt(mu, S, nu, nsim; seed=1234)
    Z = simulate_pca(S, nsim; seed=seed)
    Random.seed!(seed + 1)
    w = nu ./ rand(Chisq(nu), nsim)
    return sqrt.(w) .* Z .+ mu'
end
