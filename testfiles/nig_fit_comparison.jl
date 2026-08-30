# fit_NIG_mle against fit_nig_moments, on identical data.
#
# Both fits are in the test corpus -- 7.5 is the method of moments and 7.6 is
# the MLE. This script is the side by side view, which the CSVs do not give you.
# It needs the same Python toolchain test_setup.jl now needs; see the setup note
# at the top of test_setup.jl.
#
#   julia --project=. nig_fit_comparison.jl
#
# The two fits optimize different things and the output shows it. The method of
# moments reproduces the sample moments exactly and the MLE does not. The MLE
# reaches the higher log likelihood and the method of moments does not. Each
# wins on its own criterion, which is the point worth making to a class.

using CSV, DataFrames, Distributions, StatsBase, Random, LinearAlgebra
using Interpolations, QuadGK, JuMP, Ipopt, PyCall, Printf

include("../library/skewNormal.jl")
include("../library/fitted_model.jl")

x = CSV.read("data/test7_5.csv", DataFrame)[:, 1]
truth = NormalInverseGaussian(0.02, 40.0, -8.0, 0.05)   # how test7_5.csv was drawn

mle = fit_NIG_mle(x)
mm = fit_nig_moments(x)

println("Both return a FittedModel")
@printf("  %-18s %10s %10s\n", "", "MLE", "moments")
@printf("  %-18s %10s %10s\n", "isa FittedModel", mle isa FittedModel, mm isa FittedModel)
@printf("  %-18s %10s %10s\n", "beta", mle.beta === nothing ? "nothing" : "set",
        mm.beta === nothing ? "nothing" : "set")
@printf("  %-18s %10d %10d\n", "length(u)", length(mle.u), length(mm.u))
@printf("  %-18s %10s %10s\n", "u in [0,1]", all(0 .<= mle.u .<= 1), all(0 .<= mm.u .<= 1))
@printf("  %-18s %10.2e %10.2e\n", "max |eval(u)-x|",
        maximum(abs.(mle.eval(mle.u) .- x)), maximum(abs.(mm.eval(mm.u) .- x)))

println("\nFitted parameters")
@printf("  %-8s %10s %10s %10s\n", "param", "true", "MLE", "moments")
for (nm, t, a, b) in [("mu", truth.μ, mle.errorModel.μ, mm.errorModel.μ),
                      ("alpha", truth.α, mle.errorModel.α, mm.errorModel.α),
                      ("beta", truth.β, mle.errorModel.β, mm.errorModel.β),
                      ("delta", truth.δ, mle.errorModel.δ, mm.errorModel.δ)]
    @printf("  %-8s %10.4f %10.4f %10.4f\n", nm, t, a, b)
end

println("\nFit statistics")
@printf("  %-18s %12.2f %12.2f\n", "log likelihood",
        sum(logpdf.(mle.errorModel, x)), sum(logpdf.(mm.errorModel, x)))
@printf("  %-18s %12.2f %12.2f\n", "AICc", aicc(mle, x), aicc(mm, x))

println("\nMoments: the sample, and what each fit implies")
@printf("  %-12s %12s %12s %12s\n", "moment", "sample", "MLE", "moments")
for (nm, f) in [("mean", mean), ("std", std), ("skewness", skewness), ("kurtosis", kurtosis)]
    @printf("  %-12s %12.5f %12.5f %12.5f\n", nm, f(x), f(mle.errorModel), f(mm.errorModel))
end
