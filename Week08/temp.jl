
d = NormalInverseGaussian(0.0, 0.5, 0.2, 0.1)

x = rand(d, 100000)

using Distributions, Optimization, OptimizationOptimJL, ForwardDiff
using Optim


# re‐param and neg‐loglik as before
function unpack(θ)
    μ   = θ[1]
    α   = exp(θ[2])
    β   = α * tanh(θ[3])
    δ   = exp(θ[4])
    return μ, α, β, δ
end

function negloglik(θ, data)
    μ, α, β, δ = unpack(θ)
    d = NormalInverseGaussian(μ, α, β, δ)
    return -sum(logpdf.(d, data))
end

x = rand(Normal(0.05,.02),100000)

# initial guess
μ0, σ0 = mean(x), std(x)
θ0 = [μ0, log(1/σ0), atanh(0.0), log(σ0)]

# bounds on θ: let δ ∈ [σ0/10, 10σ0]
lower = [-Inf, -10, -10, -10]
upper = [ Inf,  10,  10, 10 ]

# optimize with box‐constraints and forward‐mode autodiff
result = optimize(
    θ -> negloglik(θ, x),
    lower, upper, θ0,
    Fminbox(LBFGS()),      # wrap BFGS in a box‐constraint solver
    autodiff = :forward    # ForwardDiff for gradients
)

θ̂ = result.minimizer
μ̂, α̂, β̂, δ̂ = unpack(θ̂)

println("Fitted NIG parameters:")
println("μ = $μ̂, α = $α̂, β = $β̂, δ = $δ̂")


using Distributions
using Plots
using StatsPlots
using CSV
using DataFrames

df = CSV.read("c:/temp/Rates_Index.csv", DataFrame)
df_wide = unstack(df, :date, :symbol, :PX_LAST)
sort!(df_wide, :date)
filter!(row -> !any(ismissing, row), df_wide)
rename!(df_wide, [Symbol("LUACOAS Index") => :Spreads, Symbol("USGG10YR Index") => :Rates])
df_wide.Spreads = Float64.(df_wide.Spreads)
df_wide.Rates = Float64.(df_wide.Rates)


density(df_wide.Spreads, label="", title="Credit Spreads")
savefig("c:/temp/Credit_Spreads.svg")
density(df_wide.Rates, label="", title="Interest Rates")
savefig("c:/temp/Interest_Rates.svg")


plot(df_wide.Spreads, df_wide.Rates,
     title="Credit Spreads vs Interest Rates",
     seriestype=:scatter,
     legend=false)
savefig("c:/temp/Spreads_Rates.svg")

df = CSV.read("c:/temp/agg_Index.csv", DataFrame)
x = df[2:end, :PX_LAST] ./ df[1:(end-1), :PX_LAST] .- 1

density(x, label="", title="BBG Aggregate Index Returns")
density!(rand(Normal(mean(x), std(x)),100000), label="Normal Distribution", color=:red, linestyle=:dash)
savefig("c:/temp/Agg_Index_Returns.svg")
