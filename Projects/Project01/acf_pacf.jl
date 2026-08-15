using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
using Plots
using StatsPlots
using LinearAlgebra
using JuMP
using Ipopt
using Dates
using ForwardDiff
using FiniteDiff
using CSV
using LoopVectorization
using Roots
using QuadGK
using StateSpaceModels

function simulate_ar_process(n, θ1, θ2, θ3; burnin=100)
    # ε = randn(n + burnin)
    # y = zeros(n)
    # for t in (burnin+1):n+burnin
    #     y[t-3] = ε[t] + θ1 * ε[t-1] + θ3 * ε[t-3]
    # end

    y = Vector{Float64}(undef,n)

    yt_last = fill(1.0,3)
    d = Normal(0,0.1)
    ε = rand(d,n+burnin)
    
    for i in 1:(n+burnin)
        y_t = 1.0 + θ1*yt_last[1] +θ2*yt_last[2] + θ3*yt_last[3] + ε[i]
        yt_last = [y_t, yt_last[1], yt_last[2]]
        if i > burnin
            y[i-burnin] = y_t
        end
    end
    return y
end

function simulate_ma_process(n, θ1, θ2, θ3; burnin=100)
    # ε = randn(n + burnin)
    # y = zeros(n)
    # for t in (burnin+1):n+burnin
    #     y[t-3] = ε[t] + θ1 * ε[t-1] + θ3 * ε[t-3]
    # end

    y = Vector{Float64}(undef,n)

    d = Normal(0,0.1)
    e = rand(d,n+burnin)

    for i in 4:(n+burnin)
        global yt_last
        y_t = 1.0 + θ1*e[i-1] + θ2*e[i-2] + θ3*e[i-3] + e[i]
        if i > burnin
            y[i-burnin] = y_t
        end
    end

    return y
end

# y = simulate_ar_process(1000, .5, -.2, .2; burnin=100)
y = simulate_ma_process(1000, .5, -.2, -.2; burnin=100)

p1 = plot(autocor(y)[1:10], title = "ACF of MA(3)", legend = false,seriestype="bar")
p2 = plot(pacf(y,1:10), title = "PACF of the MA(3)", legend = false,seriestype="bar")
p = plot(p1,p2,layout=(2,1))