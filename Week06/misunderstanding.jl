using LinearAlgebra
using Distributions
using Random
using QuadGK
using CSV 
using DataFrames
using Plots
using StatsPlots

function VaR(a; alpha=0.05)
    x = sort(a)
    nup = convert(Int64,ceil(size(a,1)*alpha))
    ndn = convert(Int64,floor(size(a,1)*alpha))
    v = 0.5*(x[nup]+x[ndn])

    return -v
end

function ES(a; alpha=0.05)
    x = sort(a)
    nup = convert(Int64,ceil(size(a,1)*alpha))
    ndn = convert(Int64,floor(size(a,1)*alpha))
    v = 0.5*(x[nup]+x[ndn])
    
    es = mean(x[x.<=v])
    return -es
end

function VaR(d::T; alpha=0.05) where T <: UnivariateDistribution
    -quantile(d,alpha)
end

function ES(d::T; alpha=0.05) where T <: UnivariateDistribution
    v = VaR(d;alpha=alpha)
    f(x) = x*pdf(d,x)
    st = quantile(d,1e-12)
    return -quadgk(f,st,-v)[1]/alpha
end

function gen_data()
    function gbsm(call::Bool, underlying, strike, ttm, rf, b, ivol)
        d1 = (log(underlying/strike) + (b+ivol^2/2)*ttm)/(ivol*sqrt(ttm))
        d2 = d1 - ivol*sqrt(ttm)

        if call
            return underlying * exp((b-rf)*ttm) * cdf(Normal(),d1) - strike*exp(-rf*ttm)*cdf(Normal(),d2)
        else
            return strike*exp(-rf*ttm)*cdf(Normal(),-d2) - underlying*exp((b-rf)*ttm)*cdf(Normal(),-d1)
        end
        return nothing
    end

    initial = gbsm(true,100,100,1,0.05,0.05,0.2)

    u = Vector{Float64}(undef,10000)
    s = Vector{Float64}(undef,10000)
    p = Vector{Float64}(undef,10000)

    for i in 1:10000
        u[i] = rand(LogNormal(log(100) + (25/255)*0.05 - 0.5*(.07)^2,.07),1)[1]
        if u[i] > 100
            s[i] = .2 
        else
            s[i] = .4
        end
        p[i] = gbsm(false,u[i],100,1-25/255,0.05,0.05,s[i]) - initial
    end

    CSV.write("misunderstanding.csv",DataFrame(p=p))
end

p = CSV.read("misunderstanding.csv",DataFrame).p

density(p,label="Empirical",lw=2)

mn, mx = extrema(p)
prng = (mn-5):0.01:(mx+2)

plot!(prng,pdf(Normal(mean(p),std(p)),prng),label="Normal",lw=2)


VaR(p,alpha=0.05)
VaR(Normal(mean(p),std(p)),alpha=0.05)

ES(p,alpha=0.05)
ES(Normal(mean(p),std(p)),alpha=0.05)
