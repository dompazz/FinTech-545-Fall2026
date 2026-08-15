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
using Printf

include("../../library/RiskStats.jl")
include("../../library/simulate.jl")
include("../../library/return_calculate.jl")
include("../../library/fitted_model.jl")
include("../../library/missing_cov.jl")
include("../../library/ewCov.jl")
include("../../library/expost_factor.jl")
include("../../library/optimizers.jl")
include("../../library/bt_american.jl")
include("../../library/gbsm.jl")

theme(:dark)

#problem 1
# Using DailyPrices.csv, and the stocks SPY, AAPL, and EQIX,
# calculate the arithmetic and geometric returns
# Remove the mean such that each series has 0 mean.  Present the last 5 
# rows of each method for each stock.  Also present the Standard Deviation.


df = CSV.read("Project02/DailyPrices.csv",DataFrame)[!,["Date","SPY","AAPL","EQIX"]] 
stocks = ["SPY", "AAPL", "EQIX"]
arets = return_calculate(df;dateColumn="Date")
lrets = return_calculate(df;dateColumn="Date",method="log")

for s in stocks
    m = mean(arets[!,s])
    arets[!,s] = arets[!,s] .- m
    m = mean(lrets[!,s])
    lrets[!,s] = lrets[!,s] .- m
end

println(last(arets,5))
# Arithmetic Returns: 
# 5×4 DataFrame
#  Row │ Date        SPY          AAPL         EQIX
#      │ Date        Float64      Float64      Float64
# ─────┼────────────────────────────────────────────────────
#    1 │ 2024-12-27  -0.0114918   -0.0146777   -0.00696614
#    2 │ 2024-12-30  -0.0123769   -0.0146989   -0.00806363
#    3 │ 2024-12-31  -0.00460328  -0.0084934    0.0065122
#    4 │ 2025-01-02  -0.00342227  -0.0276714    0.000496849
#    5 │ 2025-01-03   0.0115382   -0.00344497   0.0157454
println(std.([arets[!,s] for s in stocks]))
# [0.008076750009417146, 0.01348287725355314, 0.015360573162517398]

println(last(lrets,5))
println(std.([lrets[!,s] for s in stocks]))
# Log Returns
# 5×4 DataFrame
#  Row │ Date        SPY          AAPL         EQIX
#      │ Date        Float64      Float64      Float64
# ─────┼────────────────────────────────────────────────────
#    1 │ 2024-12-27  -0.0115146   -0.0146748   -0.00686731
#    2 │ 2024-12-30  -0.0124095   -0.0146964   -0.00797208
#    3 │ 2024-12-31  -0.00457691  -0.0084271    0.00660184
#    4 │ 2025-01-02  -0.00339229  -0.0279304    0.000612992
#    5 │ 2025-01-03   0.0114936   -0.00335567   0.0157251
# [0.00807822211819445, 0.013446433597619272, 0.015270234889095817]

#problem 2
# Using DatilyPrices.csv.  You have a portfolio of 
#  - 100 Shares of SPY
#  - 200 Shares of AAPL
#  - 150 Shares of EQIX
# a. Calculate the current value of the portfolio given today is 2025-01-03
# b. Calculate VaR and ES at the 5% level and assuming arithmetic returns
#   for the each stock. Remove the mean such that each series has 0 mean.
#   Present values as $ loss.  Use the following methods for VaR and ES:
#    1. exponentially weighted covariance with λ=0.97
#    2. T distribution for each stock
#    3. Historic Simulation  
# c. Discuss the differences in VaR and ES.

function format_currency(value)
    # Format the number to two decimal places
    formatted = @sprintf "%.2f" value
    # Split into whole and decimal parts
    whole, decimal = split(formatted, ".")
    # Insert commas in the whole number part
    reversed_chunks = Iterators.partition(reverse(whole), 3)
    whole_with_commas = reverse(join(collect(reversed_chunks), ","))
    # Return formatted currency string
    return "\$" * whole_with_commas * "." * decimal
end

portfolio = Dict{String,Float64}("SPY"=>100,"AAPL"=>200,"EQIX"=>150)
stocks = ["SPY", "AAPL", "EQIX"]

# a.
currentPrice = df[df.Date .== Date(2025,1,3),:]
currentValue = sum([portfolio[s]*currentPrice[!,s][1] for s in stocks])
println("Current Portfolio Value: $(format_currency(currentValue))")
# Current Portfolio Value: $251,862.50

# b.
# MV Normal with EW Covariance
covar =ewCovar(Matrix(arets[!,stocks]),0.97)
# 7.19328e-5  5.39026e-5   5.26658e-5
# 5.39026e-5  0.000139267  3.78231e-5
# 5.26658e-5  3.78231e-5   0.000153173

# Linear portfolio, so we can use Delta Normal for VaR and ES assuming MVNormal Returns.
pv = vcat([portfolio[s]*currentPrice[!,s][1] for s in stocks], currentValue)
w = pv[1:3]/currentValue
s = vcat(sqrt.(diag(covar)),sqrt(w'*covar*w))
pct_vars = VaR.(Normal.(0.0,s),alpha=0.05)
pct_es = ES.(Normal.(0.0,s),alpha=0.05)
dollar_var = pv.*pct_vars
dollar_es = pv.*pct_es
riskOut = DataFrame(:Stock=>vcat(stocks,"Total"),:Value=>format_currency.(pv), :DollarVaR=>format_currency.(dollar_var), :DollarES=>format_currency.(dollar_es))
# 4×4 DataFrame
#  Row │ Stock   Value        DollarVaR  DollarES  
#      │ String  String       String     String
# ─────┼───────────────────────────────────────────
#    1 │ SPY     $59,195.00   $825.80    $1,035.59
#    2 │ AAPL    $48,672.00   $944.78    $1,184.79
#    3 │ EQIX    $143,995.50  $2,931.34  $3,676.02
#    4 │ Total   $251,862.50  $3,856.32  $4,835.98

# T Distribution
# We will use the same portfolio value as above.
# Fit T Distributions 
portfolio = DataFrame(:stock=>stocks, :holding=>[100.,200.,150.])

models = Dict{String,FittedModel}()
for s in stocks
    m = fit_general_t(arets[!,s])
    models[s] = m
end

# Simulate via Gaussian Copula
nsim = 100000
U = DataFrame()
for s in stocks
    U[!,s] = models[s].u
end
C = corspearman(Matrix(U))
# 3×3 Matrix{Float64}:
#  1.0       0.624145  0.53103
#  0.624145  1.0       0.303198
#  0.53103   0.303198  1.0

simU = DataFrame(cdf(Normal(0,1),simulate_pca(C,nsim)),stocks)
simStates = DataFrame()
for s in stocks
    simStates[!,s] = models[s].eval.(simU[!,s]) 
end

iteration = [i for i in 1:nsim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))
nVals = size(values,1)
currentValue = Vector{Float64}(undef,nVals)
simulatedValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
for i in 1:nVals
    price = currentPrice[1,values.stock[i]]
    currentValue[i] = values.holding[i] * price
    simulatedValue[i] = values.holding[i] * price*(1.0+simStates[values.iteration[i],values.stock[i]])
    pnl[i] = simulatedValue[i] - currentValue[i]
end
values[!,:currentValue] = currentValue
values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl

riskOut = aggRisk(values,[:stock])[:,[:stock,:currentValue,:VaR95,:ES95]]
riskOut[!,:currentValue] = format_currency.(riskOut[!,:currentValue])
riskOut[!,:VaR95] = format_currency.(riskOut[!,:VaR95]) 
riskOut[!,:ES95] = format_currency.(riskOut[!,:ES95])
println(riskOut)
# 4×4 DataFrame
#  Row │ stock   currentValue  VaR95      ES95      
#      │ String  String        String     String
# ─────┼────────────────────────────────────────────
#    1 │ SPY     $59,195.00    $765.72    $1,031.87
#    2 │ AAPL    $48,672.00    $1,035.07  $1,473.04
#    3 │ EQIX    $143,995.50   $3,433.72  $4,957.45
#    4 │ Total   $251,862.50   $4,439.98  $6,209.04

# Historic Simulation
nhist = size(arets,1)
iteration = [i for i in 1:nhist]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))
nVals = size(values,1)
currentValue = Vector{Float64}(undef,nVals)
simulatedValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
for i in 1:nVals
    price = currentPrice[1,values.stock[i]]
    currentValue[i] = values.holding[i] * price
    simulatedValue[i] = values.holding[i] * price*(1.0+arets[values.iteration[i],values.stock[i]])
    pnl[i] = simulatedValue[i] - currentValue[i]
end
values[!,:currentValue] = currentValue
values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl

riskOut = aggRisk(values,[:stock])[:,[:stock,:currentValue,:VaR95,:ES95]]
riskOut[!,:currentValue] = format_currency.(riskOut[!,:currentValue])
riskOut[!,:VaR95] = format_currency.(riskOut[!,:VaR95]) 
riskOut[!,:ES95] = format_currency.(riskOut[!,:ES95])
println(riskOut)
# 4×4 DataFrame
#  Row │ stock   currentValue  VaR95      ES95      
#      │ String  String        String     String
# ─────┼────────────────────────────────────────────
#    1 │ SPY     $59,195.00    $873.32    $1,088.41
#    2 │ AAPL    $48,672.00    $1,082.07  $1,452.52
#    3 │ EQIX    $143,995.50   $3,672.82  $4,757.45
#    4 │ Total   $251,862.50   $4,580.03  $6,118.67

# c.
# Normal VaR and ES 
# 4×4 DataFrame
#  Row │ Stock   Value        DollarVaR  DollarES  
#      │ String  String       String     String
# ─────┼───────────────────────────────────────────
#    1 │ SPY     $59,195.00   $825.80    $1,035.59
#    2 │ AAPL    $48,672.00   $944.78    $1,184.79
#    3 │ EQIX    $143,995.50  $2,931.34  $3,676.02
#    4 │ Total   $251,862.50  $3,856.32  $4,835.98

# T VaR and ES
# 4×4 DataFrame
#  Row │ stock   currentValue  VaR95      ES95      
#      │ String  String        String     String
# ─────┼────────────────────────────────────────────
#    1 │ SPY     $59,195.00    $765.72    $1,031.87
#    2 │ AAPL    $48,672.00    $1,035.07  $1,473.04
#    3 │ EQIX    $143,995.50   $3,433.72  $4,957.45
#    4 │ Total   $251,862.50   $4,439.98  $6,209.04

# Historic VaR and ES
# 4×4 DataFrame
#  Row │ stock   currentValue  VaR95      ES95      
#      │ String  String        String     String
# ─────┼────────────────────────────────────────────
#    1 │ SPY     $59,195.00    $873.32    $1,088.41
#    2 │ AAPL    $48,672.00    $1,082.07  $1,452.52
#    3 │ EQIX    $143,995.50   $3,672.82  $4,757.45
#    4 │ Total   $251,862.50   $4,580.03  $6,118.67

# The historic VaR and ES more closely aligns with the T values than the normal values.

for s in stocks
    println(s, " : " , models[s].errorModel)
end
# Here are the fitted models for the T distributions
# SPY : LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: 0.00013517515811287465
# σ: 0.007114811600859023
# ρ: TDist{Float64}(ν=8.73299333948731)
# )

# AAPL : LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: -1.0243387037382775e-5
# σ: 0.010605997938009882
# ρ: TDist{Float64}(ν=5.231725019234337)
# )

# EQIX : LocationScale{Float64, Continuous, TDist{Float64}}(
# μ: -0.00013841493431296094
# σ: 0.011715826881866282
# ρ: TDist{Float64}(ν=4.9515764416332635)
# )

# We can see in each cash the ν parameter is less than 20, which indicates heavier tails than a normal distribution.
# It makes sense then that the T VaR and ES are more closely aligned with the historic values than the normal values.
# The retun series have excess kurtosis and the T model is able to capture this.

# Extra credit points (+1) if they calculate the VaR and ES using a non-weighted covariance matrix, making Note
# that the first method weights the more recent observations more heavily, which can also change the values.

#problem 3
# You are given a European call option with the following parameters:
#   •	Time to maturity: 3 months
#   •	Option price: 3
#   •	Risk-free rate(annually): 10% 
#   •	Stock price: 31
#   •	Strike price: 30
#   •	No Dividends are paid
#
# A.	Calculate the implied volatility. 
# B.	Calculate its Delta, Gamma, Vega and Theta. Using this information, how much should the option price change if the implied volatility increases by 1%?  Prove it.
# C.	Calculate the option value assuming it is a put. Prove put-call parity.
# D.	Given the portfolios:
#           One call option 
#           One put option 
#           One share of stock.
# Assuming the stock’s return is normally distributed and its annually volatility is 25%, the expected annual return is 0, there are 255 trading days per year,
# and the implied volatility does not change.
# calculate the 20- trading day 95% VaR and ES of the portfolio using
#  1. Delta-Normal approximation.
#  2. Monte Carlo Simulation.
# Note, don't forget to include the theta decay in the VaR and ES calculations.
# E. Discuss the differences in the two methods.

ttm = 0.25
cprice = 3.0
rf = 0.1
sprice = 31.0
strike = 30.0

# A.
f(iv) = gbsm(true,sprice,strike,ttm,rf,rf,iv).value - cprice
implied_vol = find_zero(f,1)
println("Implied Vol: $implied_vol")
# Implied Vol: 0.3350803924787909

# B.
cvals = gbsm(true,sprice,strike,ttm,rf,rf,implied_vol,includeGreeks=true)
println("Delta: $(cvals.delta)")
println("Gamma: $(cvals.gamma)")
println("Vega: $(cvals.vega)")
println("Theta: $(cvals.theta)")
# Delta: 0.665929652738692
# Gamma: 0.07006820782247573
# Vega: 5.640705439230118
# Theta: -5.5445615083589015

# The option price should increase by approximately 0.0564 if the implied volatility increases by 1%.
# Proof:
iv1 = implied_vol + 0.01
cprice2 = gbsm(true,sprice,strike,ttm,rf,rf,iv1,includeGreeks=false).value
println("Option Price with 1% increase in implied volatility: $cprice2")
println("Change in Option Price: $(cprice2 - cprice)")
# Option Price with 1% increase in implied volatility: 3.0564984275173437
# Change in Option Price: 0.056498427517343686

# C.
pvals = gbsm(false,sprice,strike,ttm,rf,rf,implied_vol,includeGreeks=true)
println("Put Option Value: $(pvals.value)")
# Put Option Value: 1.2592973608499776

# Put-Call Parity
# C + Xe^(-rfT) = P + S
println("Does Put-Call Parity Hold?: $(cvals.value + strike*exp(-rf*ttm) ≈ pvals.value + sprice)")
# Does Put-Call Parity Hold?: true

# D.
# Delta-Normal Approximation
portfolio = DataFrame(:stock=>["Call","Put","Stock"], :holding=>[1.,1.,1.])
portfolioValue = sum([cvals.value,pvals.value,sprice])
deltas = [cvals.delta, pvals.delta, 1.0]
dR = sprice*sum(deltas)/portfolioValue
s = dR * 0.25/sqrt(255) * sqrt(20)

# Theta Decay
timeDecay = pvals.value + cvals.value - (gbsm(false,sprice,strike,ttm-20/255,rf,rf,implied_vol,includeGreeks=true).value + gbsm(true,sprice,strike,ttm-20/255,rf,rf,implied_vol,includeGreeks=false).value)

# Distribution of returns expected is time decay as a % of total, with the std calculated above.
pDist = Normal(-timeDecay/portfolioValue,s)

VaR95 = VaR(pDist,alpha=0.05) * portfolioValue
ES95 = ES(pDist,alpha=0.05) * portfolioValue
println("Delta-Normal Approximation")
println("VaR95: $(format_currency(VaR95))")
println("ES95: $(format_currency(ES95))")
# Delta-Normal Approximation
# VaR95: $5.45
# ES95: $6.66

# Monte Carlo Simulation
function pricePortfolio(s,ttm)
    s + gbsm(true,s,strike,ttm,rf,rf,implied_vol,includeGreeks=false).value + gbsm(false,s,strike,ttm,rf,rf,implied_vol,includeGreeks=false).value
end

nsim = 100000
sim = sprice * (1 .+ rand(Normal(0,0.25*sqrt(20)/sqrt(255)),nsim))
portfolioSim = pricePortfolio.(sim,ttm-20/255) .- portfolioValue

VaR95 = VaR(portfolioSim,alpha=0.05)
ES95 = ES(portfolioSim,alpha=0.05)
println("Simulated")
println("VaR95: $(format_currency(VaR95))")
println("ES95: $(format_currency(ES95))")
# Simulated
# VaR95: $4.27
# ES95: $4.73

# E.
# The Delta-Normal approximation shows a much larger risk.  The portfolio must be positively convex.  As the price falls, the 
# the delta is getting smaller, i.e. the total portfolio gamma is positive.  DN assumes a 0 gamma, which will
# overestimate the risk if gamma > 0.

# Looking at the plot of the portfolio value vs the stock price, 20 days ahead, we can see this is the case.

plot(20:40, pricePortfolio.(20:40,ttm-20/255), label="Portfolio Value", xlabel="Stock Price", ylabel="Portfolio Value", title="Portfolio Value vs Stock Price", legend=:topleft)
hline!([portfolioValue], line=:dash, label="Initial Portfolio Value")
vline!([sprice], line=:dash, label="Initial Stock Price")