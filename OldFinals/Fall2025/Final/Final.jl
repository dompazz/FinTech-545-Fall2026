using BenchmarkTools
using Distributions
using Random
using StatsBase
using DataFrames
using Roots
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
using MarketData

include("../library/RiskStats.jl")
include("../library/gbsm.jl")
include("../library/expost_factor.jl")
include("../library/fitted_model.jl")
include("../library/missing_cov.jl")
include("../library/optimizers.jl")
include("../library/return_calculate.jl")
include("../library/return_accumulate.jl")
include("../library/simulate.jl")
include("../library/ewCov.jl")

# Outline
# given a simulated return series, price an option
#   compare to a european model based on vol
#   - graph the European IV curve based on the solved price.
# risk on 1 equity + put + call portfolio
#   optimize based on risk vs return SR vs ES, allow shorting
#   compare to a sharpe ratio portfolio
# Optimal SR portfolio vs RP.
#   Historical data to optimize 
#   OOS attribution


#1 -- Data Creation
rf = .04
Po = 100
ttm = 1/255

Random.seed!(100)
d = TDist(8)*.02
r = rand(d,5000)
m = log(mean(exp.(r)))
r .+= rf/255 - m 

CSV.write("problem1.csv",DataFrame(:r=>r))

r = CSV.read("problem1.csv",DataFrame).r
Psim = Po .* exp.(r)
function opt_call(sim,strike,rf,ttm)
    return exp(-rf*ttm)*mean(max.(sim .- strike,0.0))
end
function opt_put(sim,strike,rf,ttm)
    return exp(-rf*ttm)*mean(max.(strike .- sim,0.0))
end
strikes = [i for i in 95:.01:105]
calls = [opt_call(Psim,s,rf,ttm) for s in strikes]
puts = [opt_put(Psim,s,rf,ttm) for s in strikes]

# Part 1  What is the price of a call and a put at 
# P[99,100,101]
idxs = findall(r-> r in [99,100,101], strikes)
DataFrame(:Strike=>strikes[idxs],:Call=>calls[idxs],:Put=>puts[idxs])
#  Row │ Strike   Call      Put      
#      │ Float64  Float64   Float64  
# ─────┼─────────────────────────────
#    1 │    99.0  1.48705   0.471518
#    2 │   100.0  0.886014  0.870329
#    3 │   101.0  0.4832    1.46736

# Graph the implied volatility of call option prices using BSM
# for strikes in [95,105]
n = length(strikes)
call_iv = fill(0.0,n)
put_iv = fill(0.0,n)
guess = std(r)*sqrt(255)
for i in 1:n
    s = strikes[i]
    f(iv) = gbsm(true,Po,s,1/255,rf,rf,iv).value - calls[i]
    call_iv[i] = find_zero(f,guess)
end

p = plot(strikes,call_iv,label="Call Implied Vol")
savefig(p, "call_iv.png")

# What do you notice?  Theory would say this should
# be a flat line.  Why is it not?

skewness(r)
#-0.08653526270817972
kurtosis(r)
#1.7326510885593516
# The graph exhibits a "smile" (raised IV away from the current price)
# and a left side skew with larger IV in the downside cases
# The simulated returns are negatively skewed and exhibit excess kurtosis.
# this leads to a larger number of simulated values in the tails and especially 
# the left tail.  The explains higher than expected via the normal distribution 
# option values in the tails and especially the left tail, leading to higher IV 

# question 2 data Generation
spy = DataFrames.rename(DataFrame(yahoo(:SPY, YahooOpt(period1=DateTime(2000,1,1)))), Dict(:timestamp=>:Date, :AdjClose=>:SPY))[!,[:Date, :SPY]]
filter!(r->r.Date >= Date(2023,1,1),spy)

CSV.write("problem2.csv",spy)

# Div Rate = 1.09%
# Risk Free Rate 4%
# Call Option 
#   Strike=665
#   Expiration=12/5/2025
#   Price=7.05
# Put Option 
#   Strike=655
#   Expiration=12/5/2025
#   Price=7.69
# 255 Trading Days per year
# Assume 10 trading days until expiration

# Find the best fit model between a Normal and a T Distribution for 
# the return of SPY.  Do not remove the mean.  Assume arthimetic returns.
spy = CSV.read("problem2.csv",DataFrame)
r = return_calculate(spy,dateColumn="Date")
nfit = fit_normal(r.SPY)
tfit = fit_general_t(r.SPY)
aicc(nfit,r.SPY) #-4650.12207692453
aicc(tfit,r.SPY) #-4835.603910569093
# T is the better fit 

# Calculate the implied volatility of the Put and the Call 
ttm = 11 / 255
rf = 0.04
q = rf - 0.0109
cP = 7.05
pP = 7.69
cX = 665
pX = 655
S = spy.SPY[end]
f(iv) = gbsm(true,S,cX,ttm,rf,q,iv).value - cP
cIV = find_zero(f,.2)
# 0.1714605924499435

f(iv) = gbsm(false,S,pX,ttm,rf,q,iv).value - pP
pIV = find_zero(f,.2)
# 0.18255819618368563

# You own 1 put and are short 1 call and are long 1 share of the stock.
# What is the VaR and ES (in absolute % term, 5%) of the portfolio holding 
# the portfolio until 11/28/2025.  Assume no market holidays,
# only weekends (5 trading days).

pValInit = spy.SPY[end] + pP - cP

Random.seed!(200)
nsim = 100000
rsim = rand(tfit.errorModel,nsim*5)
Psim = fill(0.0,nsim)
for i in 1:nsim
    p = S
    for j in 1:5
        p *= (1 + rsim[(i-1)*5+j])
    end
    Psim[i] = p
end

pVal = copy(Psim)
pVal .+= [gbsm(false,p,pX,5/255,rf,q,pIV).value for p in Psim]
pVal .-= [gbsm(true,p,cX,5/255,rf,q,cIV).value for p in Psim]

pnl = (pVal .- pValInit)/pValInit
VaR(pnl)
# 0.005468815985288754
ES(pnl)
# 0.006469255126340947

# Assume you can hold fractional shares.  
# You may be short or long up to 2 units of each h ∈ [-2,2]
# find the portfolio of these 3 assets that maximizes the ratio
# of (mean(simulated return) - rf(for the holding period))/ES(as a %)
# the total value of your portfolio should equal the current portfolio 
# value
# HINTS
# Current Portfolio Value = 659.67
# rf_holding = rf*5/255
# the ratio of the current portfolio is 
# (mean(pnl) - rf_holding)/ES(pnl) ≈ 0.0698
# If you have convergence problems, try relaxing your tolerance to converge at 1e-5

rf_holding = rf*5/255

# just to see current holding ratio is (not needed for grading)
(mean(pnl) - rf_holding)/ES(pnl)

function toMax(h1,h2,h3)
    init = h1*S + h2*pP + h3* cP
    pVal = h1 * copy(Psim)
    pVal .+= h2 * [gbsm(false,p,pX,5/255,rf,q,pIV).value for p in Psim]
    pVal .+= h3 * [gbsm(true,p,cX,5/255,rf,q,cIV).value for p in Psim]
    pnl = (pVal .- init)/init
    return (mean(pnl) - rf_holding)/ES(pnl;alpha=0.05)
    # return init
end

round(toMax(1,1,-1);digits=4) ≈ 0.0698

model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "tol", 1e-5)
set_optimizer_attribute(model, "max_iter", 100) # Increase max iterations
set_optimizer_attribute(model, "hessian_approximation", "limited-memory") # Use L-BFGS
set_optimizer_attribute(model, "acceptable_tol", 1e-4) # Relax acceptable tolerance

ho = [1,1,-1]
@variable(model, 2>= h1 >=-2, start=ho[1])
@variable(model, 2>= h2 >=-2, start=ho[2])
@variable(model, 2>= h3 >=-2, start=ho[3])

register(model,:mr,3,toMax,autodiff=true)
@NLobjective(model,Max,mr(h1,h2,h3))
@constraint(model, (h1*S + h2*pP + h3* cP)==pValInit)
optimize!(model)
h = value.([h1,h2,h3])
DataFrame(:Asset=>["Stock", "Put", "Call"], :Holding=>round.(h;digits=4))
# 3×2 DataFrame
#  Row │ Asset   Holding 
#      │ String  Float64
# ─────┼─────────────────
#    1 │ Stock    1.0401
#    2 │ Put     -1.5224
#    3 │ Call    -2.0
toMax(h...)

# Plot the portfolio value of your optimal holdings across the range of values from your 
# simulation.  How would you judge the risk of this portfolio on a qualitative basis?

s, e = extrema(Psim)
prices = [i for i in s:5:e]
pVal = h[1] * copy(prices)
pVal .+= h[2] * [gbsm(false,p,pX,5/255,rf,q,pIV).value for p in prices]
pVal .+= h[3] * [gbsm(true,p,cX,5/255,rf,q,cIV).value for p in prices]
pnl = pVal .- pValInit
p = plot(prices,pVal,label="Portfolio Value")
hline!(p, [pValInit], label="Initial Portfolio Value", linestyle=:dash)
savefig(p, "p2_values.png")

# Not required: but let's look at the histogram of pnl
prices = copy(Psim)
pVal = h[1] * copy(prices)
pVal .+= h[2] * [gbsm(false,p,pX,5/255,rf,q,pIV).value for p in prices]
pVal .+= h[3] * [gbsm(true,p,cX,5/255,rf,q,cIV).value for p in prices]
pnl = pVal .- pValInit
p = histogram(pnl)
savefig(p,"prob2_hist_pnl.png")

# Calculate the probability of loss
prob_loss = mean(pnl .< 0)
println("Probability of loss: ", round(prob_loss, digits=4))

# this portfolio is extremely risky.  The distribution has extremely fat tails (t fit with a very small)
# df value.  There is a ~17% chance of loss with recent history as our guide.  
# When we think about the experience in the GFC, large moves down or up are possible and could
# lead to outsized losses.

# Problem 3 Data Generation
stocks = ["GOOG","JPM", "WMT", "AMD", "NKE"]

# DataFrames.rename(DataFrame(yahoo(:SPY, YahooOpt(period1=DateTime(2000,1,1)))), Dict(:timestamp=>:Date, :AdjClose=>:SPY))[!,[:Date, :SPY]]
function dnld(stocks, startDt)
    data = DataFrames.rename(DataFrame(yahoo(Symbol(stocks[1]), YahooOpt(period1=startDt))), Dict(:timestamp=>:Date, :AdjClose=>Symbol(stocks[1])))[!,[:Date, Symbol(stocks[1])]]
    if length(stocks) > 1
        return (outerjoin(
            data,
            dnld(stocks[2:end],startDt),
            on=:Date
        ))
    end
    return data
end

prices = dnld(stocks,DateTime(2020,1,1))

insample = filter(r-> r.Date < Date(2024,9,1),prices)
insample = return_calculate(insample,dateColumn="Date")
insample = return_accumulate(insample, dateColumn="Date")
CSV.write("problem3_insample.csv",insample)

outsample = filter(r-> r.Date >= Date(2024,9,1) && r.Date < Date(2025,9,1), prices)
outsample = return_calculate(outsample,dateColumn="Date")
outsample = return_accumulate(outsample, dateColumn="Date")
CSV.write("problem3_outsample.csv",outsample)

# Calculate the Max SR portfolio and the Risk Parity Portfolio
# using the insample data.  Risk Free Rate = 4%.  Use the insample
# mean as the Expected Return.  Calculate your covariance using
# exponentially weighted covariance.  Scale both to be annual values.
# HINTS
#   The data is monthly but you are holding for 12 months.  You need to scale your inputs.
#   Scale expected return as (1 + er)^12 - 1
#   Scale the covariance as ewCovar * 12

rf = 0.04
insample = CSV.read("problem3_insample.csv",DataFrame)
outsample = CSV.read("problem3_outsample.csv",DataFrame)

l = 0.97
stocks = filter(r -> r != "Date", names(insample))

insample_r = Matrix(insample[!,stocks])
outsample_r = Matrix(outsample[!,stocks])

er = mean(insample_r,dims=1)
# scale to yearly
er = (1 .+ er).^12 .- 1

covar = ewCovar(insample_r,l)*12

mSR,status = maxSR(covar,vec(er),rf)
rPP,status = riskParity(covar)
DataFrame(:Stock=>stocks,:MaxSR=>round.(mSR;digits=4),:RPP=>round.(rPP;digits=4))
# 5×3 DataFrame
#  Row │ Stock   MaxSR    RPP     
#      │ String  Float64  Float64
# ─────┼──────────────────────────
#    1 │ GOOG     0.2699   0.2134
#    2 │ JPM      0.0745   0.2071
#    3 │ WMT      0.5302   0.3149
#    4 │ AMD      0.1254   0.1007
#    5 │ NKE     -0.0      0.1639

#Assume you hold these portfolios during the out of sample period.
# you do not rebalance during the period.  Calculate the per stock
# attribution for both ex-post risk and return.  Compare these 
# values to your ex-ante values.
# HINTS
#   Compare both the absolute contribution as well as the percent of total
#   Portfolio volatility ex-post will be monthly so do not forget to annualize it
#      This is invariant as a % which is why % helps.

mSR_attr = expost_factor(mSR,outsample[:,stocks],outsample[:,stocks],I(length(stocks))).Attribution
rPP_attr = expost_factor(rPP,outsample[:,stocks],outsample[:,stocks],I(length(stocks))).Attribution

mSR_attr[:,["Value", stocks..., "Portfolio"]]
# 3×7 DataFrame
#  Row │ Value               GOOG       JPM         WMT        AMD        NKE           Portfolio 
#      │ String              Float64    Float64     Float64    Float64    Float64       Float64
# ─────┼──────────────────────────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.352721   0.39895     0.26896    0.1876     -0.0324901    0.291046
#    2 │ Return Attribution  0.0917842  0.0286741   0.147835   0.0227532   5.87927e-10  0.291046
#    3 │ Vol Attribution     0.0147203  0.00298432  0.0287636  0.0051491  -4.30373e-10  0.0516173

function analyze_attribution(mSR_attr, er, covar, mSR, stocks)
    ret_attr = Vector(mSR_attr[2,stocks])
    vol_attr = sqrt(12) .* Vector(mSR_attr[3,stocks])
    return_contrib_pct = ret_attr ./ mSR_attr[2,"Portfolio"]
    vol_contrib_pct = vol_attr ./ (sqrt(12) * mSR_attr[3,"Portfolio"])
    ea_ret_attr = (er' .* mSR)[:,1]
    ea_ret_attr_pct = ea_ret_attr ./ (er*mSR)
    ea_vol_attr = mSR .* covar * mSR / (sqrt(mSR'*covar*mSR))
    ea_vol_attr_pct = ea_vol_attr ./ sum(ea_vol_attr)
    
    return (
        DataFrame(
            :Stock => stocks,
            :Expected_Return_Attribution => (ea_ret_attr),
            :Actual_Return_Attribution => (ret_attr),
            :Expected_Return_Attrib_Pct => (ea_ret_attr_pct),
            :Actual_Return_Attrib_Pct => (return_contrib_pct)
        ),
        DataFrame(
            :Stocks => stocks,
            :Expected_Vol_Attribution => ea_vol_attr,
            :Actual_Vol_Attribution => vol_attr,
            :Expected_Vol_Attrib_Pct => ea_vol_attr_pct,
            :Actual_Vol_Attrib_Pct => vol_contrib_pct
        )
    )
end
ret_attr, vol_attr = analyze_attribution(mSR_attr, er, covar, mSR, stocks)
println(ret_attr)

# 5×5 DataFrame
#  Row │ Stock   Expected_Return_Attribution  Actual_Return_Attribution  Expected_Return_Attrib_Pct  Actual_Return_Attrib_Pct 
#      │ String  Float64                      Float64                    Float64                     Float64
# ─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ GOOG                    0.0685197                  0.0917842                   0.284836                   0.315359
#    2 │ JPM                     0.0136008                  0.0286741                   0.0565384                  0.0985209
#    3 │ WMT                     0.102443                   0.147835                    0.425856                   0.507942
#    4 │ AMD                     0.0559948                  0.0227532                   0.23277                    0.0781772
#    5 │ NKE                    -1.65876e-10                5.87927e-10                -6.89545e-10                2.02005e-9

# On an absolute basis, the portfolio outperformed expectation.  On a relative basis, AMD under performed expectations.

println(vol_attr)
# 5×5 DataFrame
#  Row │ Stocks  Expected_Vol_Attribution  Actual_Vol_Attribution  Expected_Vol_Attrib_Pct  Actual_Vol_Attrib_Pct 
#      │ String  Float64                   Float64                 Float64                  Float64
# ─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ GOOG                  0.0506996               0.0509925                0.287824               0.285181
#    2 │ JPM                   0.00932742              0.010338                 0.0529521              0.0578163
#    3 │ WMT                   0.0713475               0.09964                  0.405043               0.557248
#    4 │ AMD                   0.0447736               0.017837                 0.254181               0.0997554
#    5 │ NKE                  -1.46996e-9             -1.49086e-9              -8.34502e-9            -8.33778e-9

# Absolute Volatility attribution numbers are in line except for AMD which had a much lower volatility.  On a 
# relative basis we can see this was made up for by WMT adding more than expected volatility.



rPP_attr[:,["Value", stocks..., "Portfolio"]]
# 3×7 DataFrame
#  Row │ Value               GOOG       JPM        WMT        AMD         NKE          Portfolio 
#      │ String              Float64    Float64    Float64    Float64     Float64      Float64
# ─────┼─────────────────────────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.352721   0.39895    0.26896    0.1876      -0.0324901    0.25615
#    2 │ Return Attribution  0.0743831  0.0824089  0.0901335  0.0188854   -0.00966092   0.25615
#    3 │ Vol Attribution     0.011312   0.0102719  0.0149205  0.00591603   0.0103726    0.052793


ret_attr, vol_attr = analyze_attribution(rPP_attr, fill(0.0,(1,length(stocks))), covar, rPP, stocks)
println(vol_attr)
# 5×5 DataFrame
#  Row │ Stocks  Expected_Vol_Attribution  Actual_Vol_Attribution  Expected_Vol_Attrib_Pct  Actual_Vol_Attrib_Pct 
#      │ String  Float64                   Float64                 Float64                  Float64
# ─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ GOOG                    0.036102               0.0391858                      0.2               0.21427
#    2 │ JPM                     0.036102               0.0355828                      0.2               0.194569
#    3 │ WMT                     0.036102               0.0516861                      0.2               0.282622
#    4 │ AMD                     0.036102               0.0204937                      0.2               0.112061
#    5 │ NKE                     0.036102               0.0359318                      0.2               0.196477
#RPP - We don't have ex-ante return assumptions, only risk contribution.  Here we have 
# roughly equal ex-post risk contributions execpt for AMD which is about half the expected amount and WMT with adding
# more volatility than expected.  

# Overall the Max Sharpe Ratio portfolio outperformed by about 3.5% with roughly the same volatility.  The 
# choice to set the NKE weight to 0 allowed the Max Sharpe Ratio portfolio to add more to the other stocks.
# NKE was down 3.2% on the out of sample period and the RPP put a 16% weight to it.  While it overweighted
# JPM, the biggest winner in the out of sample period, it was unable to make up for the loss of return from 
# the allocation to NKE.  