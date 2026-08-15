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
using PyCall
using KernelDensity

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
include("../../library/SkewNormal.jl")

theme(:dark)

# prices = CSV.read("Final Project/DailyPrices.csv", DataFrame)
# returns = return_calculate(prices,dateColumn="Date")
# cprice = prices[end,:]
# nms = filter(n-> !(n in ["Date", "SPY"]), names(returns))

# Random.seed!(1234)
# onms = sort(DataFrame(:Name=>nms, :r=> rand(length(nms))), :r).Name

# portfolio = DataFrame(:Portfolio=>vcat(fill("A",33),fill("B",33),fill("C",33)), :Symbol=>onms )
# portfolio[!,:Holding] = [cld(1000000/99, cprice[s]) for s in portfolio.Symbol]

# CSV.write("Final Project/initial_portfolio.csv", portfolio)

# using MarketData
# df = DataFrames.rename(DataFrame(yahoo(:BOXX, YahooOpt(period1=DateTime(2022,9,1)))), Dict(:timestamp=>:Date, :AdjClose=>:rf))[!,[:Date, :rf]]
# rfs = return_calculate(df,dateColumn="Date")
# rfs = innerjoin(rfs, returns, on=:Date)[!,[:Date, :rf]]
# CSV.write("Final Project/rf.csv", rfs)


#Part 1
# You own the 3 portfolios in the file initial_portfolio.csv. 
# You model the returns of the stocks using CAPM with SPY as the market. 
# Bought these portfolios at the end of 2023.
# Fit the CAPM model to the returns of the stocks in the portfolios from the start of the data to the end of 2023.
# Use the fitted models to attribute the risk and return for each portfolio for the time period of 2024 
# until the end of the series (the holding period).  Split the attribution between the systematic and idiosyncratic components.
# Calculate the idiosyncratic component of each stock, but you only need to report the total.
# Then attribute the total risk and return (all 3 portfolios) into each Subportfolio
# Use the risk free rate in rf.csv as the risk free rate.  Stock prices are in DailyPrices.csv

#Read in the Data
portfolio = CSV.read("Final Project/initial_portfolio.csv", DataFrame)
rf = CSV.read("Final Project/rf.csv", DataFrame)
prices = CSV.read("Final Project/DailyPrices.csv", DataFrame)
allReturns = return_calculate(prices,method="DISCRETE", dateColumn="Date")
allReturns = leftjoin(allReturns, rf, on=:Date)
stocks = portfolio.Symbol
nStocks = length(stocks)

#Fit the CAPM Model
function OLS(X,Y)
    n = size(X,1)
    X = hcat(ones(n),X)
    b = (X'X)\(X'Y)
    return b
end

Betas = zeros(nStocks)
Alphas = zeros(nStocks)
i=1
toFit = @view allReturns[allReturns.Date .< Date(2023, 12, 31),:]

errors = zeros(nrow(toFit),nStocks)

for s in stocks
    Alphas[i], Betas[i] = OLS(toFit.SPY - toFit.rf, toFit[:,s]-toFit.rf)
    errors[:,i] = toFit[:,s] - (Betas[i]*(toFit.SPY - toFit.rf) + toFit.rf .+ Alphas[i])
    
    i += 1
end
CAPMBetas = DataFrame(:Symbol=>stocks, :Beta=>Betas)

realizedReturns = @view allReturns[allReturns.Date .>= Date(2024, 1, 1),stocks]
realizedSPY = DataFrame(
    :SPY=> allReturns[allReturns.Date .>= Date(2024, 1, 1),:SPY]
        - allReturns[allReturns.Date .>= Date(2024, 1, 1),:rf]
    )

# realizedSPY = DataFrame(
#     :SPY=> allReturns[allReturns.Date .>= Date(2024, 1, 1),:SPY]
#     )

realizedSPY = DataFrame(
    :SPY=> allReturns[allReturns.Date .>= Date(2024, 1, 1),:SPY]
        - allReturns[allReturns.Date .>= Date(2024, 1, 1),:rf],
    :rf=> allReturns[allReturns.Date .>= Date(2024, 1, 1),:rf] .+0.0
    )

lastDate = allReturns.Date[allReturns.Date .< Date(2024, 1, 1)][end] 
startPrices = @view prices[prices.Date .== lastDate, stocks]

attribs = Dict{String, ExPostAttribution}()

function RunAttribution(realizedReturns,realizedSPY,lastDate,startPrices, portfolio, attribs)
    tValue = Matrix(startPrices) * portfolio.Holding
    w = (Matrix(startPrices)' .* portfolio.Holding) ./ tValue 
    _w = copy(w)
    _Betas = hcat(Betas,ones(nStocks))
    attrib = expost_factor(w,realizedReturns, realizedSPY, _Betas)
    attribs["Total"] = attrib
    println("Total Portfolio Attribution")
    println(attrib)

    portfolios = sort(collect(Set(portfolio.Portfolio)))
    for p in portfolios
        stocks = portfolio[portfolio.Portfolio .== p, :Symbol]
        stockReturns = realizedReturns[!, stocks]
        _SP = startPrices[!, stocks]
        tValue = Matrix(_SP) * portfolio[ [_p in stocks for _p in portfolio.Symbol],:Holding]
        w = (Matrix(_SP)' .* portfolio[ [_p in stocks for _p in portfolio.Symbol],:Holding]) ./ tValue 
        _B = Betas[[_p in stocks for _p in portfolio.Symbol]]
        _B = hcat(_B,ones(length(_B)))

        attrib = expost_factor(w,stockReturns, realizedSPY, _B)
        println("$p Portfolio Attribution")
        println(attrib)
        attribs["$p"] = attrib
    end
    # return _w
end

RunAttribution(realizedReturns,realizedSPY,lastDate,startPrices, portfolio, attribs)


# Total Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY         rf          Alpha         Portfolio  
#      │ String              Float64     Float64     Float64       Float64    
# ─────┼──────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781    0.0522772   -0.0378171    0.204731
#    2 │ Return Attribution  0.190166    0.0559802   -0.0414152    0.204731
#    3 │ Vol Attribution     0.00719736  2.49085e-5  -0.000132645  0.00708962
# A Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY         rf          Alpha         Portfolio 
#      │ String              Float64     Float64     Float64       Float64   
# ─────┼─────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781    0.0522772   -0.0961828    0.136642
#    2 │ Return Attribution  0.189015    0.0543455   -0.106719     0.136642
#    3 │ Vol Attribution     0.00705167  1.84709e-5   0.000348354  0.0074185
# B Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY         rf         Alpha         Portfolio  
#      │ String              Float64     Float64    Float64       Float64    
# ─────┼─────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781    0.0522772  -0.0327798    0.203526
#    2 │ Return Attribution  0.183015    0.0559487  -0.0354381    0.203526
#    3 │ Vol Attribution     0.00639798  2.8601e-5   0.000440635  0.00686722
# C Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY         rf          Alpha        Portfolio  
#      │ String              Float64     Float64     Float64      Float64
# ─────┼─────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781    0.0522772   0.021516     0.281172
#    2 │ Return Attribution  0.198754    0.0577829   0.0246353    0.281172
#    3 │ Vol Attribution     0.00721991  2.50556e-5  0.000678773  0.00792374

for p in ["Total", "A","B","C"]
    tv = attribs[p].Attribution[3,:Portfolio]
    nAhead = length(attribs["Total"].PortfolioReturn)
    tv *= sqrt(nAhead)
    tSR = (attribs[p].Attribution[1,:Portfolio] - attribs[p].Attribution[1,:rf]) / tv
    println("Portfolio $p Sharpe Ratio: ", tSR)
end
# Portfolio Total Sharpe Ratio: 1.3492680353726818
# Portfolio A Sharpe Ratio: 0.7135544191581354
# Portfolio B Sharpe Ratio: 1.381955240785756
# Portfolio C Sharpe Ratio: 1.812549744029679

# Part 2
# Use your fitted CAPM models from Part 1, assume 0 alpha, and the mean return on SPY is the average prior to the holding period.
# assume the average risk free rate prior to the holding period is the risk free rate for the optimization
# Create the optimal maximum Sharpe Ratio portfolio for each sub portfolio for the holding period.
# Rerun your attribution from Part 1 using the new optimal portfolios.
# Discuss the results comparing back to Part 1.  
# Given the fitted CAPM you have an expectation of the idiosyncratic risk contribution of each stock.  How does the model compare
# to the actuals values?

#Run the CAPM Regression
eRf = mean(toFit.rf)
eSPY = mean(toFit.SPY - toFit.rf)
eStocks = eSPY .* Betas .+ eRf
covar = cov(Matrix(toFit[!, stocks]))
mSRw, status = maxSR(covar,eStocks,eRf;printLevel=5)
mSRw = DataFrame(:stock=>stocks, :w=>mSRw ./ sum(mSRw))

#construct the portfolio, assume 1m total value.
mSRPortfolio = portfolio[:,:]
mSRPortfolio[!,:Holding] = mSRw.w .* 1000000 ./ vec(Matrix(startPrices[!, stocks]))
mSRPortfolio[!,:Weight] = mSRw.w


mSRattribs = Dict{String, ExPostAttribution}()
RunAttribution(realizedReturns,realizedSPY,lastDate,startPrices, mSRPortfolio, mSRattribs)
# Total Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY         rf          Alpha         Portfolio  
#      │ String              Float64     Float64     Float64       Float64
# ─────┼──────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781    0.0522772    0.000172655  0.263048
#    2 │ Return Attribution  0.20461     0.0573655    0.00107259   0.263048
#    3 │ Vol Attribution     0.00790471  1.97272e-5  -0.000536871  0.00738756
# A Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY        rf          Alpha         Portfolio  
#      │ String              Float64    Float64     Float64       Float64
# ─────┼─────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781   0.0522772   -0.00791504   0.270365
#    2 │ Return Attribution  0.220428   0.0575444   -0.00760648   0.270365
#    3 │ Vol Attribution     0.0080084  1.19537e-5   0.000160543  0.00818089
# B Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY        rf          Alpha        Portfolio  
#      │ String              Float64    Float64     Float64      Float64
# ─────┼────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781   0.0522772   0.0120802    0.257097
#    2 │ Return Attribution  0.18526    0.0572306   0.0146064    0.257097
#    3 │ Vol Attribution     0.0062032  1.57498e-5  0.000800887  0.00701983
# C Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY         rf          Alpha         Portfolio  
#      │ String              Float64     Float64     Float64       Float64
# ─────┼──────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781    0.0522772   -0.00353178   0.262447
#    2 │ Return Attribution  0.208882    0.0573409   -0.00377686   0.262447
#    3 │ Vol Attribution     0.00782968  2.53989e-5   0.000758185  0.00861327


for p in ["Total", "A","B","C"]
    tv = mSRattribs[p].Attribution[3,:Portfolio]
    nAhead = length(mSRattribs["Total"].PortfolioReturn)
    tv *= sqrt(nAhead)
    tSR = (mSRattribs[p].Attribution[1,:Portfolio] - mSRattribs[p].Attribution[1,:rf]) / tv
    println("Portfolio $p Sharpe Ratio: ", tSR)
end
# Portfolio Total Sharpe Ratio: 1.7901594507538876
# Portfolio A Sharpe Ratio: 1.6726870851864648
# Portfolio B Sharpe Ratio: 1.8307479384595384
# Portfolio C Sharpe Ratio: 1.531032742083497

# Construct the risk model.  Need total Beta, weights, error terms for each stock, and variance of SPY
beta = mSRw.w' * Betas
eCovar = cov(errors)
covar = zeros(nStocks+1,nStocks+1)
covar[1,1] = var(toFit.SPY)
covar[2:end,2:end] = eCovar

function pvol(w...)
    x = collect(w)
    # println(x)
    return(sqrt(x'*covar*x))
end

function pCSD(w...)
    x = collect(w)
    pVol = pvol(w...)
    csd = x.*(covar*x)./pVol
    return (csd)
end

# Expected Component SD
mSR_CSD = pCSD(vcat(beta,mSRw.w)...)
println("EXPECTED: SPY CSD: ", mSR_CSD[1], " --  Alpha CSD: ", sum(mSR_CSD[2:end]), " --  Total: ", sum(mSR_CSD))
println("REALIZED: SPY CSD: ", mSRattribs["Total"].Attribution[3,:SPY], " --  Alpha CSD: ", mSRattribs["Total"].Attribution[3,:Alpha], " --  Total: ", mSRattribs["Total"].Attribution[3,:Portfolio])

# EXPECTED: SPY CSD: 0.008181886550851537 --  Alpha CSD: 5.171531854115315e-5 --  Total: 0.00823360186939269
# REALIZED: SPY CSD: 0.007904708583021149 --  Alpha CSD: -0.0005368713117273313 --  Total: 0.007387564435015646
# On a total level, the error term expects to add 5.2e-7 std to the daily portfolio risk, but the realized value is -5e-4 std.  This is a large difference.
# Total portfolio STD is 8.4e-4 lower than expected.  or about 1.3% annualized.  

indR = mSRattribs["Total"].ResidIndivual .* mSRattribs["Total"].carinoK
_Y =  hcat(Matrix(realizedSPY) .* mSRattribs["Total"].FactorWeights, mSRattribs["Total"].ResidIndivual )

indCSD = OLS(mSRattribs["Total"].PortfolioReturn,_Y)[2,:] 
indCSD ./= abs(sum(indCSD[3:end]))

mSR_CSD ./= sum(mSR_CSD[2:end])
individualResidCSD = DataFrame(
    :Stock=>stocks,
    :Expected=>mSR_CSD[2:end],
    :Realized=>indCSD[3:end],
    :Diff=>indCSD[3:end] - mSR_CSD[2:end] ,
)

# Individual Residual CSD as a percent of the total Alpha CSD.  Sign preserved.
# 99×4 DataFrame
#  Row │ Stock    Expected      Realized     Diff
#      │ String7  Float64       Float64      Float64
# ─────┼──────────────────────────────────────────────────
#    1 │ WFC       0.000698153  -0.00222725  -0.00292541
#    2 │ ETN       0.00124771    0.00621796   0.00497024
#    3 │ AMZN      0.0190821    -0.0120869   -0.031169
#    4 │ QCOM      0.00230237    0.0238045    0.0215021
#    5 │ LMT       0.00331314   -0.0161731   -0.0194863
#    6 │ KO        7.85628e-8   -3.71988e-7  -4.50551e-7
#    7 │ JNJ       6.09391e-9   -1.90759e-8  -2.51698e-8
#    8 │ ISRG      0.0144292    -0.0360441   -0.0504732
#   ⋮  │    ⋮          ⋮             ⋮            ⋮
#   92 │ NEE       0.00253427   -0.0291635   -0.0316977
#   93 │ ABBV     -0.000554336   0.00744857   0.0080029
#   94 │ TSLA      0.0270609     0.0695413    0.0424804
#   95 │ MSFT      0.0413983     0.0228526   -0.0185457
#   96 │ PEP       0.0101178    -0.045172    -0.0552898
#   97 │ CB        0.00442246   -0.0077719   -0.0121944
#   98 │ PANW      0.00863654    0.00948616   0.000849617
#   99 │ BLK       0.00126592   -0.00946406  -0.01073

leftjoin!(individualResidCSD, mSRPortfolio, on=(:Stock=>:Symbol))
individualResidCSD[!,:Expected] = individualResidCSD[!,:Expected] .* individualResidCSD[!,:Weight]
individualResidCSD[!,:Realized] = individualResidCSD[!,:Realized] .* individualResidCSD[!,:Weight]
individualResidCSD[!,:Diff] = individualResidCSD[!,:Diff] .* individualResidCSD[!,:Weight]
gdf = groupby(individualResidCSD, :Portfolio)
combine(gdf, 
    :Diff=>sum,
    :Expected=>sum,
    :Realized=>sum,)
# 3×4 DataFrame
# Row │ Portfolio  Diff_sum     Expected_sum  Realized_sum 
#     │ String1?   Float64      Float64       Float64
# ─────┼────────────────────────────────────────────────────
#     1 │ A          -0.0213271     0.0070203    -0.0143068
#     2 │ B          -0.0253398     0.0102366    -0.0151031
#     3 │ C          -0.00846921    0.00701789   -0.00145132

# Part 3
# Investigate the Normal Inverse Gaussian and the Skew Normal distributions.  Discuss how these distributions can be used in finance
# and in particular in the context of this class.

# Part 4
# Implement a the Normal Inverse Gaussian and Skew Normal distributions (you my use these if they are implemented in a package for you).
# create a risk model where you fit each stock to the Normal, Generalized T, Normal Inverse Gaussian, and Skew Normal distributions, taking the best fit.
# Use the data from the start though 2023.
# Make the assumed return on each stock be 0%.
# Report which model was fit for each stock and the parameters.
# Calculate the 1 day 5% VaR and ES for each portfolio and the total portfolio using a Gassian Copula and the fitted distributions.
# Do the same but with a Multivariate normal distribution for all stocks.  Discuss the difference between the two approaches.

models = Dict{String,FittedModel}()
for s in stocks
    x = copy(toFit[!,s])
    x = x .- mean(x)

    aiccs = Dict{FittedModel, Float64}()
    nm = fit_normal(x)
    aiccs[nm] = aicc(nm,x)

    try
        gt = fit_general_t(x)
        aiccs[gt] = aicc(gt,x)
    catch
    end
    try
        nig = fit_NIG_mm(x)
        aiccs[nig] = aicc(nig,x)
    catch
    end
    try
        sn = fit_skewnormal(x)
        aiccs[sn] = aicc(sn,x)
    catch
    end

    models[s] = findmin(aiccs)[2]

end

modelFits = DataFrame(
    :Stock=>stocks,
    :Model=>[typeof(models[s].errorModel) for s in stocks],
    :Params=>[params(models[s].errorModel) for s in stocks],
    :AICCs=>[aicc(models[s],toFit[!,s]) for s in stocks]
)

fits = Set(modelFits.Model)
cts = Dict()
for f in fits
    cts[f] = 0
end
for m in modelFits.Model
    cts[m] += 1
end
println("Model Counts: ", cts)
# Dict{Any, Any} with 3 entries:
#   LocationScale{Float64, Continuous, TDist{Float64}} => 86
#   SkewNormal{Float64}                                => 1
#   NormalInverseGaussian{Float64}                     => 12

nSim = 100000
iteration = [i for i in 1:nSim]
U = DataFrame()
for s in stocks
    U[!,s] = models[s].u
end

spCovar = corspearman(Matrix(U[!,stocks]))
simRets = DataFrame(cdf.(Normal(),simulateNormal(nSim, spCovar; seed=1234)),stocks)

@time Threads.@threads for s in stocks
    println("Simulating $s")
    simRets[!,s] = models[s].eval.(simRets[!,s])
end


#Protfolio Valuation
st = time()
values = crossjoin(mSRPortfolio, DataFrame(:iteration=>iteration))

nVals = size(values,1)
currentValue = Vector{Float64}(undef,nVals)
simulatedValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
Threads.@threads for i in 1:nVals
    price = startPrices[1,values.Symbol[i]]
    currentValue[i] = values.Holding[i] * price
    simulatedValue[i] = values.Holding[i] * price*(1.0+simRets[values.iteration[i],values.Symbol[i]])
    pnl[i] = simulatedValue[i] - currentValue[i]
end
values[!,:currentValue] = currentValue
values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl

println("Valuation Took $(time()-st)")
values[!,:Portfolio] = String.(values.Portfolio)
riskValues = aggRisk(values,[:Portfolio])[!,[:currentValue, :Portfolio, :VaR95, :ES95, :VaR95_Pct, :ES95_Pct]]
# 4×6 DataFrame
#  Row │ currentValue  Portfolio  VaR95     ES95      VaR95_Pct  ES95_Pct  
#      │ Float64       String     Float64   Float64   Float64    Float64
# ─────┼───────────────────────────────────────────────────────────────────
#    1 │    3.02281e5  A           4786.46   6320.72  0.0158345  0.0209101
#    2 │    3.35097e5  B           4564.08   6051.5   0.0136202  0.018059
#    3 │    3.62622e5  C           5734.84   7574.16  0.0158149  0.0208872
#    4 │    1.0e6      Total      14439.7   18980.1   0.0144397  0.0189801

covar = cov(Matrix(toFit[!, stocks]))
nsimRets  =DataFrame(simulateNormal(nSim, covar; seed=1234),stocks)

st = time()
values = crossjoin(mSRPortfolio, DataFrame(:iteration=>iteration))

nVals = size(values,1)
currentValue = Vector{Float64}(undef,nVals)
simulatedValue = Vector{Float64}(undef,nVals)
pnl = Vector{Float64}(undef,nVals)
Threads.@threads for i in 1:nVals
    price = startPrices[1,values.Symbol[i]]
    currentValue[i] = values.Holding[i] * price
    simulatedValue[i] = values.Holding[i] * price*(1.0+nsimRets[values.iteration[i],values.Symbol[i]])
    pnl[i] = simulatedValue[i] - currentValue[i]
end
values[!,:currentValue] = currentValue
values[!,:simulatedValue] = simulatedValue
values[!,:pnl] = pnl

println("Valuation Took $(time()-st)")
values[!,:Portfolio] = String.(values.Portfolio)
nriskValues = aggRisk(values,[:Portfolio])[!,[:currentValue, :Portfolio, :VaR95, :ES95, :VaR95_Pct, :ES95_Pct]]
# 4×6 DataFrame
#  Row │ currentValue  Portfolio  VaR95     ES95      VaR95_Pct  ES95_Pct  
#      │ Float64       String     Float64   Float64   Float64    Float64
# ─────┼───────────────────────────────────────────────────────────────────
#    1 │    3.02281e5  A           4547.87   5689.33  0.0150452  0.0188213
#    2 │    3.35097e5  B           4394.51   5494.09  0.0131142  0.0163955
#    3 │    3.62622e5  C           5417.77   6790.52  0.0149405  0.0187261
#    4 │    1.0e6      Total      13606.0   17005.5   0.013606   0.0170055

# Part 5
# Using your best fit risk model, calculate a risk parity portfolio for each subportfolio using ES as the risk metric.
# Rerun your attribution from Part 1 using the new optimal portfolios with the previously fit Beta.
# Discuss the results comparing back to Part 1 and Part 2.  

function riskParityES(simReturn; riskBudget=[], printLevel::Int=0)

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", printLevel)
    # set_attribute(model, "tol", 1e-4)
    set_attribute(model, "max_iter", 1000)

    n = size(simReturn,2)
    m = size(riskBudget,1)

    if !isempty(riskBudget)
        if m != n
            return(nothing, "Risk Budget not Correct Size")
        end
    else
        riskBudget = fill(1.0,n)
    end

    mult = riskBudget.^(-1)

    start = fill(1.0/n,n)

    @variable(model, w[i=1:n] >= 0, start=start[i])

    # internal ES function
    function _ES(w...)
        x = collect(w)
        r = simReturn*x 
        ES(r)
    end

    # Function for the component ES
    function CES(w...)
        x = collect(w)
        n = size(x,1)
        ces = Vector{Any}(undef,n)
        es = _ES(x...)
        e = 1e-6
        for i in 1:n
            old = x[i]
            x[i] = x[i]+e
            ces[i] = old*(_ES(x...) - es)/e
            x[i] = old
        end
        ces
    end

    # SSE of the Component ES
    function SSE_CES(w...)
        ces = CES(w...)
        ces = ces .- mean(ces)
        (ces'*ces)
    end

    if printLevel > 0
        # println("Starting Values: ", x)
        start = fill(1/n,n)
        # csd = pCSD(start...)
        # println("Starting Component Risk ", csd)
        println("Starting SSE Component Risk: ", SSE_CES(start...))

    end

    register(model,:distSSE,n,SSE_CES; autodiff = true)
    @NLobjective(model,Min, distSSE(w...))
    @constraint(model, sum(w)==1.0)
    optimize!(model)

    x = value.(w)/sum(value.(w))
    status = raw_status(model)

    if printLevel > 0
        # println("Found Value: ", value.(w))
        # println("Normalized Value: ", x)
        println("Solve Status - ", status)
        es = _ES(x...)
        println("Found ES: ", es)
        # println("Objective ", objective_value(model))
        ces = CES(x...)
        pct_RB = riskBudget ./ sum(riskBudget)
        pct_csd = ces ./ sum(ces)
        # println("Perctent Component Risk ", pct_csd)
        # println("Perctent Component Tgt  ", pct_RB)
        mDiff = max( abs.(pct_RB - pct_csd)...)
        println("Max Abs Pct Component Risk Diff From Tgt ", mDiff)
        println("SSE Component Risk: ", SSE_CES(x...))

    end

    return(x, status)
end

w,s = riskParityES(Matrix(simRets[!,stocks]); printLevel=5)
esPortfolio = portfolio[:,:]
esPortfolio[!,:Holding] = w .* 1000000 ./ vec(Matrix(startPrices[!, stocks]))

esAttribs = Dict{String, ExPostAttribution}()
RunAttribution(realizedReturns,realizedSPY,lastDate,startPrices, esPortfolio, esAttribs)

# Total Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY         rf          Alpha        Portfolio  
#      │ String              Float64     Float64     Float64      Float64
# ─────┼─────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781    0.0522772   0.0130448    0.241739
#    2 │ Return Attribution  0.169532    0.0568619   0.0153448    0.241739
#    3 │ Vol Attribution     0.00607829  2.47822e-5  0.000177111  0.00628018
# A Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY         rf          Alpha         Portfolio  
#      │ String              Float64     Float64     Float64       Float64
# ─────┼──────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781    0.0522772   -0.0457127    0.169703
#    2 │ Return Attribution  0.16404     0.0551493   -0.0494861    0.169703
#    3 │ Vol Attribution     0.00577786  1.79302e-5   0.000419761  0.00621555
# B Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY         rf          Alpha        Portfolio  
#      │ String              Float64     Float64     Float64      Float64
# ─────┼─────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781    0.0522772   0.0291168    0.25642
#    2 │ Return Attribution  0.165993    0.0572058   0.0332209    0.25642
#    3 │ Vol Attribution     0.00533559  2.72496e-5  0.000980874  0.00634371
# C Portfolio Attribution
# 3×5 DataFrame
#  Row │ Value               SPY         rf          Alpha        Portfolio  
#      │ String              Float64     Float64     Float64      Float64
# ─────┼─────────────────────────────────────────────────────────────────────
#    1 │ TotalReturn         0.198781    0.0522772   0.057741     0.301824
#    2 │ Return Attribution  0.178892    0.058267    0.0646655    0.301824
#    3 │ Vol Attribution     0.00626228  2.54413e-5  0.000921734  0.00720946

for p in ["Total", "A","B","C"]
    tv = esAttribs[p].Attribution[3,:Portfolio]
    nAhead = length(esAttribs["Total"].PortfolioReturn)
    tv *= sqrt(nAhead)
    tSR = (esAttribs[p].Attribution[2,:Portfolio] - esAttribs[p].Attribution[2,:rf]) / tv
    println("Portfolio $p Sharpe Ratio: ", tSR)
end

# Portfolio Total Sharpe Ratio: 1.8471165641303853
# Portfolio A Sharpe Ratio: 1.1564121616220644
# Portfolio B Sharpe Ratio: 1.9704255649653721
# Portfolio C Sharpe Ratio: 2.1197340512065934
