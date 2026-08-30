using DataFrames
using Plots
using Distributions
using CSV
using Dates
using LoopVectorization
using LinearAlgebra
using StatsBase
using JuMP
using Ipopt
using Random
using Printf

include("../Week04/return_calculate.jl")
include("fitted_model.jl")
include("simulate.jl")
include("RiskStats.jl")
include("../library/multivariate_t.jl")
include("../library/copula.jl")

prices = CSV.read("DailyPrices.csv",DataFrame)
#current Prices
current_prices = prices[size(prices,1),:]

#discrete returns
returns = return_calculate(prices,dateColumn="Date")

nms = names(returns)
nms = nms[nms.!="Date"]
nms = nms[nms.!="PLD"]
#remove date column
returns = returns[!,nms]

#all stock names
stocks = nms[nms.!="SPY"]

#setup how much we hold
Portfolio = DataFrame(:stock=>stocks, :holding => fill(1.0,size(stocks,1)))


#remove the mean from all returns:
for nm in nms
    v = returns[!,nm]
    returns[!,nm] = v .- mean(v)
end

st = time()
#fit model for all stocks
fittedModels = Dict{String,FittedModel}()

fittedModels["SPY"] = fit_normal(returns.SPY)

for stock in stocks
    fittedModels[stock] = fit_regression_t(returns[!,stock],returns.SPY)
end
println("Model Fitting Took $(time()-st)")

st = time()

#construct the copula:
#Start the data frame with the U of the SPY - we are assuming normallity for SPY
U = DataFrame()
for nm in nms
    U[!,nm] = fittedModels[nm].u
end
Umat = Matrix(U)

m = size(Umat,1)   #observations
n = size(Umat,2)   #variables

R = corspearman(Umat)

#what's the rank of R
evals = eigvals(R)
if min(evals...) > -1e-8
    println("Matrix is PSD")
else
    println("Matrix is not PSD")
end

#-------------------------------------------------------------------------
# Which copula should we use?
#-------------------------------------------------------------------------
# Both copulas get the same margins and the same R. R here comes from Kendall's
# tau, because rho = sin(pi*tau/2) is exact for the Gaussian and the t alike.
# Handing the two models different correlation matrices would confound a
# difference in R with a difference in copula family.

tau = corkendall(Umat)
Rk = sin.(pi .* tau ./ 2)
for i in 1:n
    Rk[i,i] = 1.0
end

# The tau matrix is PSD on complete data. The entrywise sin() transform is not
# guaranteed to preserve that, so check and repair. fix_correlation does the
# repair; the eigenvalue is reported here so the reader sees whether it fired.
# The copula densities below need a Cholesky root, so a merely PSD R is not
# enough -- it has to come back positive definite.
function report_psd(R)
    minEig = min_eigenvalue(R)
    if minEig < -1e-8
        @printf("R is not PSD (min eigenvalue %.6f). Repairing.\n", minEig)
    else
        @printf("R is PSD. Min eigenvalue %.6f\n", minEig)
    end
end

report_psd(Rk)
Rk = fix_correlation(Rk)

# Make the choice on a subset of the columns, not on all of them.
#
# A copula on n variables carries n(n-1)/2 correlations. With n = $n and only
# m = $m observations, R is estimated so poorly that it comes out nearly
# singular, and a copula likelihood built on its inverse is numerically
# meaningless -- it will report log likelihoods in the millions. The family
# choice is a question about the shape of the joint tail, and a well
# conditioned handful of names answers it.
selectOn = ["SPY","AAPL","MSFT","JPM","XOM"]
sel = [findfirst(==(s),nms) for s in selectOn]
Usel = Umat[:,sel]
Rsel = sin.(pi .* corkendall(Usel) ./ 2)
for i in 1:length(sel)
    Rsel[i,i] = 1.0
end
report_psd(Rsel)
Rsel = fix_correlation(Rsel)

# Profile nu on theta = 1/nu so the resolution goes where it matters.
thetas = range(0.01,0.49,length=100)
lls = [t_copula_ll(Usel,Rsel,1.0/th) for th in thetas]
i = argmax(lls)
lo = thetas[max(i-1,1)]
hi = thetas[min(i+1,length(thetas))]
fine = range(lo,hi,length=100)
fineLls = [t_copula_ll(Usel,Rsel,1.0/th) for th in fine]
j = argmax(fineLls)

nuhat = 1.0/fine[j]
llT = fineLls[j]
llG = gaussian_copula_ll(Usel,Rsel)

# The Gaussian is the t at nu = infinity, so the two models differ by exactly
# one free parameter.
bic(ll,k,m) = k*log(m) - 2*ll
dBIC = 2*(llT - llG) - log(m)

println("Copula family chosen on: ", join(selectOn,", "))
@printf("Gaussian copula ll %.2f   BIC %.2f\n", llG, bic(llG,0,m))
@printf("t copula ll        %.2f   BIC %.2f   nu = %.1f\n", llT, bic(llT,1,m), nuhat)
@printf("delta BIC = %.2f  -->  the %s copula fits better\n", dBIC, dBIC > 0 ? "t" : "Gaussian")

println("Copula Fitting Took $(time()-st)")

#-------------------------------------------------------------------------
# Simulation
#-------------------------------------------------------------------------
NSim = 5000

# Gaussian copula: correlated standard normals, then the normal CDF.
# t copula: the same correlated normals scaled by one chi-square draw per
# iteration, then the t CDF with the copula's nu. Both live in library/copula.jl.

# Take the simulated U values back through the fitted models.
function simulate_returns(simU)
    simulatedReturns = DataFrame(:SPY => fittedModels["SPY"].eval(simU.SPY))
    for stock in stocks
        simulatedReturns[!,stock] = fittedModels[stock].eval(simulatedReturns.SPY,simU[!,stock])
    end
    return simulatedReturns
end

#-------------------------------------------------------------------------
# Valuation and risk
#-------------------------------------------------------------------------
function portfolio_risk(simulatedReturns)
    nsim = size(simulatedReturns,1)
    iteration = [i for i in 1:nsim]
    values = crossjoin(Portfolio, DataFrame(:iteration=>iteration))

    nVals = size(values,1)
    currentValue = Vector{Float64}(undef,nVals)
    simulatedValue = Vector{Float64}(undef,nVals)
    pnl = Vector{Float64}(undef,nVals)
    for i in 1:nVals
        price = current_prices[values.stock[i]]
        currentValue[i] = values.holding[i] * price
        simulatedValue[i] = values.holding[i] * price*(1.0+simulatedReturns[values.iteration[i],values.stock[i]])
        pnl[i] = simulatedValue[i] - currentValue[i]
    end
    values[!,:currentValue] = currentValue
    values[!,:simulatedValue] = simulatedValue
    values[!,:pnl] = pnl

    #Stock Level Metrics
    gdf = groupby(values,:stock)

    stockRisk = combine(gdf,
        :currentValue => (x-> first(x,1)) => :currentValue,
        :pnl => (x -> VaR(x,alpha=0.05)) => :VaR95,
        :pnl => (x -> ES(x,alpha=0.05)) => :ES95,
        :pnl => (x -> VaR(x,alpha=0.01)) => :VaR99,
        :pnl => (x -> ES(x,alpha=0.01)) => :ES99,
        :pnl => std => :Standard_Dev,
        :pnl => (x -> [extrema(x)]) => [:min, :max],
        :pnl => mean => :mean
    )

    #Total Metrics
    gdf = groupby(values,:iteration)
    #aggregate to totals per simulation iteration
    totalValues = combine(gdf,
        :currentValue => sum => :currentValue,
        :simulatedValue => sum => :simulatedValue,
        :pnl => sum => :pnl
    )

    #calculate Risk
    totalRisk = combine(totalValues,
        :currentValue => (x-> first(x,1)) => :currentValue,
        :pnl => (x -> VaR(x,alpha=0.05)) => :VaR95,
        :pnl => (x -> ES(x,alpha=0.05)) => :ES95,
        :pnl => (x -> VaR(x,alpha=0.01)) => :VaR99,
        :pnl => (x -> ES(x,alpha=0.01)) => :ES99,
        :pnl => std => :Standard_Dev,
        :pnl => (x -> [extrema(x)]) => [:min, :max],
        :pnl => mean => :mean
    )

    totalRisk[!,:stock] = ["Total"]

    return vcat(stockRisk, totalRisk), totalValues
end

st = time()

simU_g = DataFrame(simulate_gaussian_copula(R,NSim), nms)
simU_t = DataFrame(simulate_t_copula(Rk,nuhat,NSim), nms)

riskGauss, totalGauss = portfolio_risk(simulate_returns(simU_g))
riskT, totalT = portfolio_risk(simulate_returns(simU_t))

println("Simulation and Valuation Took $(time()-st)")

CSV.write("ExampleRisk.csv",riskGauss)
CSV.write("ExampleRisk_tCopula.csv",riskT)

#-------------------------------------------------------------------------
# What the copula choice actually changed
#-------------------------------------------------------------------------
# The portfolio risk numbers barely move between the two copulas, which is the
# expected result. Watch the count of iterations on which nearly everything
# loses at once, because that is the event the copula choice is actually about.
# On these residuals the fitted nu comes back near 40, so the t copula is
# almost the Gaussian and the two counts land on top of each other. Re-run the
# selection on raw returns instead of residuals and they separate.

function broad_loss_days(simulatedReturns; threshold=0.9)
    cols = [simulatedReturns[!,nm] for nm in stocks]
    need = threshold*length(cols)
    return count(i -> count(c -> c[i] < 0, cols) >= need, 1:size(simulatedReturns,1))
end

retG = simulate_returns(simU_g)
retT = simulate_returns(simU_t)

g = riskGauss[riskGauss.stock .== "Total", :]
t_ = riskT[riskT.stock .== "Total", :]

println("\nTotal portfolio, $NSim iterations")
@printf("  %-14s %14s %14s\n", "", "Gaussian", "t")
@printf("  %-14s %14.2f %14.2f\n", "VaR 95", g.VaR95[1], t_.VaR95[1])
@printf("  %-14s %14.2f %14.2f\n", "ES 95",  g.ES95[1],  t_.ES95[1])
@printf("  %-14s %14.2f %14.2f\n", "VaR 99", g.VaR99[1], t_.VaR99[1])
@printf("  %-14s %14.2f %14.2f\n", "ES 99",  g.ES99[1],  t_.ES99[1])
@printf("  %-14s %14d %14d\n", "90% lose", broad_loss_days(retG), broad_loss_days(retT))
