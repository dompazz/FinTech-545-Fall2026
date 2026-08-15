using MarketData
using Dates
using DataFrames
using CSV
using LoopVectorization

top100  = CSV.read("Project01/holdingsSPY.csv", DataFrame)

tickers = vcat(["SPY"],top100.Ticker)

histPrices = DataFrame[]

for t in Symbol.(tickers)
    print(t)
    df = DataFrames.rename(DataFrame(yahoo(t, YahooOpt(period1=DateTime(2023,1,1)))), Dict(:timestamp=>:Date, :AdjClose=>t))[!,[:Date, t]]
    println(" $(min(df.Date...))")
    append!(histPrices,[df])
end

prices = innerjoin(histPrices...,on=:Date)

CSV.write("DailyPrices.csv",prices)

include("../../library/return_calculate.jl")

returns = return_calculate(prices,dateColumn="Date")

CSV.write("DailyReturn.csv",returns)
