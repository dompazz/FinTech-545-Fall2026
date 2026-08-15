using MarketData
using Dates
using DataFrames
using XLSX
using CSV
using LoopVectorization

top100  = XLSX.readtable("holdings-daily-us-en-spy.xlsx","top100") |> DataFrame

tickers = vcat(["SPY"],top100.Ticker)

histPrices = DataFrame[]

for t in Symbol.(tickers)
    df = DataFrames.rename(DataFrame(yahoo(t, YahooOpt(period1=DateTime(2022,9,1)))), Dict(:timestamp=>:Date, :AdjClose=>t))[!,[:Date, t]]
    append!(histPrices,[df])
end

prices = innerjoin(histPrices...,on=:Date)

CSV.write("DailyPrices.csv",prices)

include("return_calculate.jl")

returns = return_calculate(prices,dateColumn="Date")

CSV.write("DailyReturn.csv",returns)

ps = vcat(
    ["A" for i in 1:33],
    ["B" for i in 1:33],
    ["C" for i in 1:34]
)
portfolio = DataFrame(:Portfolio=>ps,:Stock=>top100.Ticker,:Holding=>rand(50:200,100))

CSV.write("Project/portfolio.csv",portfolio)