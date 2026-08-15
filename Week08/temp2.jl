
using BenchmarkTools
using InlineStrings
using DataFrames
using Distributions

include("../library/gbsm.jl")

struct Instrument
    call::Bool
    strike::Float64
    ttm::Float64
    b::Float64
    ivol::Float64
    cc1::Vector{String}
    p::Function
end

state = DataFrame(price=randn(100).+100, rf=randn(100)*.001 .+ .05)

function pricer(state,instrument)
    cc1 = instrument.cc1[1]
    return(cc1, gbsm(instrument.call, 
        state.price[end], 
        instrument.strike, 
        instrument.ttm, 
        state.rf[end], 
        instrument.b, 
        instrument.ivol).value)
end
function randstring15()
    len = rand(1:15)
    return String(rand('A':'Z', len))
end

n = 1000
portfolio = Vector{Instrument}(undef, n)
for i in 1:n
    portfolio[i] = Instrument(
        rand(Bool), 
        randn() + 100, 
        rand(0.01:1), 
        rand(-0.1:0.1), 
        rand(0.1:0.5), 
        [randstring15()],
        pricer
    )
end

values = [instrument.p(state, instrument) for instrument in portfolio]

@benchmark values = [instrument.p(state, instrument) for instrument in portfolio]

i = Instrument(
        rand(Bool), 
        randn() + 100, 
        rand(0.01:1), 
        rand(-0.1:0.1), 
        rand(0.1:0.5), 
        randstring15(),
        pricer
    )