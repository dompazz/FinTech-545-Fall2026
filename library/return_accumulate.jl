function return_accumulate(returns::DataFrame;
    method="DISCRETE",
    toFreq="MONTH",
    dateColumn="date"
)

if uppercase(toFreq) in ("DAY", "DAILY")
    # println("Daily")
    byVars = [:year, :month, :day]
    _selector = "YMD"
elseif uppercase(toFreq) in ("MONTH","MONTHLY","MTH")
    # println("Monthly")
    byVars = [:year, :month]
    _selector = "YM"
elseif uppercase(toFreq) in ("QTR","QUARTERLY","QUARTER")
    # println("Quarterly")
    byVars = [:year, :qtr]
    _selector = "YQ"
elseif uppercase(toFreq) in ("YR","YEARLY","YEAR")
    # println("Yearly")
    byVars = [:year]
    _selector = "Y"
else
    throw(ArgumentError(string("toFreq: ", toFreq, " must be one of ",
            ("DAY", "DAILY")," / ",
            ("MONTH","MONTHLY","MTH")," / ",
            ("QTR","QUARTERLY","QUARTER")," / ",
            ("YR","YEARLY","YEAR")
            )))
end

# println(byVars)

vars = names(returns)
nVars = length(vars)
vars = Symbol.(vars[vars.!=dateColumn])
if nVars == length(vars)
    throw(ArgumentError(string("dateColumn: ", dateColumn, " not in DataFrame: ",vars)))
end
nVars = nVars-1
n = size(returns,1)
dc = Symbol(dateColumn)

out = returns[:, vars]
dates = returns[:,dc]
out[!,:year] =  Dates.Year.(dates)
if findfirst("M", _selector) != nothing
    out[!,:month] = Dates.Month.(dates)
end
if findfirst("Q", _selector) != nothing
    out[!,:qtr] = Dates.quarterofyear.(dates)
end
if findfirst("D", _selector) != nothing
    out[!,:day] = Dates.Day.(dates)
end


if uppercase(method) == "DISCRETE"
    out = DataFrames.rename(
            combine(
                groupby(out,byVars),
                [vars[i] => aggDiscreteReturns for i in 1:size(vars,1)]
            ),
            Dict(Symbol(String(string(vars[i],"_aggDiscreteReturns"))) => vars[i] for i in 1:size(vars,1))
        )
elseif uppercase(method) == "LOG"
    out = DataFrames.rename(
            combine(
                groupby(out,byVars),
                [vars[i] => sum for i in 1:size(vars)[1]]
            ),
            Dict(Symbol(String(string(vars[i],"_sum"))) => vars[i] for i in 1:size(vars,1))
        )
else
    throw(ArgumentError(string("method: ", method, " must be in (\"LOG\",\"DISCRETE\")")))
end

if "YM" == _selector
    out[!, dc] = Date.(out.year,out.month).+ Dates.Month(1) .- Dates.Day(1)
elseif "YQ"  == _selector
    out[!, dc] = Date.(out.year) + Dates.Month.( out.qtr .* 3) .- Dates.Day(1)
elseif "YMD" == _selector
    out[!, Symbol(dateColumn)] = Date.(out.year,out.month,out.day)
else
    out[!, Symbol(dateColumn)] = Date.(out.year) .+ Dates.Year(1) .- Dates.Day(1)
end

return(out[!, vcat(dc, vars)])
end


function aggDiscreteReturns(returns)
    n = size(returns,1)
    p = 1.0
    @inbounds @avx for i ∈ 1:n
        p = p * (returns[i] + 1.0)
    end
    return(p-1.0)
end