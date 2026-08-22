using Ipopt
using JuMP
using LinearAlgebra
# using HiGHS
using DataFrames
# using Roots
# using Statistics
# using GaussianMixtures
using Random

function riskParity(cov::AbstractArray{Float64,2}; riskBudget=[], printLevel::Int=0)

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", printLevel)
    n = size(cov,1)
    m = size(riskBudget,1)

    if !isempty(riskBudget)
        if m != n
            return(nothing, "Risk Budget not Correct Size")
        end
    else
        riskBudget = fill(1.0,n)
    end

    mult = riskBudget.^(-1)

    @variable(model, w[1:n] >= 0, start=1/n)

    function pvol(w...)
        x = collect(w)
        # println(x)
        return(sqrt(x'*cov*x))
    end

    function pCSD(w...)
        x = collect(w)
        pVol = pvol(w...)
        csd = x.*(cov*x)./pVol
        return (csd)
    end

    function sseCSD(w...)
        csd = mult.*pCSD(w...)
        mCSD = sum(csd)/n
        dCsd = csd .- mCSD
        se = dCsd .*dCsd
        return(1.0e5*sum(se))
    end

    # register(model,:vol,n,pvol,autodiff=true)
    register(model,:distCSD,n,sseCSD,autodiff=true)

    # @NLobjective(model,Min,vol(w...))
    @NLobjective(model,Min,distCSD(w...))
    @constraint(model, (sum(w[i] for i in 1:n))==1)
    # @NLconstraint(model, (sum(log(w[i]) for i in 1:n))==1)


    if printLevel > 0
        # println("Starting Values: ", x)
        start = fill(1/n,n)
        # csd = pCSD(start...)
        # println("Starting Component Risk ", csd)
        println("Starting SSE Component Risk: ", sseCSD(start...))

    end

    optimize!(model)
    x = value.(w)/sum(value.(w))
    status = raw_status(model)

    if printLevel > 0
        # println("Found Value: ", value.(w))
        # println("Normalized Value: ", x)
        println("Solve Status - ", status)
        pVol = pvol(x...)
        println("Found Vol: ", pVol)
        # println("Objective ", objective_value(model))
        csd = pCSD(x...)
        pct_RB = riskBudget ./ sum(riskBudget)
        pct_csd = csd ./ sum(csd)
        # println("Perctent Component Risk ", pct_csd)
        # println("Perctent Component Tgt  ", pct_RB)
        mDiff = max( abs.(pct_RB - pct_csd)...)
        println("Max Abs Pct Component Risk Diff From Tgt ", mDiff)
        println("SSE Component Risk: ", sseCSD(x...))

    end

    return(x, status)
end

function maxSR(cov::AbstractArray{Float64,2},mean::AbstractArray{Float64,1},rf::Float64;printLevel::Int=0)
    n = size(cov,1)
    bounds = fill(0.0,(n,2))
    bounds[:,2] .= 1.0
    maxSR(cov,mean,rf,bounds;printLevel=printLevel)
end


function maxSR(cov::AbstractArray{Float64,2},mean::AbstractArray{Float64,1},rf::Float64, bounds::AbstractArray{Float64,2};printLevel::Int=0)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", printLevel)
    n = size(cov,1)

    @variable(model, w[1:n], start=1/n)

    function _sr(w...)
        x = collect(w)
        _vol = sqrt(x'*cov*x)
        (x'*mean - rf)/_vol
    end

    register(model,:sr,n,_sr,autodiff=true)

    @NLobjective(model,Max,sr(w...))
    @constraint(model, (sum(w[i] for i in 1:n))==1)
    for i in 1:n
        @constraint(model, w[i] >= bounds[i,1])
        @constraint(model, w[i] <= bounds[i,2])
    end

    optimize!(model)
    x = value.(w)
    status = raw_status(model)
    return(x, status)

end

function maxESR(simReturn::AbstractArray{Float64,2},rf::Float64, bounds::AbstractArray{Float64,2};printLevel::Int=0)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", printLevel)    
    set_attribute(model, "tol", 1e-3)
    set_attribute(model, "max_iter", 1000)
    n = size(simReturn,2)

    @variable(model, w[1:n], start=1/n)

    # internal ES function
    function _ES(w...)
        x = collect(w)
        r = simReturn*x 
        ES(r)
    end

    function _sr(w...)
        x = collect(w)
        _vol = _ES(x...)
        m = mean(simReturn*x)
        (m - rf)/_vol
    end

    register(model,:sr,n,_sr,autodiff=true)

    @NLobjective(model,Max,sr(w...))
    @constraint(model, (sum(w[i] for i in 1:n))==1)
    for i in 1:n
        @constraint(model, w[i] >= bounds[i,1])
        @constraint(model, w[i] <= bounds[i,2])
    end

    optimize!(model)
    x = value.(w)
    status = raw_status(model)
    return(x, status)

end

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
        ces = mult.*CES(w...)
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


tailSkew(x,rf,alpha=.05) = (ES(-x,alpha=alpha)-rf) / ES(x,alpha=alpha)

function maxTailSkew(factors::AbstractArray{Float64,2}, rf::Float64, bounds::AbstractArray{Float64,2};alpha::Float64=0.05, printLevel::Int=0)

    function _TS(w...)
        _w = collect(w)
        r = factors * _w
        tailSkew(r,rf,alpha)
        # std(r)
    end
    
    function ∇_TS(grad, w...)
        x = collect(w)
        n = size(x,1)
        # ces = Vector{Any}(undef,n)
        es = _TS(x...)
        e = 1e-6
        for i in 1:n
            old = x[i]
            x[i] = x[i]+e
            grad[i] = (_TS(x...) - es)/e
            x[i] = old
        end
    end

    n = size(factors,2)

    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", printLevel)
    set_optimizer_attribute(m, "max_iter", 1000)
    set_optimizer_attribute(m, "tol", 5e-2)

    @variable(m, w[i=1:n] ,start=1/n)

    register(m,:func,n,_TS,∇_TS)

    @NLobjective(m,Max, func(w...))

    @constraint(m, sum(w)==1.0)
    @constraint(m, minc, w .>= bounds[:,1])
    @constraint(m, maxc, w .<= bounds[:,2])

    optimize!(m)
    x = value.(w)
    status = raw_status(m)
    return(x, status)
end