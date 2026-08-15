
#Type to hold model outputs
struct FittedModel
    beta::Union{Vector{Float64},Nothing}
    errorModel::UnivariateDistribution
    eval::Function
    errors::Vector{Float64}
    u::Vector{Float64}
end

#OLS
function fit_ols(y,x)
    n = size(x,1)
    __x = hcat(fill(1.0,n),x)
    __y = y
    beta = inv(__x'*__x)*__x'*__y
    e = __y - __x*beta
    std_e = std(e)

    #Define the fitted error model
    errorModel = Normal(0,std_e)
    u = cdf(errorModel,e)

    #function to evaluate the model for a given x and u
    function eval_model(x,u)
        n = size(x,1)
        _temp = hcat(fill(1.0,n),x)
        return _temp*beta .+ quantile(errorModel,u)
    end

    return FittedModel(beta, errorModel, eval_model, e, u)
end

#general t sum ll function
function general_t_ll(mu,s,nu,x)
    td = TDist(nu)*s + mu
    sum(log.(pdf.(td,x)))
end

#fit regression model with MLE, normal errors
function fit_regression_mle(y,x)
    n = size(x,1)

    global __x, __y
    __x = hcat(fill(1.0,n),x)
    __y = y

    nB = size(__x,2)

    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    @variable(mle, m)
    @variable(mle, s>=1e-6, start=1)
    @variable(mle, B[i=1:nB],start=0)
    @constraint(mle, m==0)

    #Inner function to abstract away the X value
    function _gtl(mu,s,B...)
        beta = collect(B)
        xm = __y - __x*beta
        sum(log.(pdf.(Normal(mu,s),xm)))
    end

    register(mle,:tLL,nB+2,_gtl;autodiff=true)
    @NLobjective(
        mle,
        Max,
        tLL(m, s, B...)
    )
    optimize!(mle)

    m = value(m) #Should be 0 or very near it.
    s = value(s)
    beta = value.(B)

    #Define the fitted error model
    errorModel = Normal(0,s)

    #function to evaluate the model for a given x and u
    function eval_model(x,u)
        n = size(x,1)
        _temp = hcat(fill(1.0,n),x)
        return _temp*beta .+ quantile(errorModel,u)
    end

    #Calculate the regression errors and their U values
    errors = y - eval_model(x,fill(0.5,size(x,1)))
    u = cdf(errorModel,errors)

    return FittedModel([beta...,s], errorModel, eval_model, errors, u,)
end

#fit regression model with T errors
function fit_regression_t(y,x)
    n = size(x,1)

    global __x, __y
    __x = hcat(fill(1.0,n),x)
    __y = y

    nB = size(__x,2)

    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    #approximate values based on moments and OLS
    b_start = inv(__x'*__x)*__x'*__y
    e = __y - __x*b_start
    start_m = mean(e)
    start_nu = 6.0/kurtosis(e) + 4
    start_s = sqrt(var(e)*(start_nu-2)/start_nu)

    @variable(mle, m)
    @variable(mle, s>=1e-6, start=1)
    @variable(mle, nu>=2.0001, start=start_s)
    @variable(mle, B[i=1:nB],start=b_start[i])
    @constraint(mle, m==0)

    #Inner function to abstract away the X value
    function _gtl(mu,s,nu,B...)
        beta = collect(B)
        xm = __y - __x*beta
        general_t_ll(mu,s,nu,xm)
    end

    register(mle,:tLL,nB+3,_gtl;autodiff=true)
    @NLobjective(
        mle,
        Max,
        tLL(m, s, nu, B...)
    )
    optimize!(mle)

    m = value(m) #Should be 0 or very near it.
    s = value(s)
    nu = value(nu)
    beta = value.(B)

    #Define the fitted error model
    errorModel = TDist(nu)*s

    #function to evaluate the model for a given x and u
    function eval_model(x,u)
        n = size(x,1)
        _temp = hcat(fill(1.0,n),x)
        return _temp*beta .+ quantile(errorModel,u)
    end

    #Calculate the regression errors and their U values
    errors = y - eval_model(x,fill(0.5,size(x,1)))
    u = cdf(errorModel,errors)

    return FittedModel([beta...,s,nu], errorModel, eval_model, errors, u)
end