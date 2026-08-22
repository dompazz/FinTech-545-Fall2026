
#Type to hold model outputs
struct FittedModel
    beta::Union{Vector{Float64},Nothing}
    errorModel::UnivariateDistribution
    eval::Function
    errors::Vector{Float64}
    u::Vector{Float64}
end

#AICC
function aicc(d::T,data) where T <: UnivariateDistribution
    ll = sum(logpdf.(d,data))
    n = size(data,1)
    k = size(Distributions.params(d),1)
    return -2*ll + 2*k + 2*k*(k+1)/(n-k-1)
end

function aicc(m::FittedModel,data)
    d = m.errorModel
    ll = sum(logpdf.(d,data))
    n = size(data,1)
    k = size(Distributions.params(d),1)
    if !isnothing(m.beta)
        k += size(m.beta,1)
    end
    return -2*ll + 2*k + 2*k*(k+1)/(n-k-1)
end

function aic(m::FittedModel,data)
    d = m.errorModel
    ll = sum(logpdf.(d,data))
    n = size(data,1)
    k = size(params(d),1)
    if !isnothing(m.beta)
        k += size(m.beta,1)
    end
    return -2*ll + 2*k 
end

#general t sum ll function
function general_t_ll(mu,s,nu,x)
    td = TDist(nu)*s + mu
    sum(log.(pdf.(td,x)))
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

    return FittedModel(beta, errorModel, eval_model, errors, u)
end

#MLE for a Generalize T
function fit_general_t(x)
    global __x
    __x = x
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    #approximate values based on moments
    start_m = mean(x)
    start_nu = 6.0/kurtosis(x) + 4
    start_s = sqrt(var(x)*(start_nu-2)/start_nu)

    @variable(mle, m, start=start_m)
    @variable(mle, s>=1e-6, start=1)
    @variable(mle, nu>=2.0001, start=start_s)

    #Inner function to abstract away the X value
    function _gtl(mu,s,nu)
        general_t_ll(mu,s,nu,__x)
    end

    register(mle,:tLL,3,_gtl;autodiff=true)
    @NLobjective(
        mle,
        Max,
        tLL(m, s, nu)
    )
    optimize!(mle)
    if !(string(termination_status(mle)) ∈ ["OPTIMAL", "LOCALLY_SOLVED","ALMOST_OPTIMAL","ALMOST_LOCALLY_SOLVED"])
        throw(ErrorException("Opimization Failed - $(termination_status(mle))"))
    end
    m = value(m)
    s = value(s)
    nu = value(nu)

    #create the error model
    errorModel = TDist(nu)*s + m
    #calculate the errors and U
    errors = x .- m
    u = cdf(errorModel,x)

    eval(u) = quantile(errorModel,u)

    return FittedModel(nothing, errorModel, eval, errors, u)

    #return the parameters as well as the Distribution Object
    # return (m, s, nu, TDist(nu)*s+m)
end


function fit_normal(x)
    #Mean and Std values
    m = mean(x)
    s = std(x)
    
    #create the error model
    errorModel = Normal(m,s)
    #calculate the errors and U
    errors = x .- m
    u = cdf(errorModel,x)

    eval(u) = quantile(errorModel,u)

    return FittedModel(nothing, errorModel, eval, errors, u)

end

#general JohnsonSU sum ll function
function general_johnsonsu_ll(γ, ξ, δ, λ, x)
    d = JohnsonSU(γ, ξ, δ, λ)
    ll = sum(logpdf.(d,x))
    return ll
end


#MLE for a JohnsonSU

# Generic function that should help with convergence
function fit_general_johnsonsu(x)
    m = nothing
    try
        #Fastest if it works
        m = fit_general_johnsonsu(x, 0.0, 1.0, 1.0, 1.0)
    catch e
        #Error in MLE, try SMM, then use that to start the MLE
        println("Error in MLE fit $e, attempting to condition starting values")
        _prefit = fit_general_johnsonsu_mm(x)
        println("Prefit values $(params(_prefit.errorModel))")
        m = fit_general_johnsonsu(x, _prefit.errorModel.γ, _prefit.errorModel.ξ, _prefit.errorModel.δ, _prefit.errorModel.λ)
    end

    return m
end

function fit_general_johnsonsu(x, start_γ, start_ξ, start_δ, start_λ)
    global __x
    __x = x
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    @variable(mle, γ, start = start_γ)
    @variable(mle, ξ, start = start_ξ)
    @variable(mle, δ >= 1e-6, start = start_δ)
    @variable(mle, λ >= 1e-6, start = start_λ)

    #Inner function to abstract away the X value
    function _gtl(γ, ξ, δ, λ)
        general_johnsonsu_ll(γ, ξ, δ, λ, __x)
    end

    register(mle,:suLL,4,_gtl;autodiff=true)
    @NLobjective(
        mle,
        Max,
        suLL(γ, ξ, δ, λ)
    )
    optimize!(mle)
    if !(string(termination_status(mle)) ∈ ["OPTIMAL", "LOCALLY_SOLVED","ALMOST_OPTIMAL","ALMOST_LOCALLY_SOLVED"])
        throw(ErrorException("Opimization Failed - $(termination_status(mle))"))
    end
    g = value(γ) 
    e = value(ξ)
    d = value(δ)
    l = value(λ)

    #create the error model
    errorModel = JohnsonSU(g,e,d,l)
    
    #calculate the errors and U
    errors = x .- mean(errorModel)

    u = cdf(errorModel,x)

    eval(u) = quantile(errorModel,u)

    return FittedModel(nothing, errorModel, eval, errors, u)

    #return the parameters as well as the Distribution Object
    # return (m, s, nu, TDist(nu)*s+m)
end

#SMM for JohnsonSU
function fit_general_johnsonsu_smm(x)
    fit_general_johnsonsu_smm(x,100)
end

function fit_general_johnsonsu_smm(x,n=100)
    
    global __x
    __x = x
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    @variable(mle, γ, start = 0.0)
    @variable(mle, ξ , start = 0.0)
    @variable(mle, δ >= 1e-6, start = 1.0)
    @variable(mle, λ >= 1e-6, start = 1.0)

    global __m, __v, __sk, __k
    __m = mean(__x)
    __v = var(__x)
    __sk = skewness(__x)
    __k = kurtosis(__x)

    #Inner function to abstract away the X value
    function _gtl(γ, ξ, δ, λ)
        d = JohnsonSU(γ, ξ, δ, λ)
        # n=100
        means = Vector{Float64}(undef,n)
        vars = Vector{Float64}(undef,n)
        skews = Vector{Float64}(undef,n)
        kurts = Vector{Float64}(undef,n)
        r = Vector{Float64}(undef,500)
        rng = MersenneTwister(12345)
        for i in 1:n
            rand!(rng,d,r)
            means[i] = mean(r)
            vars[i] = var(r)
            skews[i] = skewness(r)
            kurts[i] = kurtosis(r)
        end

        s = 0.0
        s += (mean(means)-__m)^2
        s += (mean(vars)-__v)^2
        s += (mean(skews)-__sk)^2
        s += (mean(kurts)-__k)^2
        return s
    end

    function gfun!(grad,γ, ξ, δ, λ)
        s0 = _gtl(γ, ξ, δ, λ)
        _eps = 1e-8
        grad[1] = (_gtl(γ+_eps, ξ, δ, λ) - s0 ) / _eps
        grad[2] = (_gtl(γ, ξ+_eps, δ, λ) - s0 ) / _eps
        grad[3] = (_gtl(γ, ξ, δ+_eps, λ) - s0 ) / _eps
        grad[4] = (_gtl(γ, ξ, δ, λ+_eps) - s0 ) / _eps
    end

    register(mle,:suSMM,4,_gtl,gfun!;autodiff=false)
    @NLobjective(
        mle,
        Min,
        suSMM(γ, ξ, δ, λ)
    )
    optimize!(mle)
    # println(termination_status(mle))

    if !(string(termination_status(mle)) ∈ ["OPTIMAL", "LOCALLY_SOLVED","ALMOST_OPTIMAL","ALMOST_LOCALLY_SOLVED"])
        throw(ErrorException("Opimization Failed - $(termination_status(mle))"))
    end

    g = value(γ) 
    e = value(ξ)
    d = value(δ)
    l = value(λ)

    #create the error model
    errorModel = JohnsonSU(g,e,d,l)
    
    #calculate the errors and U
    errors = x .- mean(errorModel)

    u = cdf(errorModel,x)

    eval(u) = quantile(errorModel,u)

    return FittedModel(nothing, errorModel, eval, errors, u)

end


#MM for JohnsonSU
function fit_general_johnsonsu_mm(x;start_γ=0.0, start_ξ = 0.0, start_δ = 1.0, start_λ = 1.0)
   
    global __x
    __x = x
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    @variable(mle, γ, start = start_γ)
    @variable(mle, ξ, start = start_ξ)
    @variable(mle, δ >= 1e-6, start = start_δ)
    @variable(mle, λ >= 1e-6, start = start_λ)


    global __m, __v, __sk, __k
    __m = mean(__x)
    __v = var(__x)
    __sk = skewness(__x)
    __k = kurtosis(__x)

    #Inner function to abstract away the X value
    function _gtl(γ, ξ, δ, λ)
        d = JohnsonSU(γ, ξ, δ, λ)
        
        s = 0.0
        s += (mean(d)-__m)^2
        s += (var(d)-__v)^2
        s += (skewness(d)-__sk)^2
        s += (kurtosis(d)-__k)^2
        return s
    end

    function gfun!(grad,γ, ξ, δ, λ)
        s0 = _gtl(γ, ξ, δ, λ)
        _eps = 1e-8
        grad[1] = (_gtl(γ+_eps, ξ, δ, λ) - s0 ) / _eps
        grad[2] = (_gtl(γ, ξ+_eps, δ, λ) - s0 ) / _eps
        grad[3] = (_gtl(γ, ξ, δ+_eps, λ) - s0 ) / _eps
        grad[4] = (_gtl(γ, ξ, δ, λ+_eps) - s0 ) / _eps
    end

    register(mle,:suSMM,4,_gtl,gfun!;autodiff=false)
    # register(mle,:suSMM,4,_gtl;autodiff=true)
    @NLobjective(
        mle,
        Min,
        suSMM(γ, ξ, δ, λ)
    )
    optimize!(mle)

    if !(string(termination_status(mle)) ∈ ["OPTIMAL", "LOCALLY_SOLVED","ALMOST_OPTIMAL", "ALMOST_LOCALLY_SOLVED"])
        throw(ErrorException("Opimization Failed - $(termination_status(mle))"))
        # println("ERROR")
        # return termination_status(mle)
    end

    g = value(γ) 
    e = value(ξ)
    d = value(δ)
    l = value(λ)

    #create the error model
    errorModel = JohnsonSU(g,e,d,l)
    
    #calculate the errors and U
    errors = x .- mean(errorModel)

    u = cdf(errorModel,x)

    eval(u) = quantile(errorModel,u)

    return FittedModel(nothing, errorModel, eval, errors, u)

end


#MM for NormalInverseGaussian MM
function fit_NIG_mm(x)
    scipy = pyimport("scipy")

    fit = scipy.stats.norminvgauss.fit(x)

    mu = fit[3] 
    delta = fit[4] 
    alpha = fit[1]/delta
    beta = fit[2]/delta

    #create the error model
    errorModel = NormalInverseGaussian(mu,alpha,beta,delta)
    
    #calculate the errors and U
    errors = x .- mean(errorModel)

    u = cdf(errorModel,x)
    # u = scipy.stats.norminvgauss.cdf(x,fit...)
    # u = Vector{Float64}()

    fast_cdf = FastNIGCDF(errorModel)

    eval(u) = quantile(errorModel,u,x->fast_cdf(x))
    # eval(u) = quantile(errorModel,u)
    # eval(u) = scipy.stats.norminvgauss.ppf(u,fit...)

    return FittedModel(nothing, errorModel, eval, errors, u)

end

function fit_weibull(x)
    errorModel = fit(Weibull, x)
    errors = x .- mean(errorModel)
    u = cdf(errorModel,x)
    eval(u) = quantile(errorModel,u)
    return FittedModel(nothing, errorModel, eval, errors, u)
end

function fit_skewnormal(x::Vector{Float64})
    global __x
    __x = x
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    #approximate values based on moments
    sk = skewness(x)
    sgn = sign(sk)
    sk = abs(sk)
    δ = sqrt( (π/2) * (sk^(2/3)) / (sk^(2/3) + ((4-π)/2)^(2/3)) ) * sgn
    a = δ / sqrt(1-δ^2)
    s_hat = std(x)
    m_hat = mean(x)
    ω = s_hat / sqrt((1-(2*δ^2)/π))
    e = m_hat - ω*δ*sqrt(2/π)

    @variable(mle, m, start=e)
    @variable(mle, s>=1e-6, start=ω)
    @variable(mle, a, start=a)

    #Inner function to abstract away the X value
    function _gskl(m,s,a)
        d = SkewNormal(m,s,a)
        sum(log.(pdf.(d,__x)))
    end

    register(mle,:tLL,3,_gskl;autodiff=true)
    @NLobjective(
        mle,
        Max,
        tLL(m, s, a)
    )
    optimize!(mle)
    if !(string(termination_status(mle)) ∈ ["OPTIMAL", "LOCALLY_SOLVED","ALMOST_OPTIMAL","ALMOST_LOCALLY_SOLVED"])
        throw(ErrorException("Opimization Failed - $(termination_status(mle))"))
    end
    m = value(m)
    s = value(s)
    a = value(a)

    #create the error model
    errorModel = SkewNormal(m,s,a)
    #calculate the errors and U
    errors = x .- m
    u = cdf(errorModel,x)

    eval(u) = quantile(errorModel,u)

    return FittedModel(nothing, errorModel, eval, errors, u)

end