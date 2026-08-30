
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


# FastNIGCDF and the two NIG fitters below need Interpolations and QuadGK in
# scope, and the NIG cdf/quantile methods from skewNormal.jl. Like everything
# else in this file, the caller supplies them -- include skewNormal.jl first.
# The interp field is parameterized rather than typed as AbstractInterpolation
# so that including this file does not itself require Interpolations.

"""
    FastNIGCDF(d::NormalInverseGaussian; n_points=1000)

Creates a fast interpolation-based CDF evaluator for the NormalInverseGaussian distribution.
Returns a callable object that evaluates the CDF at any point.

Parameters:
- d: NormalInverseGaussian distribution
- n_points: Number of points for interpolation grid
"""
struct FastNIGCDF{I}
    interp::I
    d::NormalInverseGaussian
    x_min::Float64
    x_max::Float64

    function FastNIGCDF(d::NormalInverseGaussian; n_points=1600)
        # Extract parameters
        μ = d.μ
        α = d.α
        β = d.β
        δ = d.δ
        γ = sqrt(α^2 - β^2)
        
        # Determine reasonable range for the distribution
        # Use theoretical properties for tighter bound calculation
        variance = δ * α^2 / γ^3
        std_dev = sqrt(variance)
        
        # Range covers μ ± 5 standard deviations
        x_min = μ - 10 * std_dev
        x_max = μ + 10 * std_dev
        
        # Create a non-uniform grid with more points in the center
        # This gives better accuracy where the density changes rapidly
        center_points = n_points ÷ 2
        tail_points = n_points ÷ 4
        
        # Create three segments with different point densities
        left_segment = collect(range(x_min, μ - 0.5 * std_dev, length=tail_points))
        center_segment = collect(range(μ - 0.5 * std_dev, μ + 0.5 * std_dev, length=center_points))
        right_segment = collect(range(μ + 0.5 * std_dev, x_max, length=tail_points))
        
        # Combine segments
        x_grid = vcat(left_segment, center_segment[2:end], right_segment[2:end])
        
        # Compute CDF values accurately using quadrature only once
        function accurate_cdf(x)
            function f(_x)
                out = pdf(d, _x)
                isnan(out) ? 0.0 : out
            end
            return quadgk(f, x_min - 20*std_dev, x, rtol=1e-6)[1]
        end
        
        # Calculate CDF at grid points
        cdf_values = map(accurate_cdf, x_grid)
        
        # Use linear interpolation which is more robust and still fast
        interp = LinearInterpolation(x_grid, cdf_values, extrapolation_bc=Line())
        
        # Return the FastNIGCDF object
        new{typeof(interp)}(interp, d, x_min, x_max)
    end
end

# Make the FastNIGCDF struct callable
function (f::FastNIGCDF)(x::Real)
    d = f.d
    
    # Handle values outside the interpolation range
    if x <= f.x_min
        return 0.0
    elseif x >= f.x_max
        return 1.0
    else
        # Use interpolation for values within range
        return f.interp(x)
    end
end

# Method of moments for the NIG, in closed form.
#
# The NIG(mu, alpha, beta, delta) moments, with gamma = sqrt(alpha^2 - beta^2):
#
#   mean  = mu + delta*beta/gamma
#   var   = delta*alpha^2/gamma^3
#   skew  = 3*beta/(alpha*sqrt(delta*gamma))
#   exkur = 3*(1 + 4*beta^2/alpha^2)/(delta*gamma)
#
# Write rho = beta/alpha. Then skew^2/exkur = 3*rho^2/(1 + 4*rho^2), which
# inverts for rho without touching the other two parameters, and the rest
# follows one at a time. The NIG can only reach shapes with
# exkur > (5/3)*skew^2; outside that the sample is telling you to fit
# something else.
function fit_nig_moments(x)
    m = mean(x)
    v = var(x)
    s = skewness(x)
    k = kurtosis(x)   # excess

    k > 0 || error("NIG method of moments needs positive excess kurtosis, got $k")
    t = s^2 / k
    t < 3/5 || error("Sample is outside the NIG region: excess kurtosis must exceed (5/3)*skew^2")

    rho2 = t / (3 - 4t)
    rho = sign(s) * sqrt(rho2)

    D = 3 * (1 + 4 * rho2) / k          # delta*gamma
    alpha = sqrt(D / (v * (1 - rho2)^2))
    beta = rho * alpha
    gamma = alpha * sqrt(1 - rho2)
    delta = D / gamma
    mu = m - delta * beta / gamma

    return _nig_fitted_model(NormalInverseGaussian(mu, alpha, beta, delta), x)
end

#MLE for NormalInverseGaussian, via scipy
function fit_NIG_mle(x)
    scipy = pyimport("scipy")

    fit = scipy.stats.norminvgauss.fit(x)

    mu = fit[3]
    delta = fit[4]
    alpha = fit[1]/delta
    beta = fit[2]/delta

    return _nig_fitted_model(NormalInverseGaussian(mu,alpha,beta,delta), x)
end

# Shared tail of both NIG fits. The direct quantile on a NIG is slow enough that
# the interpolated CDF is worth building once and closing over.
function _nig_fitted_model(errorModel::NormalInverseGaussian, x)
    #calculate the errors and U
    errors = x .- mean(errorModel)

    u = cdf(errorModel,x)

    fast_cdf = FastNIGCDF(errorModel)
    _cdf(z) = fast_cdf(z)

    # The 3 argument quantile in skewNormal.jl takes a scalar u. Broadcast so
    # eval accepts a vector, which is how every other FittedModel's eval is
    # called when simulating.
    eval(u) = quantile.(Ref(errorModel),u,Ref(_cdf))

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