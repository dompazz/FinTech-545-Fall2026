function fit_regression_nig(y,x)
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
    
    scipy = pyimport("scipy")

    f = scipy.stats.norminvgauss.fit(e)
    # println("f: ", f)
    # println("b_start: ", b_start)
    
    @variable(mle, mu, start=f[3])
    @variable(mle, delta>=1e-6, start=f[4])
    @variable(mle, alpha>=1e-6, start=f[1]/f[4])
    @variable(mle, beta, start=f[2]/f[4])
    @variable(mle, B[i=1:nB],start=b_start[i])
    # @constraint(mle, mu==0)
    @NLconstraint(mle, alpha^2 - beta^2 >= 0.0 )

    #Inner function to abstract away the X value
    function _gtl(mu,delta,alpha, beta ,B...)
        # parms = [alpha*delta,beta*delta,mu,delta]
        b = collect(B)
        xm = __y - __x*b
        # sum(scipy.stats.norminvgauss.logpdf(xm,fit...))
        d = NormalInverseGaussian(mu,alpha,beta,delta)
        sum(logpdf(d, xm))
    end

    function g_gtl(g, mu,delta,alpha, beta ,B...)
        b = collect(B)
        s_val = _gtl(mu,delta,alpha, beta ,b...)
        n = size(b,1)+4
        x = [mu,delta,alpha, beta ,b...]
        e = eps()
        for i=1:n
            x_s = x[i]
            x[i] = x_s + e
            g[i] = (_gtl(x...) - s_val)/e
            x[i] = x_s
        end
    end
    
    register(mle,:tLL,nB+4,_gtl,g_gtl)
    @NLobjective(
        mle,
        Max,
        tLL(mu,delta,alpha, beta ,B...)
    )
    optimize!(mle)

    mu = value(mu)
    delta = value(delta)
    alpha = value(alpha)
    beta = value(beta)
    B = value.(B)

    errorModel = NormalInverseGaussian(mu,alpha,beta,delta)
    errors = __y - __x*B
    return FittedModel(B,errorModel,(x,u)-> 0.0, errors, [])
end

using XLSX
returns = XLSX.readtable("c:/temp/pi_data.xlsx", "pi_data") |> DataFrame

filter!(r-> r.vintage_year > 1999, returns)
filter!(r-> !ismissing(r.total_return) , returns)
filter!(r-> r.total_return != 0.0 , returns)

returns[!,:annual_return] .+= 0.0
returns[!,:abm_return] .+= 0.0

# returns[!,:annual_return] = log.(returns[!,:annual_return] .+ 1.0)
# returns[!,:abm_return] = log.(returns[!,:abm_return] .+ 1.0)

fm = fit_regression_nig(returns.annual_return, returns.abm_return)

B = OLS(returns.abm_return, returns.annual_return)
e = returns.annual_return - hcat(ones(length(returns.abm_return)),returns.abm_return)*B
fn = fit_normal(e)
fe = fit_regression_t(returns.annual_return, returns.abm_return)

x = sort(returns.abm_return)
y = hcat(ones(length(x)),x)*fe.beta
scatter(returns.abm_return,returns.annual_return,label="")
plot!(x,y,label="T Reg Line")
y = hcat(ones(length(x)),x)*fm.beta
plot!(x,y,label="NIG Reg Line")
y = hcat(ones(length(x)),x)*B
plot!(x,y,label="OLS Reg Line")

println("OLS aicc: $(aicc(fn.errorModel,e))")
println("NIG aicc: $(aicc(fm.errorModel,fm.errors))")
println("T aicc: $(aicc(fe.errorModel,fe.errors))")
println(" ")
println("T Alpha: $(fe.beta[1])")
println("T Beta: $(fe.beta[2])")


