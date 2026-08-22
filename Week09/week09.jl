using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Statistics
using StatsBase
using Dates
using Plots
using Random
using Printf

ENV["GKSwstype"] = "100"          # headless plotting

# =====================================================================
# Week 09 -- Factor Models: Construction and Use
#
# Data is reused from Week08.  Adjust the path if you move this file.
# =====================================================================
const DATA = "../Week08/"

# ---------------------------------------------------------------------
# 1. THE PROBLEM: the sample covariance matrix is singular
# ---------------------------------------------------------------------
returns = CSV.read(DATA * "DailyReturn.csv", DataFrame)
returns[!, :Date] = Date.(returns.Date, dateformat"mm/dd/yyyy")
sort!(returns, :Date)

# 100 stocks -- drop SPY, it is an index not a stock
stocks = Symbol.(setdiff(names(returns), ["Date", "SPY"]))
R = Matrix(returns[!, stocks])
T, n = size(R)

println("=== 1. The parameter counting problem ===")
@printf("T = %d observations, n = %d assets\n", T, n)
@printf("Parameters in a full covariance matrix: %d\n", n * (n + 1) ÷ 2)
@printf("Data points available:                  %d\n", T * n)

S = cov(R)
evS = eigvals(Symmetric(S))
@printf("rank(S)            = %d   (= min(T-1, n))\n", rank(S))
@printf("smallest eigenvalue = %.3e\n", minimum(evS))
@printf("condition number    = %.3e\n", maximum(evS) / max(minimum(evS), eps()))

# The PD Cholesky fails -- but the PSD version from Week 03 does NOT.
println("isposdef(S) = ", isposdef(Symmetric(S)), "   <- the PD algorithm fails")

"""
    chol_psd(A)

Week 03's PSD-capable Cholesky: when a pivot is non-positive, leave that
column of the root as zeros and carry on.  Returns a valid L with
A = L*L', where the number of non-zero columns equals rank(A).
"""
function chol_psd(A; tol = 1e-12)
    N = size(A, 1)
    L = zeros(N, N)
    for j in 1:N
        d = A[j, j] - L[j, 1:j-1]' * L[j, 1:j-1]
        if d <= tol
            L[j:N, j] .= 0.0          # zero column -- this direction carries no variance
            continue
        end
        L[j, j] = sqrt(d)
        for i in (j+1):N
            L[i, j] = (A[i, j] - L[i, 1:j-1]' * L[j, 1:j-1]) / L[j, j]
        end
    end
    return L
end

L = chol_psd(S)
nzcols = count(j -> any(!iszero, L[:, j]), 1:n)
@printf("PSD Cholesky SUCCEEDS: %d non-zero columns, %d zero columns\n", nzcols, n - nzcols)
@printf("reconstruction error max|LL' - S| = %.3e\n", maximum(abs.(L * L' - S)))
println("So we can factor it and simulate from it.  That is exactly the problem:")
@printf("every simulated scenario lies in the %d-dim span of the %d historical days.\n",
        rank(L), T)

# What the missing directions cost, measured out of sample.
# Estimate on the first 30 days; the null space is then 71-dimensional.
h1, h2 = R[1:30, :], R[31:end, :]
S1 = cov(h1)
e1 = eigen(Symmetric(S1))
nullsp = e1.vectors[:, e1.values .< 1e-14]

# Any portfolio in the null space with weights summing to 1
a = reshape(ones(n)' * nullsp, 1, :)
wn0 = nullsp * (a \ [1.0])

@printf("\nA null-space portfolio (max |w| = %.1f%%, gross = %.1f):\n",
        maximum(abs.(wn0)) * 100, sum(abs.(wn0)))
@printf("  forecast vol (in-sample)      = %.2e %%\n", sqrt(max(wn0' * S1 * wn0, 0) * 255) * 100)
@printf("  realized vol, same 30 days    = %.2e %%\n", std(h1 * wn0) * sqrt(255) * 100)
@printf("  realized vol, NEXT 30 days    = %.2f%%   <- the estimate called this riskless\n",
        std(h2 * wn0) * sqrt(255) * 100)

# ---------------------------------------------------------------------
# 2. TIME-SERIES MODEL: Fama-French 3
# ---------------------------------------------------------------------
ff3 = CSV.read(DATA * "F-F_Research_Data_Factors_daily.CSV", DataFrame)
rename!(ff3, Symbol("Mkt-RF") => :Mkt_RF)
ff3[!, Not(:Date)] = Matrix(ff3[!, Not(:Date)]) ./ 100.0
ff3[!, :Date] = Date.(string.(ff3.Date), dateformat"yyyymmdd")

d = innerjoin(returns, ff3, on = :Date)
sort!(d, :Date)

fnames = [:Mkt_RF, :SMB, :HML]
m = length(fnames)

# Excess returns on the left-hand side
Y = Matrix(d[!, stocks]) .- d.RF
Fr = Matrix(d[!, fnames])
T = size(Y, 1)

# Stacked OLS: one (m+1)x(m+1) inverse serves all n assets
X = hcat(ones(T), Fr)
Θ = (X' * X) \ (X' * Y)          # (m+1) x n
α = Θ[1, :]
B = Θ[2:end, :]'                  # n x m
E = Y .- X * Θ                    # T x n residuals

Fcov = cov(Fr)                                        # m x m
Dv   = vec(sum(E .^ 2, dims = 1)) ./ (T - m - 1)      # note the T-m-1

Σ_ff = B * Fcov * B' + Diagonal(Dv)

println("\n=== 2. FF3 time-series model ===")
sysvar = [B[i, :]' * Fcov * B[i, :] for i in 1:n]
R2 = sysvar ./ (sysvar .+ Dv)
@printf("mean R2 = %.3f   min = %.3f (%s)   max = %.3f (%s)\n",
        mean(R2), minimum(R2), stocks[argmin(R2)], maximum(R2), stocks[argmax(R2)])

betas = DataFrame(:Stock => String.(stocks), :Mkt => B[:, 1], :SMB => B[:, 2],
                  :HML => B[:, 3], :R2 => R2, :AnnSD => sqrt.((sysvar .+ Dv) .* 255))
show(sort(betas, :Mkt, rev = true)[1:10, :], allrows = true)

ev_ff = eigvals(Symmetric(Σ_ff))
@printf("\n\nrank = %d   min eig = %.3e   cond = %.1f   isposdef = %s\n",
        rank(Σ_ff), minimum(ev_ff), maximum(ev_ff) / minimum(ev_ff), isposdef(Symmetric(Σ_ff)))
@printf("parameters: %d  (vs %d for the sample covariance)\n",
        n * m + m * (m + 1) ÷ 2 + n, n * (n + 1) ÷ 2)

# Revisit the "riskless" portfolio from section 1.  A factor model can never
# call a portfolio riskless, because D is strictly positive.
@printf("factor model's forecast for the null-space portfolio: %.2f%%  (realized 10.50%%)\n",
        sqrt(wn0' * Σ_ff * wn0 * 255) * 100)

# ---------------------------------------------------------------------
# 3. STATISTICAL MODEL: PCA
# ---------------------------------------------------------------------
println("\n=== 3. Statistical (PCA) model ===")
ef = eigen(Symmetric(S))
ord = sortperm(ef.values, rev = true)
λ = ef.values[ord]
V = ef.vectors[:, ord]

tot = sum(λ)
cumvar = cumsum(λ) ./ tot
for k in [1, 2, 3, 5, 10, 20]
    @printf("  k = %2d   cumulative variance explained = %5.1f%%\n", k, cumvar[k] * 100)
end

k = 3
Bp = V[:, 1:k] * Diagonal(sqrt.(λ[1:k]))     # loadings; factors have unit variance
Dp = diag(S) .- vec(sum(Bp .^ 2, dims = 2))
@printf("minimum specific variance = %.3e  (negative => Heywood case, floor it)\n", minimum(Dp))
Dp = max.(Dp, 1e-8)
Σ_pca = Bp * Bp' + Diagonal(Dp)
ev_p = eigvals(Symmetric(Σ_pca))
@printf("rank = %d   min eig = %.3e   cond = %.1f\n",
        rank(Σ_pca), minimum(ev_p), maximum(ev_p) / minimum(ev_p))

# Is PC1 "the market"?  Check, do not assume -- loadings are only identified up to rotation.
@printf("corr(PC1 loading, FF3 market beta) = %+.3f\n", cor(Bp[:, 1], B[:, 1]))
@printf("share of PC1 loadings with a common sign = %.2f\n",
        mean(sign.(Bp[:, 1]) .== sign(Bp[1, 1])))

scree = plot(1:30, λ[1:30] ./ tot .* 100, seriestype = :bar, legend = false,
             xlabel = "Component", ylabel = "% of variance", title = "Scree")
cumpl = plot(1:30, cumvar[1:30] .* 100, marker = :circle, legend = false,
             xlabel = "Number of components", ylabel = "Cumulative %",
             title = "Cumulative variance", ylims = (0, 100))
savefig(plot(scree, cumpl, layout = (1, 2), size = (900, 340)), "figures/scree_julia.png")

# ---------------------------------------------------------------------
# 4. CROSS-SECTIONAL MODEL: market + industries, sum-to-zero constrained
# ---------------------------------------------------------------------
println("\n=== 4. Cross-sectional (fundamental) model ===")

sectors = Dict(
 :AAPL=>"Tech", :MSFT=>"Tech", :AMZN=>"Discr", :TSLA=>"Discr", :GOOGL=>"Comm",
 :GOOG=>"Comm", :FB=>"Comm", :NVDA=>"Tech", Symbol("BRK-B")=>"Fin", :JPM=>"Fin",
 :JNJ=>"Health", :UNH=>"Health", :HD=>"Discr", :PG=>"Staples", :V=>"Fin",
 :BAC=>"Fin", :MA=>"Fin", :PFE=>"Health", :XOM=>"Other", :DIS=>"Comm",
 :CSCO=>"Tech", :AVGO=>"Tech", :ADBE=>"Tech", :CVX=>"Other", :PEP=>"Staples",
 :TMO=>"Health", :KO=>"Staples", :ABBV=>"Health", :CMCSA=>"Comm", :NFLX=>"Comm",
 :ABT=>"Health", :ACN=>"Tech", :COST=>"Staples", :CRM=>"Tech", :INTC=>"Tech",
 :WFC=>"Fin", :VZ=>"Comm", :PYPL=>"Fin", :WMT=>"Staples", :QCOM=>"Tech",
 :MRK=>"Health", :LLY=>"Health", :MCD=>"Discr", :T=>"Comm", :NKE=>"Discr",
 :DHR=>"Health", :LOW=>"Discr", :LIN=>"Other", :TXN=>"Tech", :NEE=>"Other",
 :AMD=>"Tech", :UNP=>"Indust", :PM=>"Staples", :INTU=>"Tech", :UPS=>"Indust",
 :HON=>"Indust", :MS=>"Fin", :MDT=>"Health", :BMY=>"Health", :AMAT=>"Tech",
 :ORCL=>"Tech", :SCHW=>"Fin", :CVS=>"Health", :RTX=>"Indust", :C=>"Fin",
 :GS=>"Fin", :AMGN=>"Health", :BLK=>"Fin", :BA=>"Indust", :CAT=>"Indust",
 :IBM=>"Tech", :SBUX=>"Discr", :AMT=>"Other", :PLD=>"Other", :GE=>"Indust",
 :ISRG=>"Health", :COP=>"Other", :TGT=>"Discr", :ANTM=>"Health", :AXP=>"Fin",
 :DE=>"Indust", :MU=>"Tech", :SPGI=>"Fin", :MMM=>"Indust", :NOW=>"Tech",
 :BKNG=>"Discr", :F=>"Discr", :ADP=>"Tech", :ZTS=>"Health", :LRCX=>"Tech",
 :PNC=>"Fin", :MDLZ=>"Staples", :MO=>"Staples", :ADI=>"Tech", :GILD=>"Health",
 :LMT=>"Indust", :SYK=>"Health", :GM=>"Discr", :TFC=>"Fin", :TJX=>"Discr")

industries = sort(unique(getindex.(Ref(sectors), stocks)))
Bx = hcat(ones(n), [sectors[s] == g ? 1.0 : 0.0 for s in stocks, g in industries])
xnames = vcat(["Market"], industries)
mx = size(Bx, 2)

# Industry dummies sum to the market column -> Bx is rank deficient.
# Constrain the industry factor returns to sum to zero.
C = reshape(vcat(0.0, ones(length(industries))), 1, mx)
W = Diagonal(1.0 ./ max.(Dv, 1e-8))        # GLS: weight by inverse specific variance
Rx = Matrix(d[!, stocks])

KKT = [Bx' * W * Bx  C'; C  zeros(1, 1)]
KKTi = inv(KKT)
Fhat = zeros(T, mx)
for t in 1:T
    Fhat[t, :] = (KKTi * vcat(Bx' * W * Rx[t, :], 0.0))[1:mx]
end
Ex = Rx .- Fhat * Bx'

# Fama-MacBeth: inference from the time series of the cross-sectional estimates
println("Factor                mean(bp)   ann vol    FM t-stat")
for j in 1:mx
    f = Fhat[:, j]
    tstat = mean(f) / (std(f) / sqrt(T))
    @printf("  %-16s %+8.1f  %8.1f%%  %+9.2f\n",
            xnames[j], mean(f) * 1e4, std(f) * sqrt(255) * 100, tstat)
end
println("Note: nothing clears |t| > 2.  60 days says nothing about factor premia.")

Fcx = cov(Fhat)
Dx = vec(sum(Ex .^ 2, dims = 1)) ./ (T - mx)
Σ_x = Bx * Fcx * Bx' + Diagonal(Dx)

# ---------------------------------------------------------------------
# 5. WOODBURY: (D + B F B')^-1 without an n x n inverse
# ---------------------------------------------------------------------
println("\n=== 5. Woodbury identity ===")
function woodbury_solve(B, F, dv, x)
    Di = 1.0 ./ dv
    inner = inv(inv(F) + (B' .* Di') * B)
    return Di .* x .- Di .* (B * (inner * (B' * (Di .* x))))
end

b = ones(n)
direct = Σ_ff \ b
wood   = woodbury_solve(B, Fcov, Dv, b)
@printf("max relative difference: %.3e   (same answer)\n",
        maximum(abs.(direct .- wood)) / maximum(abs.(direct)))
println("Direct solve is O(n^3); Woodbury is O(n m^2).  With m = 3 that is effectively O(n).")

# ---------------------------------------------------------------------
# 6. EXPOSURES, VARIANCE DECOMPOSITION, TRACKING ERROR
# ---------------------------------------------------------------------
println("\n=== 6. Ex-ante risk ===")
w = fill(1.0 / n, n)
x = B' * w                                  # portfolio factor exposures

sys  = x' * Fcov * x
idio = sum(w .^ 2 .* Dv)
tot_v = sys + idio

@printf("exposures: Mkt %+.3f  SMB %+.3f  HML %+.3f\n", x...)
@printf("ann vol total %.2f%%  = systematic %.2f%%  (+) idio %.2f%%\n",
        sqrt(tot_v * 255) * 100, sqrt(sys * 255) * 100, sqrt(idio * 255) * 100)
@printf("idiosyncratic share of variance: %.2f%%   (single stock average: %.1f%%)\n",
        idio / tot_v * 100, mean(1 .- R2) * 100)

mc = Fcov * x
for (j, nm) in enumerate(fnames)
    @printf("  %-8s share of variance %+7.2f%%\n", nm, x[j] * mc[j] / tot_v * 100)
end

sd = sqrt(tot_v)
z = quantile(Normal(), 0.05)
@printf("1-day VaR(5%%) = %.3f%%   ES(5%%) = %.3f%%\n",
        -z * sd * 100, pdf(Normal(), z) / 0.05 * sd * 100)

# Tracking error against a benchmark of the 20 largest names
bench = zeros(n); bench[1:20] .= 1 / 20
a = w .- bench
ax = B' * a
te_f = sqrt(ax' * Fcov * ax * 255)
te_s = sqrt(sum(a .^ 2 .* Dv) * 255)
@printf("TE %.2f%% = factor bets %.2f%% (+) stock selection %.2f%%\n",
        sqrt(te_f^2 + te_s^2) * 100, te_f * 100, te_s * 100)
@printf("active exposures: Mkt %+.3f  SMB %+.3f  HML %+.3f\n", ax...)

# ---------------------------------------------------------------------
# 7. PORTFOLIO CONSTRUCTION
# ---------------------------------------------------------------------
println("\n=== 7. Portfolio construction ===")
minvar(Σ) = (u = Σ \ ones(size(Σ, 1)); u ./ sum(u))

for (nm, Σ) in [("FF3", Σ_ff), ("PCA-3", Σ_pca), ("Cross-sec", Σ_x)]
    wv = minvar(Σ)
    @printf("%-10s predicted ann vol %.2f%%  realized %.2f%%  max w %.1f%%  gross %.2f\n",
            nm, sqrt(wv' * Σ * wv * 255) * 100, std(R * wv) * sqrt(255) * 100,
            maximum(wv) * 100, sum(abs.(wv)))
end
println("Predicted < realized: the diagonal-D assumption is understating risk.  See section 9.")

# Factor-neutral: 1'w = 1 and B'w = 0 are both linear constraints
A = hcat(ones(n), B)
c = vcat(1.0, zeros(m))
Σi = inv(Σ_ff)
wn = Σi * A * ((A' * Σi * A) \ c)
@printf("\nfactor-neutral: sum(w) = %.6f   B'w = %s\n", sum(wn), round.(B' * wn, digits = 12))
@printf("ann vol %.2f%% (unconstrained MVP %.2f%%)  gross %.2f\n",
        sqrt(wn' * Σ_ff * wn * 255) * 100,
        sqrt(minvar(Σ_ff)' * Σ_ff * minvar(Σ_ff) * 255) * 100, sum(abs.(wn)))

# ---------------------------------------------------------------------
# 8. DIAGNOSTIC: is D really diagonal?
# ---------------------------------------------------------------------
println("\n=== 8. Residual correlation diagnostic ===")
Ec = cor(E)
pairs = [(abs(Ec[i, j]), i, j) for i in 1:n for j in (i+1):n]
sort!(pairs, rev = true)

# Under the null of zero residual correlation the sampling sd is ~1/sqrt(T-m-1).
noise = 1 / sqrt(T - m - 1)
@printf("mean |residual corr| = %.3f   noise floor under the null ~ %.3f\n",
        mean(first.(pairs)), 0.8 * noise)
println("The average is uninformative.  The tail is where the signal is:")
for (v, i, j) in pairs[1:10]
    @printf("  %-7s %-7s  %+.3f\n", stocks[i], stocks[j], Ec[i, j])
end
println("\nDual share classes, and industry pairs the FF3 model has no factor for.")
