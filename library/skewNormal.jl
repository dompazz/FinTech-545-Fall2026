using Distributions
using Random
using StatsBase
using Statistics
using SpecialFunctions
using Roots
using QuadGK
using Interpolations

import Distributions.logpdf, Distributions.mean, Distributions.median, Random.rand, Distributions.params, Distributions.pdf
import Distributions.cdf, Distributions.quantile, Distributions.var, Distributions.skewness, Distributions.kurtosis

struct SkewNormal{T<:Real} <: ContinuousUnivariateDistribution
    ξ::T
    ω::T
    α::T
    SkewNormal{T}(ξ::T, ω::T, α::T) where {T<:Real} = new{T}(ξ, ω, α)
end

function SkewNormal(ξ::T, ω::T, α::T; check_args::Bool=true) where {T <: Real}
    Distributions.@check_args SkewNormal (ω, ω >= zero(ω))
    return SkewNormal{T}(ξ, ω, α)
end

#### TODO: Outer constructors handling passing Ints.

####

Distributions.@distr_support SkewNormal -Inf Inf

params(d::SkewNormal) = (d.ξ, d.ω, d.α)

const p2 = 2*π
const ipi2 = 2/π

function OwensT(h,a)
    f(x) = (exp(-0.5 * h^2 * (1+x^2)))/(1+x^2)
    integ = quadgk(f,0,a)[1]
    return integ/p2
end

pdf(d::SkewNormal, x::Real) = 2/d.ω * pdf(Normal(),(x-d.ξ)/d.ω) * cdf(Normal(),d.α * (x-d.ξ)/d.ω)
cdf(d::SkewNormal, x::Real) = cdf(Normal(),(x-d.ξ)/d.ω) - 2*OwensT((x-d.ξ)/d.ω,d.α)
quantile(d::SkewNormal, u::Real) = find_zero(x->cdf(d,x)-u,0.0)

function logpdf(d::SkewNormal, x::Real) 
    log(pdf(d,x))
end

rand(rng::AbstractRNG, d::SkewNormal{T}) where {T} = quantile.(d,randn(rng, float(T)))

_δ(d::SkewNormal) = d.α / (sqrt(1+d.α^2))
mean(d::SkewNormal) = d.ξ + d.ω*_δ(d)*sqrt(ipi2)
# median(d::SkewNormal) = d.ξ - d.λ * sinh(d.γ/d.δ)
var(d::SkewNormal) = d.ω^2 * (1 - (ipi2 * _δ(d)))

function skewness(d::SkewNormal)
    first = 2 - ipi2
    second = (_δ(d)*sqrt(ipi2))^3
    third = sqrt(1 - ipi2*_δ(d)^2)^3
    return first * second / third
end

function kurtosis(d::SkewNormal)
    first = p2 - 6
    second = (_δ(d)*sqrt(ipi2))^4
    third = (1 - ipi2*_δ(d)^2)^2
    return first * second / third
end

function cdf(d::NormalInverseGaussian,x::Real)
    # y = max(min(x,1-2*eps()),2*eps())
    function f(_x)
        out = pdf(d,_x)
        isnan(out) ? 0.0 : out
    end
    integ = quadgk(f,-10,x;rtol=1e-5)[1]
end

function quantile(d::NormalInverseGaussian,u::Real)
    st = quantile(Normal(mean(d),std(d)),u)
    # st = 0.0
    try
        return find_zero(x->cdf(d,x)-u,st)
    catch e
        try
            return find_zero(x->cdf(d,x)-u+1e-6,st)
        catch
            return NaN
        end
    end
end

function quantile(d::NormalInverseGaussian,u::Real,in_cdf::Function)   
    st = quantile(Normal(mean(d),std(d)),u)
    try
        return find_zero(x->in_cdf(x)-u,st)
    catch e
        try
            return find_zero(x->in_cdf(x)-u+1e-6,st)
        catch
            return NaN
        end
    end
end

"""
    FastNIGCDF(d::NormalInverseGaussian; n_points=1000)

Creates a fast interpolation-based CDF evaluator for the NormalInverseGaussian distribution.
Returns a callable object that evaluates the CDF at any point.

Parameters:
- d: NormalInverseGaussian distribution
- n_points: Number of points for interpolation grid
"""
struct FastNIGCDF
    interp::AbstractInterpolation
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
        new(interp, d, x_min, x_max)
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