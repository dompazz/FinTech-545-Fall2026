using Distributions
using FFTW
using SpecialFunctions
using QuadGK

x = copy(toFit[!,"NOW"])
d = fit_NIG_mm(x).errorModel

using Interpolations
using QuadGK
using Distributions

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
    
    function FastNIGCDF(d::NormalInverseGaussian; n_points=1000)
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
        x_min = μ - 5 * std_dev
        x_max = μ + 5 * std_dev
        
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
            return quadgk(f, x_min - 10*std_dev, x, rtol=1e-6)[1]
        end
        
        # Calculate CDF at grid points
        cdf_values = map(accurate_cdf, x_grid)
        
        # Create interpolation object - fix this part to use the correct constructor
        # Use linear interpolation which is more robust and still fast
        interp = LinearInterpolation(x_grid, cdf_values, extrapolation_bc=Line())
        
        # Return the interpolator object
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

# Example usage
function create_fast_cdf(d::NormalInverseGaussian)
    # Precompute the interpolation
    fast_cdf = FastNIGCDF(d)
    
    # Return a simple function that calls the interpolator
    return x -> fast_cdf(x)
end

# For quantile function (inverse CDF)
function fast_quantile(d::NormalInverseGaussian, p::Real; 
                       tol=1e-10, max_iter=100)
    # Create fast CDF evaluator
    fast_cdf = FastNIGCDF(d)
    
    # Extract parameters for reasonable bounds
    μ = d.μ
    α = d.α
    β = d.β
    δ = d.δ
    γ = sqrt(α^2 - β^2)
    
    # Calculate variance for initial guess
    variance = δ * α^2 / γ^3
    std_dev = sqrt(variance)
    
    # Initial bounds
    left = μ - 10 * std_dev
    right = μ + 10 * std_dev
    
    # Binary search for the quantile
    for _ in 1:max_iter
        mid = (left + right) / 2
        mid_cdf = fast_cdf(mid)
        
        if abs(mid_cdf - p) < tol
            return mid
        elseif mid_cdf < p
            left = mid
        else
            right = mid
        end
    end
    
    # Return final approximation
    return (left + right) / 2
end

function quantile2(d::NormalInverseGaussian,u::Real,in_cdf::Function)   
    st = quantile(Normal(mean(d),std(d)),u)

    # st = 0.0
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

function quantile3(d::NormalInverseGaussian,u::Real)
    st = quantile(Normal(mean(d),std(d)),u)
    # fast_cdf = FastNIGCDF(d)
    # st = 0.0
    try
        return find_zero(x->cdf(d,x)-u,st;xatol=1e-6,atol=1e-6)
    catch e
        try
            return find_zero(x->cdf(d,x)-u+1e-6,st;xatol=1e-6,atol=1e-6)
        catch
            return NaN
        end
    end
end

using BenchmarkTools

u = rand(10)

cdf2(d,0.0)
@btime cdf2(d,0.0)
@btime cdf(d,0.0)
@btime quantile(d,$u[1])
@btime quantile2(d,$u[1],x->cdf(d,x))
@btime quantile3(d,$u[1])
@btime fast_quantile(d,$u[1])

fast_cdf = FastNIGCDF(d)
@btime quantile(d,$u[1],x->fast_cdf(x))

quantile2(d,u[1],x->fast_cdf(x)) - quantile(d,u[1])


u = rand(100000)
fm = fit_NIG_mm(toFit[!,"NOW"])
@time fm.eval.(u)