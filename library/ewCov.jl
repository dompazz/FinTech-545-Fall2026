
#Function to calculate expoentially weighted covariance.  
function ewCovar(x,λ)
    m,n = size(x)

    #Calculate the weights
    w = expW(m,λ)

    #Remove the weighted mean from the series and add the weights to the covariance calculation
    xm = sqrt.(w) .* (x .- w' * x)

    #covariance = (sqrt(w) # x)' * (sqrt(w) # x)  where # is elementwise multiplication.
    return xm' * xm
end

function expW(m,λ)
    w = Vector{Float64}(undef,m)
    @inbounds for i in 1:m
        w[i] = (1-λ)*λ^(m-i)
    end
    #normalize weights to 1
    w = w ./ sum(w)
    return w
end