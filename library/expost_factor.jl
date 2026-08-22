
struct ExPostAttribution
    Attribution::DataFrame
    Weights::Array{Float64,2}
    FactorWeights::Array{Float64,2}
    ResidIndivual::Array{Float64,2}
    ResidReturn::Vector{Float64}
    PortfolioReturn::Vector{Float64}
    carinoK::Vector{Float64}
end

import Base.println
println(x::ExPostAttribution) = println(x.Attribution)

function expost_factor(w,upReturns,upFfData,Betas)
    _upFfData = upFfData[:,:]
    stocks = names(upReturns)
    factors = names(_upFfData)
    
    n = size(upReturns,1)
    m = size(stocks,1)
    
    pReturn = Vector{Float64}(undef,n)
    residReturn = Vector{Float64}(undef,n)
    weights = Array{Float64,2}(undef,n,length(w))
    factorWeights = Array{Float64,2}(undef,n,length(factors))
    lastW = copy(w)
    matReturns = Matrix(upReturns[!,stocks])
    ffReturns = Matrix(_upFfData[!,factors])
    
    residIndivual = matReturns - ffReturns * Betas'

    for i in 1:n
        # Save Current Weights in Matrix
        weights[i,:] = lastW
    
        #Factor Weight
        factorWeights[i,:] = sum.(eachcol(Betas .* lastW))
    
        # Update Weights by return
        lastW = lastW .* (1.0 .+ matReturns[i,:])
        
        # Portfolio return is the sum of the updated weights
        pR = sum(lastW)
        # Normalize the wieghts back so sum = 1
        lastW = lastW / pR
        # Store the return
        pReturn[i] = pR - 1
    
        #Residual
        residReturn[i] = (pR-1) - factorWeights[i,:]'*ffReturns[i,:]
    
    end
    
    
    # Set the portfolio return in the Update Return DataFrame
    _upFfData[!,:Alpha] = residReturn
    _upFfData[!,:Portfolio] = pReturn
    
    # Calculate the total return
    totalRet = exp(sum(log.(pReturn .+ 1)))-1
    # println("Total Return: $totalRet")
    # Calculate the Carino K
    k = log(totalRet + 1 ) / totalRet
    
    # Carino k_t is the ratio scaled by 1/K 
    carinoK = log.(1.0 .+ pReturn) ./ pReturn / k
    # Calculate the return attribution
    attrib = DataFrame(ffReturns .* factorWeights .* carinoK, factors)
    attrib[!,:Alpha] = residReturn .* carinoK
    residIndivual .*= weights

    # Set up a Dataframe for output.
    Attribution = DataFrame(:Value => ["TotalReturn", "Return Attribution"])
    
    newFactors = [factors..., :Alpha]
    # Loop over the factors
    for s in vcat(newFactors, :Portfolio)
        # Total Stock return over the period
        tr = exp(sum(log.(_upFfData[!,s] .+ 1)))-1
        # Attribution Return (total portfolio return if we are updating the portfolio column)
        atr =  s != :Portfolio ?  sum(attrib[:,s]) : tr
        # Set the values
        Attribution[!,s] = [ tr,  atr ]
    end
    
    
    # Realized Volatility Attribution
    
    # Y is our stock returns scaled by their weight at each time
    Y =  hcat(ffReturns .* factorWeights, residReturn)
    # Set up X with the Portfolio Return
    X = hcat(fill(1.0, size(pReturn,1)),pReturn)
    # Calculate the Beta and discard the intercept
    B = (inv(X'*X)*X'*Y)[2,:]
    # Component SD is Beta times the standard Deviation of the portfolio
    cSD = B * std(pReturn)
    
    #Check that the sum of component SD is equal to the portfolio SD
    if !(sum(cSD) ≈ std(pReturn))
        println("Component SD does not equal Portfolio SD")
    else
        # println("Component SD equals Portfolio SD")
    end

    # Add the Vol attribution to the output 
    Attribution = vcat(Attribution,    
        DataFrame(:Value=>"Vol Attribution", [Symbol(newFactors[i])=>cSD[i] for i in 1:size(newFactors,1)]... , :Portfolio=>std(pReturn))
    )

    return ExPostAttribution(
        Attribution, 
        weights, 
        factorWeights, 
        residIndivual, 
        residReturn,
        pReturn,
        carinoK
    )
end
