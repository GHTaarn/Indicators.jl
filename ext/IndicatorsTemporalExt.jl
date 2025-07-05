module IndicatorsTemporalExt

using Indicators, Temporal
import Temporal.acf  # used for running autocorrelation function
import Indicators:
        runmean, runsum, runvar, runsd, runcov, runcor, runmax, runmin, runmad, runquantile, runacf,
        wilder_sum, mode, diffn,
        sma, trima, wma, ema, mma, kama, mama, hma, swma, dema, tema, alma, zlema, vwma, vwap, hama,
        mlr_beta, mlr_slope, mlr_intercept, mlr, mlr_se, mlr_ub, mlr_lb, mlr_bands, mlr_rsq,
        aroon, donch, momentum, roc, macd, rsi, adx, psar, kst, wpr, cci, stoch, smi,
        bbands, tr, atr, keltner,
        crossover, crossunder,
        renko,
        maxima, minima, support, resistance,
        rsrange, hurst

include("../src/temporal.jl")

"""
```
function runacf(x::Vector{T};
                n::Int = 10,
                maxlag::Int = n-3,
                lags::AbstractVector{Int,1} = 0:maxlag,
                cumulative::Bool = true)::Matrix{T} where {T<:Real}
                runacf(X::Matrix; n::Int=10, cumulative::Bool=true, maxlag::Int=n-3, lags::AbstractVector{Int}=0:maxlag)::Matrix{Float64}
```

Compute the running/rolling autocorrelation of a vector.
"""
function runacf(x::AbstractVector{T};
                n::Int = 10,
                maxlag::Int = n-3,
                lags::AbstractVector{Int} = 0:maxlag,
                cumulative::Bool = true)::Matrix{T} where {T<:Real}
    @assert size(x, 2) == 1 "Autocorrelation input array must be one-dimensional"
    N = size(x, 1)
    @assert n < N && n > 0
    if length(lags) == 1 && lags[1] == 0
        return ones((N, 1))
    end
    out = zeros((N, length(lags))) * NaN
    if cumulative
        @inbounds for i in n:N
            out[i,:] = acf(x[1:i], lags=lags)
        end
    else
        @inbounds for i in n:N
            out[i,:] = acf(x[i-n+1:i], lags=lags)
        end
    end
    return out
end
runacf(X::AbstractMatrix; n::Int=10, cumulative::Bool=true, maxlag::Int=n-3, lags::AbstractVector{Int}=0:maxlag)::Matrix{Float64} = hcat((runacf(X[:,j], n=n, cumulative=cumulative, maxlag=maxlag, lags=lags) for j in 1:size(X,2))...)

end

