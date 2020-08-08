function loglikelihoods!(LL::AbstractArray, hmm::AbstractHMM{Univariate}, observations)
    T, K, N = size(observations, 1), size(hmm, 1), size(observations, 2)
    @argcheck size(LL) == (T, K, N)
    @inbounds for n in OneTo(N)
        T = length(filter(!isnothing, observations[:, n]))
        for i in OneTo(K), t in OneTo(T)
            LL[t, i, n] = logpdf(hmm.B[i], observations[t, n])
        end
    end
end

function loglikelihoods!(LL::AbstractArray, hmm::AbstractHMM{Multivariate}, observations)
    T, K, N = size(observations, 1), size(hmm, 1), size(observations, 3)
    @argcheck size(LL) == (T, K, N)
    @inbounds for n in OneTo(N)
        T = size(remove_nothing(observations[:, :, n]), 1)
        for i in OneTo(K), t in OneTo(T)
            LL[t, i, n] = logpdf(hmm.B[i], view(observations, t, :, n))
        end
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::AbstractHMM{Univariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        LL[t, i] = logpdf(hmm.B[i], observations[t])
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::AbstractHMM{Multivariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        LL[t, i] = logpdf(hmm.B[i], view(observations, t, :))
    end
end

"""
    loglikelihoods(hmm, observations; robust) -> Matrix

Return the log-likelihood per-state and per-observation.

**Output**
- `Matrix{Float64}`: log-likelihoods matrix (`T x K`).

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000) # or
y = rand(hmm, 1000, 2)
LL = loglikelihoods(hmm, y)
```
"""
function loglikelihoods(hmm::AbstractHMM, observations::AbstractArray; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    T, K, N = size(observations, 1), size(hmm, 1), last(size(observations))
    LL = Array{Union{Float64,Nothing}}(nothing, T, K, N)

    loglikelihoods!(LL, hmm, observations)
    if robust
        replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    end
    LL
end

function loglikelihoods(hmm::AbstractHMM, observations::AbstractVector; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    T, K = size(observations, 1), size(hmm, 1)
    LL = Matrix{Float64}(undef, T, K)

    loglikelihoods!(LL, hmm, observations)
    if robust
        replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    end
    LL
end