function loglikelihoods!(LL::AbstractArray, hmm::AbstractHMM{Univariate}, observations)
    T, K, N = size(observations, 1), size(hmm, 1), size(observations, 2)
    @argcheck size(LL) == (T, K, N)
    @inbounds for i in OneTo(K), t in OneTo(T), n in OneTo(N)
        LL[t, i, n] = logpdf(hmm.B[i], observations[t, n])
    end
end

# haven't changed this yet.
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
y = rand(hmm, 1000)
LL = likelihoods(hmm, y)
```
"""
function loglikelihoods(hmm::AbstractHMM, observations; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    T, K, N = size(observations, 1), size(hmm, 1), size(observations, 2)
    LL = Array{Float64}(undef, T, K, N)

    loglikelihoods!(LL, hmm, observations)
    if robust
        replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    end
    LL
end