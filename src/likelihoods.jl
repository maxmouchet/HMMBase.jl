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
y = rand(hmm, 1000)
LL = likelihoods(hmm, y)
```
"""
function loglikelihoods(hmm::AbstractHMM, observations; logl = nothing, robust = false)
    !isnothing(logl) && deprecate_kwargs("logl")
    T, K = size(observations, 1), size(hmm, 1)
    LL = Matrix{Float64}(undef, T, K)

    loglikelihoods!(LL, hmm, observations)
    if robust
        replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    end
    LL
end
