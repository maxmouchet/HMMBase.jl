"""
    likelihoods(hmm, observations; logl) -> Matrix

Return the likelihood per-state and per-observation.

**Output**
- `Matrix{Float64}`: likelihoods matrix (`T x K`).

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000)
L = likelihoods(hmm, y)
```
"""
function likelihoods(hmm::AbstractHMM, observations; logl = false, robust = false)
    T, K = size(observations, 1), size(hmm, 1)
    L = Matrix{Float64}(undef, T, K)

    if logl
        loglikelihoods!(L, hmm, observations)
        robust && replace!(L, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    else
        likelihoods!(L, hmm, observations)
        robust && replace!(L, -Inf => nextfloat(-Inf), Inf => prevfloat(Inf))
    end

    L
end
