"""
    likelihoods(hmm, observations) -> Matrix

Return the likelihood per-state and per-observation.

# Arguments
- `logl::Bool`: see common options (TODO).

# Output
- `Matrix{Float64}`: a TxK likelihoods matrix.
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
