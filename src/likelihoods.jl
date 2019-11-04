function likelihoods!(L::AbstractMatrix, hmm::AbstractHMM{Univariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(L) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        L[t,i] = pdf(hmm.B[i], observations[t])
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::AbstractHMM{Univariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        LL[t,i] = logpdf(hmm.B[i], observations[t])
    end
end

function likelihoods!(L::AbstractMatrix, hmm::AbstractHMM{Multivariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(L) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        L[t,i] = pdf(hmm.B[i], view(observations, t, :))
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::AbstractHMM{Multivariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        LL[t,i] = logpdf(hmm.B[i], view(observations, t, :))
    end
end

"""
    likelihoods(hmm, observations) -> Matrix

Return the likelihood per-state and per-observation.

# Output
- `Matrix{Float64}`: a TxK likelihoods matrix.
"""
function likelihoods(hmm::AbstractHMM, observations)
    T, K = size(observations, 1), size(hmm, 1)
    L = Matrix{Float64}(undef, T, K)
    likelihoods!(L, hmm, observations)
    L
end

"""
    loglikelihoods(hmm, observations) -> Matrix

Return the log-likelihood per-state and per-observation.

# Output
- `Matrix{Float64}`: a TxK log-likelihoods matrix.
"""
function loglikelihoods(hmm::AbstractHMM, observations)
    T, K = size(observations, 1), size(hmm, 1)
    LL = Matrix{Float64}(undef, T, K)
    loglikelihoods!(LL, hmm, observations)
    LL
end
