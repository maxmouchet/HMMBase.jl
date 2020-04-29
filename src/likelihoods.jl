function likelihoods!(L::AbstractMatrix, hmm::AbstractHMM{Univariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(L) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        L[t, i] = pdf(hmm.B[i], observations[t])
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::AbstractHMM{Univariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        LL[t, i] = logpdf(hmm.B[i], observations[t])
    end
end

function likelihoods!(L::AbstractMatrix, hmm::AbstractHMM{Multivariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(L) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        L[t, i] = pdf(hmm.B[i], view(observations, t, :))
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::AbstractHMM{Multivariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, K)
    @inbounds for i in OneTo(K), t in OneTo(T)
        LL[t, i] = logpdf(hmm.B[i], view(observations, t, :))
    end
end
