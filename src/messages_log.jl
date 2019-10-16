# Same methods as in `messages.jl` but using the
# samples log-likelihood instead of the likelihood.

# In-place forward pass, where α and c are allocated beforehand.
function forwardlog!(α::AbstractMatrix, c::AbstractVector, a::AbstractVector, A::AbstractMatrix, LL::AbstractMatrix)
    T, K = size(LL)

    fill!(α, 0.0)
    fill!(c, 0.0)

    m = vec_maximum(view(LL, 1,:))

    for j in Base.OneTo(K)
        α[1,j] = a[j] * exp(LL[1,j] - m)
        c[1] += α[1,j]
    end

    for j in Base.OneTo(K)
        α[1,j] /= c[1]
    end

    c[1] *= exp(m)

    for t in 2:T
        m = vec_maximum(view(LL, t, :))

        for j in Base.OneTo(K)
            for i in Base.OneTo(K)
                α[t,j] += α[t-1,i] * A[i,j] * exp(LL[t,j] - m)
            end
            c[t] += α[t,j]
        end

        for j in Base.OneTo(K)
            α[t,j] /= c[t]
        end

        c[t] *= exp(m)
    end
end

# In-place backward pass, where β and c are allocated beforehand.
function backwardlog!(β::AbstractMatrix, c::AbstractVector, a::AbstractVector, A::AbstractMatrix, LL::AbstractMatrix)
    T, K = size(LL)

    fill!(β, 0.0)
    fill!(c, 0.0)

    for j in Base.OneTo(K)
        β[end,j] = 1.0
    end

    for t in T-1:-1:1
        m = vec_maximum(view(LL, t+1, :))

        for j in Base.OneTo(K)
            for i in Base.OneTo(K)
                β[t,j] += β[t+1,i] * A[j,i] * exp(LL[t+1,i] - m)
            end
            c[t+1] += β[t,j]
        end

        for j in Base.OneTo(K)
            β[t,j] /= c[t+1]
        end

        c[t+1] *= exp(m)
    end

    m = vec_maximum(view(LL, 1,:))

    for j in Base.OneTo(K)
        c[1] += a[j] * exp(LL[1,j] - m) * β[1,j]
    end

    c[1] *= exp(m);
end

"""
    posteriorslog(init_distn, trans_matrix, log_likelihoods)

See [`posteriors`](@ref).
"""
function posteriorslog(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
    alphas, _ = forwardlog(init_distn, trans_matrix, log_likelihoods)
    betas, _ = backwardlog(init_distn, trans_matrix, log_likelihoods)
    gammas = alphas .* betas
    gammas ./ sum(gammas, dims=2)
end

# Convenience functions

for (f, s) in (:forwardlog => "α", :backwardlog => "β", :posteriorslog => "γ")
    @eval begin
        """
            $($f)(hmm, observations)
        """
        function $(f)(hmm::AbstractHMM, observations)
            $(f)(hmm.π0, hmm.π, loglikelihoods(hmm, observations))
        end
    end
end
