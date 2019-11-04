# In-place update of the initial state distribution.
function update_a!(a::AbstractVector, α::AbstractMatrix, β::AbstractMatrix)
    @argcheck size(α, 1) == size(β, 1)
    @argcheck size(α, 2) == size(β, 2) == size(a, 1)

    K = length(a)
    c = 0.0
    
    for i in OneTo(K)
        a[i] = α[1,i] * β[1,i]
        c += a[i]
    end

    for i in OneTo(K)
        a[i] /= c
    end
end

# In-place update of the transition matrix.
function update_A!(A::AbstractMatrix, ξ::AbstractArray, α::AbstractMatrix, β::AbstractMatrix, LL::AbstractMatrix)
    @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1)
    @argcheck size(α, 2) == size(β, 2) == size(LL, 2) == size(A, 1) == size(A, 2) == size(ξ, 2) == size(ξ, 3)

    T, K = size(LL)

    @inbounds for t in OneTo(T - 1)
        m = vec_maximum(view(LL, t, :))
        c = 0.0

        for i in OneTo(K), j in OneTo(K)
            ξ[t,i,j] = α[t,i] * A[i,j] * exp(LL[t + 1,j] - m) * β[t + 1,j]
            c += ξ[t,i,j]
        end

        for i in OneTo(K), j in OneTo(K)
            ξ[t,i,j] /= c
        end
    end

    fill!(A, 0.0)

    @inbounds for i in OneTo(K)
        c = 0.0

        for j in OneTo(K)
            for t in OneTo(T - 1)
                A[i,j] += ξ[t,i,j]
            end
            c += A[i,j]
        end
        
        for j in OneTo(K)
            A[i,j] /= c
        end
    end
end

# In-place update of the observations distributions.
function update_B!(B::AbstractVector, γ::AbstractMatrix, observations)
    @argcheck size(γ, 1) == size(observations, 1)
    @argcheck size(γ, 2) == size(B, 1)
    K = length(B)
    for i in OneTo(K)
        if sum(γ[:,i]) > 0
            B[i] = fit_mle(typeof(B[i]), permutedims(observations), γ[:,i])
        end
    end
end

function fit_mle!(hmm::AbstractHMM, observations; tol = 1e-3, maxiter = 100, display = :none)
    T, K = size(observations, 1), size(hmm, 1)

    # Allocate memory for in-place updates
    c = zeros(T)
    α = zeros(T, K)
    β = zeros(T, K)
    γ = zeros(T, K)
    ξ = zeros(T, K, K)
    LL = zeros(T, K)

    loglikelihoods!(LL, hmm, observations)

    forwardlog!(α, c, hmm.a, hmm.A, LL)
    backwardlog!(β, c, hmm.a, hmm.A, LL)
    posteriors!(γ, α, β)

    logtot = sum(log.(c))
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it in 1:maxiter
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, ξ, α, β, LL)
        update_B!(hmm.B, γ, observations)

        loglikelihoods!(LL, hmm, observations)
    
        forwardlog!(α, c, hmm.a, hmm.A, LL)
        backwardlog!(β, c, hmm.a, hmm.A, LL)
        posteriors!(γ, α, β)

        logtotp = sum(log.(c))
        (display == :iter) && println("Iteration $it: logtot = $logtotp")

        # The likelihood should never decrease.
        # We should probably use propre tests for this instead...
        if logtotp < logtot + eps()
            @warn "The likelihood has decreased during the EM step. This is probably a bug."
        end

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) && println("EM converged in $it iterations, logtot = $logtotp")
            return
        end

        logtot = logtotp
    end

    (display in [:iter, :final]) && println("EM has not converged after $maxiter iterations, logtot = $logtot")
end

"""
    fit_mle(hmm, observations; kwargs...) -> AbstractHMM

Estimate the HMM parameters using the EM (Baum-Welch) algorithm.

# Arguments
- `display::Symbol` (`:none` by default): when to display convergence logs, can be set to `:iter` or `:final`.
- `init::Symbol` (`:none` by default): if set to `:kmeans` the HMM parameters will be initialized using a K-means clustering.
- `maxiter::Integer` (100 by default): maximum number of iterations to perform.
- `tol::Integer` (1e-3 by default): stop the algorithm when the improvement in the log-likelihood is less than `tol`.

# Output
- `<:AbstractHMM`: a copy of the original HMM with the updated parameters.
"""
function fit_mle(hmm::AbstractHMM, observations; init = :none, kwargs...)
    hmm = copy(hmm)

    if init == :kmeans
        kmeans_init!(hmm, observations, display = get(kwargs, :display, :none))
    end

    fit_mle!(hmm, observations; kwargs...)
    hmm
end
