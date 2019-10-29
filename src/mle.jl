# TODO: Viterbi EM

# In-place update of the initial state distribution.
function update_a!(a::AbstractVector, α::AbstractMatrix, β::AbstractMatrix)
    @argcheck size(α, 1) == size(β, 1)
    @argcheck size(α, 2) == size(β, 2) == size(a, 1)

    K = length(a)
    c = 0.0
    
    for i in Base.OneTo(K)
        a[i] = α[1,i] * β[1,i]
        c += a[i]
    end

    for i in Base.OneTo(K)
        a[i] /= c
    end
end

# In-place update of the transition matrix.
function update_A!(A::AbstractMatrix, ξ::AbstractArray, α::AbstractMatrix, β::AbstractMatrix, LL::AbstractMatrix)
    @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1)
    @argcheck size(α, 2) == size(β, 2) == size(LL, 2) == size(A, 1) == size(A, 2) == size(ξ, 2) == size(ξ, 3)

    T, K = size(LL)

    @inbounds for t in 1:T - 1
        m = vec_maximum(view(LL, t, :))
        c = 0.0

        for i in Base.OneTo(K), j in Base.OneTo(K)
            ξ[t,i,j] = α[t,i] * A[i,j] * exp(LL[t + 1,j] - m) * β[t + 1,j]
            c += ξ[t,i,j]
        end

        for i in Base.OneTo(K), j in Base.OneTo(K)
            ξ[t,i,j] /= c
        end
    end

    fill!(A, 0.0)

    @inbounds for i in Base.OneTo(K)
        c = 0.0

        for j in Base.OneTo(K)
            for t in 1:T - 1
                A[i,j] += ξ[t,i,j]
            end
            c += A[i,j]
        end
        
        for j in Base.OneTo(K)
            A[i,j] /= c
        end
    end
end

# In-place update of the observations distributions.
function update_B!(B::AbstractVector, γ::AbstractMatrix, observations)
    @argcheck size(γ, 1) == size(observations, 1)
    @argcheck size(γ, 2) == size(B, 1)
    K = length(B)
    for i in Base.OneTo(K)
        if sum(γ[:,i]) > 0
            B[i] = fit_mle(typeof(B[i]), permutedims(observations), γ[:,i])
        end
    end
end

function fit_mle!(hmm::AbstractHMM, observations; tol = 1e-3, maxit = 100, verbose = false)
    # TODO: In-place loglikelihoods update
    LL = loglikelihoods(hmm, observations)
    T, K = size(LL)

    # Allocate memory for in-place updates
    c = zeros(T)
    α = zeros(T, K)
    β = zeros(T, K)
    γ = zeros(T, K)
    ξ = zeros(T, K, K)

    forwardlog!(α, c, hmm.a, hmm.A, LL)
    backwardlog!(β, c, hmm.a, hmm.A, LL)
    posteriors!(γ, α, β)

    logtot = sum(log.(c))
    verbose && println("Iteration 0: logtot = $logtot")

    for it in 1:maxit
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, ξ, α, β, LL)
        update_B!(hmm.B, γ, observations)

        LL = loglikelihoods(hmm, observations)

        forwardlog!(α, c, hmm.a, hmm.A, LL)
        backwardlog!(β, c, hmm.a, hmm.A, LL)
        posteriors!(γ, α, β)

        logtotp = sum(log.(c))
        verbose && println("Iteration $it: logtot = $logtotp")

        if abs(logtotp - logtot) < tol
            break
        end

        logtot = logtotp
    end
end

function fit_mle(hmm::AbstractHMM, observations; kwargs...)
    hmm = copy(hmm)
    fit_mle!(hmm, observations; kwargs...)
    hmm
end
