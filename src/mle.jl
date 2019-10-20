# TODO: Viterbi EM

# In-place update of the initial state distribution.
function update_a!(a::AbstractVector, α::AbstractMatrix, β::AbstractMatrix)
    for i in 1:length(a)
        a[i] = α[1,i] * β[1,i]
    end
end

# In-place update of the transition matrix.
function update_A!(A::AbstractMatrix, ξ::AbstractArray, α::AbstractMatrix, β::AbstractMatrix, LL::AbstractMatrix)
    T, K = size(LL)

    @inbounds for t in 1:T-1
        m = vec_maximum(view(LL, t, :))
        c = 0.0

        for i in 1:K, j in 1:K
            ξ[t,i,j] = α[t,i] * A[i,j] * exp(LL[t+1,j] - m) * β[t+1,j]
            c += ξ[t,i,j]
        end

        for i in 1:K, j in 1:K
            ξ[t,i,j] /= c
        end
    end

    fill!(A, 0.0)

    @inbounds for i in 1:K
        c = 0.0

        for j in 1:K
            for t in 1:T
                A[i,j] += ξ[t,i,j]
            end
            c += A[i,j]
        end
        
        for j in 1:K
            A[i,j] /= c
        end
    end
end

# In-place update of the observations distributions.
function update_B!(B::AbstractVector, γ::AbstractMatrix, observations)
    for i in 1:length(B)
        B[i] = fit_mle(typeof(B[i]), permutedims(observations), γ[:,i])
    end
end

function fit_mle!(hmm::AbstractHMM, observations; eps=1e-3, maxit=100, verbose=false)
    # TODO: In-place loglikelihoods update
    LL = loglikelihoods(hmm, observations)
    T, K = size(LL)

    # Allocate memory for in-place updates
    c = Vector{Float64}(undef, T)
    α = Matrix{Float64}(undef, T, K)
    β = Matrix{Float64}(undef, T, K)
    γ = Matrix{Float64}(undef, T, K)
    ξ = Array{Float64}(undef, T, K, K)

    forwardlog!(α, c, hmm.π0, hmm.π, LL)
    backwardlog!(β, c, hmm.π0, hmm.π, LL)
    posteriors!(γ, α, β)

    logtot = sum(log.(c))
    verbose && println("Iteration 0: logtot = $logtot")

    for it in 1:maxit
        update_a!(hmm.π0, α, β)
        update_A!(hmm.π, ξ, α, β, LL)
        update_B!(hmm.D, γ, observations)

        LL = loglikelihoods(hmm, observations)

        forwardlog!(α, c, hmm.π0, hmm.π, LL)
        backwardlog!(β, c, hmm.π0, hmm.π, LL)
        posteriors!(γ, α, β)

        logtotp = sum(log.(c))
        println("Iteration $it: logtot = $logtotp")

        if abs(logtotp - logtot) < eps
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
