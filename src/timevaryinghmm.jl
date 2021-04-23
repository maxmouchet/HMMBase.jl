
"""
    HMM([a, ]A, B) -> HMM

Build an HMM with transition matrix `A` and observation distributions `B`.  
If the initial state distribution `a` is not specified, a uniform distribution is assumed. 

Observations distributions can be of different types (for example `Normal` and `Exponential`),  
but they must be of the same dimension.

Alternatively, `B` can be an emission matrix where `B[i,j]` is the probability of observing symbol `j` in state `i`.

**Arguments**
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractMatrix{T}`: transition matrix.
- `B::AbstractVector{<:Distribution{F}}`: observations distributions.
- or `B::AbstractMatrix`: emission matrix.

**Example**
```julia
using Distributions, HMMBase
# from distributions
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
# from an emission matrix
hmm = HMM([0.9 0.1; 0.1 0.9], [0. 0.5 0.5; 0.25 0.25 0.5])
```
"""
struct TimeVaryingHMM{F,T} <: AbstractHMM{F}
    a::Vector{T}
    A::AbstractArray{T,3}
    B::Matrix{Distribution{F}}
    TimeVaryingHMM{F,T}(a, A, B) where {F,T} = assert_timevaryinghmm(a, A, B) && new(a, A, B)
end

TimeVaryingHMM(
    a::AbstractVector{T},
    A::AbstractArray{T,3},
    B::AbstractMatrix{<:Distribution{F}},
) where {F,T} = TimeVaryingHMM{F,T}(a, A, B)

TimeVaryingHMM(A::AbstractArray{T,3}, B::AbstractMatrix{<:Distribution{F}}) where {F,T} =
    TimeVaryingHMM{F,T}(ones(size(A, 1)) ./ size(A, 1), A, B)

function assert_timevaryinghmm(a::AbstractVector, A::AbstractArray{T,3} where T, B::AbstractMatrix{<:Distribution})
    @argcheck isprobvec(a)
    @argcheck istransmats(A)
    @argcheck all(length.(B) .== length(B[1])) ArgumentError("All distributions must have the same dimensions")
    @argcheck size(A,3) + 1 == size(B,2) ArgumentError("Number of transition rates must match length of chain")
    @argcheck length(a) == size(A,1) == size(B,1)
    return true
end

istransmats(A::AbstractArray{T,3}) where {T} = all(i->istransmat(@view A[:,:,i]), 1:size(A,3))
    
==(h1::TimeVaryingHMM, h2::TimeVaryingHMM) = (h1.a == h2.a) && (h1.A == h2.A) && (h1.B == h2.B)

function rand(
    rng::AbstractRNG,
    hmm::TimeVaryingHMM;
    init = rand(rng, Categorical(hmm.a)),
    seq = false,
)
    T = size(hmm.B, 2)
    z = Vector{Int}(undef, T)
    (T >= 1) && (z[1] = init)
    for t = 2:T
        z[t] = rand(rng, Categorical(hmm.A[z[t-1],:,t-1]))
    end
    y = rand(rng, hmm, z)
    seq ? (z, y) : y
end

function rand(rng::AbstractRNG, hmm::TimeVaryingHMM{Univariate}, z::AbstractVector{<:Integer})
    y = Vector{Float64}(undef, length(z))
    for t in eachindex(z)
        y[t] = rand(rng, hmm.B[z[t],t])
    end
    y
end

function rand(
    rng::AbstractRNG,
    hmm::TimeVaryingHMM{Multivariate},
    z::AbstractVector{<:Integer},
)
    y = Matrix{Float64}(undef, length(z), size(hmm,2))
    for t in eachindex(z)
        y[t, :] = rand(rng, hmm.B[z[t],t])
    end
    y
end

"""
    size(hmm, [dim]) -> Int | Tuple

Return the number of states in `hmm`, the dimension of the observations and the length of the chain.
"""
size(hmm::TimeVaryingHMM, dim = :) = (size(hmm.B, 1), length(hmm.B[1]), size(hmm.B, 2))[dim]

copy(hmm::TimeVaryingHMM) = TimeVaryingHMM(copy(hmm.a), copy(hmm.A), copy(hmm.B))

function nparams(hmm::TimeVaryingHMM)
    (length(hmm.a) - 1) + (size(hmm.A,1) * size(hmm.A,2) - size(hmm.A,1)) * size(hmm.A,3)
end


###

function loglikelihoods!(LL::AbstractMatrix, hmm::TimeVaryingHMM{Univariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, K)
    @argcheck T == size(hmm, 3)
    @inbounds for i in OneTo(K), t in OneTo(T)
        LL[t,i] = logpdf(hmm.B[i,t], observations[t])
    end
end

function loglikelihoods!(LL::AbstractMatrix, hmm::TimeVaryingHMM{Multivariate}, observations)
    T, K = size(observations, 1), size(hmm, 1)
    @argcheck size(LL) == (T, K)
    @argcheck T == size(hmm, 3)
    @inbounds for i in OneTo(K), t in OneTo(T)
        LL[t,i] = logpdf(hmm.B[i,t], view(observations, t, :))
    end
end



####

function update_A!(
    A::AbstractArray{T,3} where {T},
    ξ::AbstractArray,
    α::AbstractMatrix,
    β::AbstractMatrix,
    LL::AbstractMatrix,
)
    @argcheck size(α, 1) == size(β, 1) == size(LL, 1) == size(ξ, 1) == size(A,3) + 1
    @argcheck size(α, 2) ==
              size(β, 2) ==
              size(LL, 2) ==
              size(A, 1) ==
              size(A, 2) ==
              size(ξ, 2) ==
              size(ξ, 3)

    T, K = size(LL)

    @inbounds for t in OneTo(T - 1)
        m = vec_maximum(view(LL, t + 1, :))
        c = 0.0

        for i in OneTo(K), j in OneTo(K)
            ξ[t, i, j] = α[t, i] * A[i, j, t] * exp(LL[t+1, j] - m) * β[t+1, j]
            c += ξ[t, i, j]
        end

        for i in OneTo(K), j in OneTo(K)
            ξ[t, i, j] /= c
        end
    end

    fill!(A, 0.0)

    @inbounds for t in OneTo(T - 1)
        for i in OneTo(K)
            c = 0.0

            for j in OneTo(K)
                A[i, j, t] += ξ[t, i, j]
                c += A[i, j, t]
            end

            for j in OneTo(K)
                A[i, j, t] /= c
            end
        end
    end
end

# # In-place update of the observations distributions.
# function update_B!(B::AbstractMatrix, γ::AbstractMatrix, observations, estimator)
#     @argcheck size(γ, 1) == size(observations, 1)
#     @argcheck size(γ, 2) == size(B, 1)
#     K = length(B)
#     for i in OneTo(K)
#         if sum(γ[:, i]) > 0
#             B[i] = estimator(typeof(B[i]), permutedims(observations), γ[:, i])
#         end
#     end
# end

function fit_mle!(
    hmm::TimeVaryingHMM,
    observations;
    display = :none,
    maxiter = 100,
    tol = 1e-3,
    robust = false,
    estimator = fit_mle,
    fit_dists = false
)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0
    @argcheck fit_dists === false "Not yet supported"

    T, K = size(observations, 1), size(hmm, 1)
    @argcheck T == size(hmm.B, 2)
    history = EMHistory(false, 0, [])

    # Allocate memory for in-place updates
    c = zeros(T)
    α = zeros(T, K)
    β = zeros(T, K)
    γ = zeros(T, K)
    ξ = zeros(T, K, K)
    LL = zeros(T, K)

    loglikelihoods!(LL, hmm, observations)
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))

    forwardlog!(α, c, hmm.a, hmm.A, LL)
    backwardlog!(β, c, hmm.a, hmm.A, LL)
    posteriors!(γ, α, β)

    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter
        update_a!(hmm.a, α, β)
        update_A!(hmm.A, ξ, α, β, LL)
        fit_dists && update_B!(hmm.B, γ, observations, estimator)

        # Ensure the "connected-ness" of the states,
        # this prevents case where there is no transitions
        # between two extremely likely observations.
        robust && (hmm.A .+= eps())

        @check isprobvec(hmm.a)
        @check istransmats(hmm.A)

        if fit_dists
            loglikelihoods!(LL, hmm, observations)
            robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
        end

        forwardlog!(α, c, hmm.a, hmm.A, LL)
        backwardlog!(β, c, hmm.a, hmm.A, LL)
        posteriors!(γ, α, β)

        logtotp = sum(c)
        (display == :iter) && println("Iteration $it: logtot = $logtotp")

        push!(history.logtots, logtotp)
        history.iterations += 1

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history.converged = true
            break
        end

        logtot = logtotp
    end

    if !history.converged
        if display in [:iter, :final]
            println("EM has not converged after $(history.iterations) iterations, logtot = $logtot")
        end
    end

    history
end






# In-place forward pass, where α and c are allocated beforehand.
function forwardlog!(
    α::AbstractMatrix,
    c::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractMatrix,
)
    @argcheck size(α, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(α, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    T, K = size(LL)
    @argcheck T == size(A,3) + 1

    fill!(α, 0.0)
    fill!(c, 0.0)

    m = vec_maximum(view(LL, 1, :))

    for j in OneTo(K)
        α[1, j] = a[j] * exp(LL[1, j] - m)
        c[1] += α[1, j]
    end

    for j in OneTo(K)
        α[1, j] /= c[1]
    end

    c[1] = log(c[1]) + m

    @inbounds for t = 2:T
        m = vec_maximum(view(LL, t, :))

        for j in OneTo(K)
            for i in OneTo(K)
                α[t,j] += α[t-1, i] * A[i,j,t-1]
            end
            α[t, j] *= exp(LL[t, j] - m)
            c[t] += α[t, j]
        end

        for j in OneTo(K)
            α[t, j] /= c[t]
        end

        c[t] = log(c[t]) + m
    end
end

# In-place backward pass, where β and c are allocated beforehand.
function backwardlog!(
    β::AbstractMatrix,
    c::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractMatrix,
)
    @argcheck size(β, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(β, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    T, K = size(LL)
    @argcheck T == size(A,3) + 1
    L = zeros(K)
    (T == 0) && return

    fill!(β, 0.0)
    fill!(c, 0.0)

    for j in OneTo(K)
        β[end, j] = 1.0
    end

    @inbounds for t = T-1:-1:1
        m = vec_maximum(view(LL, t + 1, :))

        for i in OneTo(K)
            L[i] = exp(LL[t+1, i] - m)
        end

        for j in OneTo(K)
            for i in OneTo(K)
                β[t,j] += β[t+1, i] * A[j,i,t] * L[i]
            end
            c[t+1] += β[t, j]
        end

        for j in OneTo(K)
            β[t, j] /= c[t+1]
        end

        c[t+1] = log(c[t+1]) + m
    end

    m = vec_maximum(view(LL, 1, :))

    for j in OneTo(K)
        c[1] += a[j] * exp(LL[1, j] - m) * β[1, j]
    end

    c[1] = log(c[1]) + m
end

function forward(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix)
    m = Matrix{Float64}(undef, size(LL))
    c = Vector{Float64}(undef, size(LL, 1))
    forwardlog!(m, c, a, A, LL)
    m, sum(c)
end

function backward(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix)
    m = Matrix{Float64}(undef, size(LL))
    c = Vector{Float64}(undef, size(LL, 1))
    backwardlog!(m, c, a, A, LL)
    m, sum(c)
end

function posteriors(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix; kwargs...)
    α, _ = forward(a, A, LL; kwargs...)
    β, _ = backward(a, A, LL; kwargs...)
    posteriors(α, β)
end

function viterbilog!(
    T1::AbstractMatrix,
    T2::AbstractMatrix,
    z::AbstractVector,
    a::AbstractVector,
    A::AbstractArray{T,3} where {T},
    LL::AbstractMatrix,
)
    T, K = size(LL)
    @argcheck T == size(A,3) + 1
    (T == 0) && return

    fill!(T1, 0.0)
    fill!(T2, 0)

    al = log.(a)
    Al = log.(A)

    for i in OneTo(K)
        T1[1, i] = al[i] + LL[1, i]
    end

    @inbounds for t = 2:T
        for j in OneTo(K)
            amax = 0
            vmax = -Inf

            for i in OneTo(K)
                v = T1[t-1, i] + Al[i, j, t-1]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end

            T1[t, j] = vmax + LL[t, j]
            T2[t, j] = amax
        end
    end

    z[T] = argmax(T1[T, :])
    for t = T-1:-1:1
        z[t] = T2[t+1, z[t+1]]
    end
end

function viterbi(a::AbstractVector, A::AbstractArray{T,3} where {T}, LL::AbstractMatrix)
    T1 = Matrix{Float64}(undef, size(LL))
    T2 = Matrix{Int}(undef, size(LL))
    z = Vector{Int}(undef, size(LL, 1))
    viterbilog!(T1, T2, z, a, A, LL)
    z
end
