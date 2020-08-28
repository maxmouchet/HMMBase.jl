# Original implementations by @nantonel
# https://github.com/maxmouchet/HMMBase.jl/pull/6

# In-place forward pass, where α and c are allocated beforehand.
function forwardlog!(
    α::AbstractMatrix,
    c::AbstractVector,
    a::AbstractVector,
    A::AbstractMatrix,
    LL::AbstractMatrix,
)
    @argcheck size(α, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(α, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    T, K = size(LL)
    (T == 0) && return

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
                α[t, j] += α[t-1, i] * A[i, j]
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
    A::AbstractMatrix,
    LL::AbstractMatrix,
)
    @argcheck size(β, 1) == size(LL, 1) == size(c, 1)
    @argcheck size(β, 2) == size(LL, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    T, K = size(LL)
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
                β[t, j] += β[t+1, i] * A[j, i] * L[i]
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

# In-place posterior computation, where γ is allocated beforehand.
function posteriors!(γ::AbstractMatrix, α::AbstractMatrix, β::AbstractMatrix)
    @argcheck size(γ) == size(α) == size(β)
    T, K = size(α)
    for t in OneTo(T)
        c = 0.0
        for i in OneTo(K)
            γ[t, i] = α[t, i] * β[t, i]
            c += γ[t, i]
        end

        for i in OneTo(K)
            γ[t, i] /= c
        end
    end
end

"""
    forward(a, A, LL) -> (Vector, Float)

Compute forward probabilities using samples likelihoods.
See [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).

**Output**
- `Vector{Float64}`: forward probabilities.
- `Float64`: log-likelihood of the observed sequence.
"""
function forward(a::AbstractVector, A::AbstractMatrix, LL::AbstractMatrix; logl = nothing)
    (logl !== nothing) && deprecate_kwargs("logl")
    m = Matrix{Float64}(undef, size(LL))
    c = Vector{Float64}(undef, size(LL, 1))
    forwardlog!(m, c, a, A, LL)
    m, sum(c)
end

"""
    backward(a, A, LL) -> (Vector, Float)

Compute backward probabilities using samples likelihoods.
See [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).

**Output**
- `Vector{Float64}`: backward probabilities.
- `Float64`: log-likelihood of the observed sequence.
"""
function backward(a::AbstractVector, A::AbstractMatrix, LL::AbstractMatrix; logl = nothing)
    (logl !== nothing) && deprecate_kwargs("logl")
    m = Matrix{Float64}(undef, size(LL))
    c = Vector{Float64}(undef, size(LL, 1))
    backwardlog!(m, c, a, A, LL)
    m, sum(c)
end

"""
    forward(hmm, observations; robust) -> (Vector, Float)

Compute forward probabilities of the `observations` given the `hmm` model.

**Output**
- `Vector{Float64}`: forward probabilities.
- `Float64`: log-likelihood of the observed sequence.

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000)
probs, tot = forward(hmm, y)
```
"""
function forward(hmm::AbstractHMM, observations; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations; robust = robust)
    forward(hmm.a, hmm.A, LL)
end

"""
    backward(hmm, observations; robust) -> (Vector, Float)

Compute forward probabilities of the `observations` given the `hmm` model.

**Output**
- `Vector{Float64}`: backward probabilities.
- `Float64`: log-likelihood of the observed sequence.

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000)
probs, tot = forward(hmm, y)
```
"""
function backward(hmm::AbstractHMM, observations; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations; robust = robust)
    backward(hmm.a, hmm.A, LL)
end

"""
    posteriors(α, β) -> Vector

Compute posterior probabilities from `α` and `β`.

**Arguments**
- `α::AbstractVector`: forward probabilities.
- `β::AbstractVector`: backward probabilities.
"""
function posteriors(α::AbstractMatrix, β::AbstractMatrix)
    γ = Matrix{Float64}(undef, size(α))
    posteriors!(γ, α, β)
    γ
end

"""
    posteriors(a, A, LL; kwargs...) -> Vector

Compute posterior probabilities using samples likelihoods.
"""
function posteriors(a::AbstractVector, A::AbstractMatrix, LL::AbstractMatrix; kwargs...)
    α, _ = forward(a, A, LL; kwargs...)
    β, _ = backward(a, A, LL; kwargs...)
    posteriors(α, β)
end

"""
    posteriors(hmm, observations; robust) -> Vector

Compute posterior probabilities using samples likelihoods.

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000)
γ = posteriors(hmm, y)
```
"""
function posteriors(hmm::AbstractHMM, observations; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations; robust = robust)
    posteriors(hmm.a, hmm.A, LL)
end

"""
    loglikelihood(hmm, observations; robust) -> Float64

Compute the log-likelihood of the observations under the model.  
This is defined as the sum of the log of the normalization coefficients in the forward filter.

**Output**
- `Float64`: log-likelihood of the observations sequence under the model.

**Example**
```jldoctest
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
loglikelihood(hmm, [0.15, 0.10, 1.35])
# output
-4.588183811489616
```
"""
function loglikelihood(hmm::AbstractHMM, observations; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    forward(hmm, observations, robust = robust)[2]
end
