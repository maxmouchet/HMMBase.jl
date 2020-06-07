# Forward/Backward
# TODO: Rename L -> LL

# {forward,backward}(a, A, L)
for f in (:forward, :backward)
    fl! = Symbol("$(f)log!") # forwardlog!

    @eval begin
        """
            $($f)(a, A, L) -> (Vector, Float)

        Compute $($f) probabilities using samples likelihoods.
        See [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).

        **Output**
        - `Vector{Float64}`: $($f) probabilities.
        - `Float64`: log-likelihood of the observed sequence.
        """
        function $(f)(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix; logl = nothing)
            (logl !== nothing) && deprecate_kwargs("logl")
            m = Matrix{Float64}(undef, size(L))
            c = Vector{Float64}(undef, size(L)[1])
            $(fl!)(m, c, a, A, L)
            m, sum(log.(c))
        end
    end
end

# {forward,backward}(hmm, observations)
for f in (:forward, :backward)
    fl! = Symbol("$(f)log!") # forwardlog!

    @eval begin
        """
            $($f)(hmm, observations; robust) -> (Vector, Float)

        Compute $($f) probabilities of the `observations` given the `hmm` model.

        **Output**
        - `Vector{Float64}`: $($f) probabilities.
        - `Float64`: log-likelihood of the observed sequence.

        **Example**
        ```julia
        using Distributions, HMMBase
        hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
        y = rand(hmm, 1000)
        probs, tot = $($f)(hmm, y)
        ```
        """
        function $(f)(hmm::AbstractHMM, observations; logl = nothing, robust = false)
            (logl !== nothing) && deprecate_kwargs("logl")
            LL = loglikelihoods(hmm, observations; robust = robust)
            $(f)(hmm.a, hmm.A, LL)
        end
    end
end


# Posteriors

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
    posteriors(a, A, L; logl) -> Vector

Compute posterior probabilities using samples likelihoods.
"""
function posteriors(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix; kwargs...)
    α, _ = forward(a, A, L; kwargs...)
    β, _ = backward(a, A, L; kwargs...)
    posteriors(α, β)
end

"""
    posteriors(hmm, observations; logl, robust) -> Vector

Compute posterior probabilities using samples likelihoods.

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000)
γ = posteriors(hmm, y)
```
"""
function posteriors(hmm::AbstractHMM, observations; robust = false, kwargs...)
    L = loglikelihoods(hmm, observations; robust = robust, kwargs...)
    posteriors(hmm.a, hmm.A, L; kwargs...)
end

# Likelihood

"""
    loglikelihood(hmm, observations; logl, robust) -> Float64

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
function loglikelihood(hmm::AbstractHMM, observations; robust = false)
    forward(hmm, observations, robust = robust)[2]
end
