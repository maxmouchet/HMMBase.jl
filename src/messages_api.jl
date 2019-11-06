# Forward/Backward

# {forward,backward}(a, A, L)
for f in (:forward, :backward)
    f!  = Symbol("$(f)!")    # forward!
    fl! = Symbol("$(f)log!") # forwardlog!

    @eval begin
        """
            $($f)(a, A, L; ...) -> (Vector, Float)

        Compute $($f) probabilities using samples likelihoods.
        See [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
        
        # Arguments
        - `a::AbstractVector`: initial probabilities vector.
        - `A::AbstractMatrix`: transition matrix.
        - `L::AbstractMatrix`: (log-)likelihoods.
        - `logl`: see [common options](@ref common_options).

        # Output
        - `Vector{Float64}`: $($f) probabilities.
        - `Float64`: log-likelihood of the observed sequence.
        """
        function $(f)(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix; logl = false)
            m = Matrix{Float64}(undef, size(L))
            c = Vector{Float64}(undef, size(L)[1])
            if logl
                $(fl!)(m, c, a, A, L)
            else
                warn_logl(L)
                $(f!)(m, c, a, A, L)
            end
            m, sum(log.(c))
        end
    end
end

# {forward,backward}(hmm, observations)
for f in (:forward, :backward)
    f!  = Symbol("$(f)!")    # forward!
    fl! = Symbol("$(f)log!") # forwardlog!

    @eval begin
        """
            $($f)(hmm, observations; ...) -> (Vector, Float)

        Compute $($f) probabilities of the `observations` given the `hmm` model.

        # Arguments
        - `logl`: see [common options](@ref common_options).

        # Output
        - `Vector{Float64}`: $($f) probabilities.
        - `Float64`: log-likelihood of the observed sequence.

        # Example
        ```julia
        hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
        z, y = rand(hmm, 1000)
        probs, tot = $($f)(hmm, y)
        ```
        """
        function $(f)(hmm::AbstractHMM, observations; logl = false, robust = false)
            L = likelihoods(hmm, observations, logl = logl, robust = robust)
            $(f)(hmm.a, hmm.A, L, logl = logl)
        end
    end
end


# Posteriors

"""
    posteriors(α, β)

Compute posterior probabilities from `α` and `β`.
"""
function posteriors(α::AbstractMatrix, β::AbstractMatrix)
    γ = Matrix{Float64}(undef, size(α))
    posteriors!(γ, α, β)
    γ
end

"""
    posteriors(a, A, L)

Compute posterior probabilities using samples likelihoods.
"""
function posteriors(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix; kwargs...)
    α, _ = forward(a, A, L; kwargs...)
    β, _ = backward(a, A, L; kwargs...)
    posteriors(α, β)
end

"""
    posteriors(hmm, observations)

Compute posterior probabilities using samples likelihoods.

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
γ = posteriors(hmm, y)
```
"""
function posteriors(hmm::AbstractHMM, observations; logl = false, robust = false)
    L = likelihoods(hmm, observations, logl = logl, robust = robust)
    posteriors(hmm.a, hmm.A, L, logl = logl)
end

# Likelihood

function loglikelihood(hmm::AbstractHMM, observations; logl = false, robust = false)
    forward(hmm, observations, logl = logl, robust = robust)[2]
end