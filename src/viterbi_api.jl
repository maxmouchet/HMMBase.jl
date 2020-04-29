# Convenience functions

# The following methods are defined:
# viterbi(a, A, L)              -> z
# viterbi(hmm, observations)    -> z

"""
    viterbi(a, A, L; logl) -> Vector

Find the most likely hidden state sequence, see [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm).
"""
function viterbi(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix; logl = false)
    T1 = Matrix{Float64}(undef, size(L))
    T2 = Matrix{Int}(undef, size(L))
    z = Vector{Int}(undef, size(L, 1))
    if logl
        viterbilog!(T1, T2, z, a, A, L)
    else
        warn_logl(L)
        viterbi!(T1, T2, z, a, A, L)
    end
    z
end

"""
    viterbi(hmm, observations; logl, robust) -> Vector

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000)
zv = viterbi(hmm, y)
```
"""
function viterbi(hmm::AbstractHMM, observations; robust = false, kwargs...)
    L = likelihoods(hmm, observations; robust = robust, kwargs...)
    viterbi(hmm.a, hmm.A, L; kwargs...)
end
