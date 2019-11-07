# Convenience functions

# The following methods are defined:
# viterbi(a, A, L)              -> z
# viterbi(hmm, observations)    -> z

"""
    viterbi(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix) -> Vector

Find the most likely hidden state sequence, see [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm).
"""
function viterbi(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix; logl = false)
    T1 = Matrix{Float64}(undef, size(L))
    T2 = Matrix{Int}(undef, size(L))
    z = Vector{Int}(undef, size(L,1))
    if logl
        viterbilog!(T1, T2, z, a, A, L)
    else
        warn_logl(L)
        viterbi!(T1, T2, z, a, A, L)
    end
    z
end

"""
    viterbi(hmm, observations) -> Vector
"""
function viterbi(hmm::AbstractHMM, observations; logl = false, robust = false)
    L = likelihoods(hmm, observations, logl = logl, robust = robust)
    viterbi(hmm.a, hmm.A, L, logl = logl)
end
