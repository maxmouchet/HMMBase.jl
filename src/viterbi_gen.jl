# Convenience functions

# The following methods are defined:
# viterbi(a, A, L)           -> z
# viterbi(hmm, observations) -> z

function viterbi(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix)
    T1 = Matrix{Float64}(undef, size(L))
    T2 = Matrix{Int}(undef, size(L))
    z = Vector{Int}(undef, size(L,1))
    viterbi!(T1, T2, z, a, A, L)
    z
end

function viterbi(hmm::AbstractHMM, observations)
    viterbi(hmm.π0, hmm.π, likelihoods(hmm, observations))
end