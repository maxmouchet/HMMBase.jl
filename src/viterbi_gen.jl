# Convenience functions

# The following methods are defined:
# viterbi(a, A, L)              -> z
# viterbi(hmm, observations)    -> z
# viterbilog(a, A, LL)          -> z
# viterbilog(hmm, observations) -> z

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

function viterbilog(a::AbstractVector, A::AbstractMatrix, LL::AbstractMatrix)
    T1 = Matrix{Float64}(undef, size(LL))
    T2 = Matrix{Int}(undef, size(LL))
    z = Vector{Int}(undef, size(LL,1))
    viterbilog!(T1, T2, z, a, A, LL)
    z
end

function viterbilog(hmm::AbstractHMM, observations)
    viterbilog(hmm.π0, hmm.π, loglikelihoods(hmm, observations))
end