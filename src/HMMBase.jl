__precompile__()

"""
Hidden Markov Models for Julia.
"""
module HMMBase

using Distributions
using StaticArrays

import Base: rand, size
import LinearAlgebra: transpose
import StatsFuns: logsumexp

export
    # hmm.jl
    AbstractHMM,
    HMM,
    StaticHMM,
    assert_hmm,
    rand,
    # messages.jl
    messages_backwards,
    messages_backwards_log,
    messages_forwards,
    messages_forwards_log,
    forward_backward,
    # viterbi.jl
    viterbi,
    # utils.jl,
    compute_transition_matrix

# TEMP: https://github.com/JuliaStats/Distributions.jl/issues/812
# TODO: Specify minimum Distributions version in compat
function Distributions.allnonneg(x::AbstractArray{T}) where T<:Real
    for i = 1 : length(x)
        if !(x[i] >= zero(T))
            return false
        end
    end
    return true
end
Distributions.isprobvec(p::AbstractVector{T}) where {T<:Real} = Distributions.allnonneg(p) && isapprox(sum(p), one(T))

include("hmm.jl")
include("messages.jl")
include("viterbi.jl")
include("utilities.jl")

end