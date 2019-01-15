__precompile__()

module HMMBase

using Distributions
using StaticArrays

import Base: rand
import StatsFuns: logsumexp

export
    # hmm.jl
    HMM,
    StaticHMM,
    assert_hmm,
    rand,
    # decoding.jl
    viterbi,
    # filtering.jl
    messages_backwards,
    messages_backwards_log,
    messages_forwards,
    messages_forwards_log,
    forward_backward,
    # utils.jl,
    compute_transition_matrix

# TEMP: https://github.com/JuliaStats/Distributions.jl/issues/812
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
include("decoding.jl")
include("filtering.jl")
include("utils.jl")

end