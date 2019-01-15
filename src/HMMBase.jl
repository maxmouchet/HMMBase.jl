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

include("hmm.jl")
include("decoding.jl")
include("filtering.jl")
include("utils.jl")

end