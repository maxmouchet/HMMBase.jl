__precompile__()

"""
Hidden Markov Models for Julia.
"""
module HMMBase

using ArgCheck
using Distributions

import Base: rand, size
import LinearAlgebra: mul!, transpose
import StatsFuns: logsumexp

export
    # hmm.jl
    AbstractHMM,
    HMM,
    assert_hmm,
    rand,
    istransmat,
    n_parameters,
    likelihoods,
    loglikelihoods,
    # messages.jl
    messages_backwards,
    messages_backwards_log,
    messages_forwards,
    messages_forwards_log,
    forward_backward,
    # mle.jl
    mle_step,
    fit_mle,
    fit_mle!,
    # viterbi.jl
    viterbi,
    # utils.jl,
    compute_transition_matrix

include("hmm.jl")
include("messages.jl")
include("mle.jl")
include("viterbi.jl")
include("utilities.jl")

@deprecate log_likelihoods(hmm, observations) loglikelihoods(hmm, observations)
end
