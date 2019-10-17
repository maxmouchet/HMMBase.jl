__precompile__()

"""
Hidden Markov Models for Julia.
"""
module HMMBase

using ArgCheck
using Distributions

import Base: rand, size
import StatsFuns: logsumexp # TODO: Remove after mle update

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
    # messages*.jl
    forward,
    forwardlog,
    backward,
    backwardlog,
    posteriors,
    posteriorslog,
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
include("messages_log.jl")
include("messages_gen.jl")
include("mle.jl")
include("viterbi.jl")
include("utilities.jl")

export
    forward_backward,
    messages_backwards,
    messages_backwards_log,
    messages_forwards,
    messages_forwards_log,

@deprecate log_likelihoods(hmm, observations) loglikelihoods(hmm, observations)
@deprecate forward_backward(init_distn, trans_matrix, log_likelihoods) posteriorslog(init_distn, trans_matrix, log_likelihoods)
@deprecate messages_forwards(init_distn, trans_matrix, log_likelihoods) forwardlog(init_distn, trans_matrix, log_likelihoods)
@deprecate messages_backwards(init_distn, trans_matrix, log_likelihoods) backwardlog(init_distn, trans_matrix, log_likelihoods)
@deprecate forward_backward(hmm, observations) posteriorslog(hmm, observations)
@deprecate messages_forwards(hmm, observations) forwardlog(hmm, observations)
@deprecate messages_backwards(hmm, observations) backwardlog(hmm, observations)
@deprecate messages_forwards_log(init_distn, trans_matrix, log_likelihoods) log.(forwardlog(init_distn, trans_matrix, log_likelihoods)[1])
@deprecate messages_backwards_log(trans_matrix, log_likelihoods) log.(backwardlog(init_distn, trans_matrix, log_likelihoods)[1])

end
