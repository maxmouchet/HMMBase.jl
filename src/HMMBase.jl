__precompile__()

"""
Hidden Markov Models for Julia.
"""
module HMMBase

using ArgCheck
using Distributions

import Base: copy, rand, size
import Distributions: fit_mle

export
    # hmm.jl
    AbstractHMM,
    HMM,
    copy,
    rand,
    size,
    nparams,
    permute,
    istransmat,
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
    fit_mle,
    # viterbi*.jl
    viterbi,
    viterbilog,
    # utils.jl,
    compute_transition_matrix,
    rand_transition_matrix

include("hmm.jl")
include("mle.jl")
include("messages.jl")
include("messages_log.jl")
include("messages_gen.jl")
include("viterbi.jl")
include("viterbi_log.jl")
include("viterbi_gen.jl")
include("utilities.jl")

# To be removed in a future version
# ---------------------------------

export
    n_parameters,
    log_likelihoods,
    forward_backward,
    messages_backwards,
    messages_backwards_log,
    messages_forwards,
    messages_forwards_log

@deprecate n_parameters(hmm) nparams(hmm)
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
