"""
Hidden Markov Models for Julia.
    
[Documentation](https://maxmouchet.github.io/HMMBase.jl/stable/).  
[Issues](https://github.com/maxmouchet/HMMBase.jl/issues).
"""
module HMMBase

using ArgCheck
using Clustering
using Distributions
using Hungarian
using LinearAlgebra

using Base: OneTo
using Random: AbstractRNG, GLOBAL_RNG

# Extended functions
import Base: ==, copy, rand, size
import Distributions: MixtureModel, fit_mle, loglikelihood

export
    # hmm.jl
    AbstractHMM,
    HMM,
    copy,
    rand,
    size,
    nparams,
    permute,
    statdists,
    istransmat,
    likelihoods,
    # messages_api.jl
    forward,
    backward,
    posteriors,
    loglikelihood,
    # mle_api.jl
    fit_mle,
    # viterbi_api.jl
    viterbi,
    # utilities.jl,
    gettransmat,
    randtransmat,
    remapseq

include("hmm.jl")
include("mle.jl")
include("mle_api.jl")
include("mle_init.jl")
include("messages.jl")
include("messages_api.jl")
include("viterbi.jl")
include("viterbi_api.jl")
include("likelihoods.jl")
include("likelihoods_api.jl")
include("utilities.jl")
include("experimental.jl")

# To be removed in a future version
# ---------------------------------
#!format: off

export n_parameters,
    log_likelihoods,
    forward_backward,
    messages_backwards,
    messages_backwards_log,
    messages_forwards,
    messages_forwards_log,
    compute_transition_matrix,
    rand_transition_matrix

@deprecate n_parameters(hmm) nparams(hmm)
@deprecate log_likelihoods(hmm, observations) likelihoods(hmm, observations, logl = true)

@deprecate forward_backward(init_distn, trans_matrix, log_likelihoods) posteriors(init_distn, trans_matrix, log_likelihoods, logl = true)
@deprecate messages_forwards(init_distn, trans_matrix, log_likelihoods) forward(init_distn, trans_matrix, log_likelihoods, logl = true)
@deprecate messages_backwards(init_distn, trans_matrix, log_likelihoods) backward(init_distn, trans_matrix, log_likelihoods, logl = true)

@deprecate forward_backward(hmm, observations) posteriors(hmm, observations, logl = true)
@deprecate messages_forwards(hmm, observations) forward(hmm, observations, logl = true)
@deprecate messages_backwards(hmm, observations) backward(hmm, observations, logl = true)

@deprecate messages_forwards_log(init_distn, trans_matrix, log_likelihoods) log.(forward(init_distn, trans_matrix, log_likelihoods, logl = true)[1])
@deprecate messages_backwards_log(trans_matrix, log_likelihoods) log.(backward(init_distn, trans_matrix, log_likelihoods, logl = true)[1])

@deprecate compute_transition_matrix(seq) gettransmat(seq, relabel = true)
@deprecate rand_transition_matrix(K, α = 1.0) randtransmat(K, α)

end
