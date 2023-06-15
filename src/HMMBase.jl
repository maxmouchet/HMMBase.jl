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
    TimeVaryingHMM,
    fit_mle!,
    copy,
    rand,
    size,
    nparams,
    permute,
    statdists,
    istransmat,
    likelihoods,
    # messages.jl
    forward,
    backward,
    posteriors,
    loglikelihood,
    # likelihoods.jl
    loglikelihoods,
    # mle.jl
    fit_mle,
    # viterbi.jl
    viterbi,
    # utilities.jl,
    gettransmat,
    randtransmat,
    remapseq

include("hmm.jl")
include("mle.jl")
include("mle_init.jl")
include("messages.jl")
include("viterbi.jl")
include("likelihoods.jl")
include("utilities.jl")
include("experimental.jl")

include("timevaryinghmm.jl")

# To be removed in a future version
# ---------------------------------
#!format: off

# < v1.1

export likelihoods

function likelihoods(args...; logl = false, kwargs...)
    @warn "`likelihoods(...)` is deprecated, use `loglikelihoods(...)` or `exp.(loglikelihoods(...))` instead."
    logl ? loglikelihoods(args...; kwargs...) : exp.(loglikelihoods(args...; kwargs...))
end

function deprecate_kwargs(name)
    @warn "`$(name)` keyword argument is deprecated."
end

# < v1.0

@deprecate n_parameters(hmm) nparams(hmm)
@deprecate log_likelihoods(hmm, observations) loglikelihoods(hmm, observations)

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
