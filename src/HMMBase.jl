__precompile__()

"""
Hidden Markov Models for Julia.
"""
module HMMBase

using Distributions
using LinearAlgebra
using Printf 

import Base: rand, size
export AbstractHMM, 
  HMM, 
  forward,
  backward,
  posteriors,
  likelihoods,
  forward!,
  backward!,
  posteriors!,
  viterbi,
  nviterbi!,
  viterbi!,
  baum_welch!

include("hmm.jl")
include("messages.jl")
include("baum_welch.jl")

end
