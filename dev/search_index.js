var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "HMMBase.jl",
    "title": "HMMBase.jl",
    "category": "page",
    "text": ""
},

{
    "location": "#HMMBase.jl-1",
    "page": "HMMBase.jl",
    "title": "HMMBase.jl",
    "category": "section",
    "text": "(View project on GitHub)A lightweight and efficient hidden Markov model abstraction for Julia.# HMMBase supports any observations distributions implementing\n# the `Distribution` interface from Distributions.jl.\n\n# Univariate continuous observations\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Gamma(1,1)])\n\n# Multivariate continuous observations\nhmm = HMM([0.9 0.1; 0.1 0.9], [MvNormal([0.,0.],[1.,1.]), MvNormal([0.,0.],[1.,1.])])\n\n# Univariate discrete observations\nhmm = HMM([0.9 0.1; 0.1 0.9], [Categorical([0.3, 0.7]), Categorical([0.8, 0.2])])\n\n# Multivariate discrete observations\nhmm = HMM([0.9 0.1; 0.1 0.9], [Multinomial(10, [0.3, 0.7]), Multinomial(10, [0.8, 0.2])])\n\n# Read the manual for more information.Logo: Blockchain by Pablo Rozenberg from the Noun Project."
},

{
    "location": "hmm/#",
    "page": "Types",
    "title": "Types",
    "category": "page",
    "text": ""
},

{
    "location": "hmm/#HMMBase.AbstractHMM",
    "page": "Types",
    "title": "HMMBase.AbstractHMM",
    "category": "type",
    "text": "AbstractHMM{F<:VariateForm}\n\nAn HMM type must at-least implement the following interface:\n\nstruct CustomHMM{F,T} <: AbstractHMM{F}\n    π0::AbstractVector{T}              # Initial state distribution\n    π::AbstractMatrix{T}               # Transition matrix\n    D::AbstractVector{Distribution{F}} # Observations distributions\n    # Custom fields ....\nend\n\n\n\n\n\n"
},

{
    "location": "hmm/#HMMBase.HMM",
    "page": "Types",
    "title": "HMMBase.HMM",
    "category": "type",
    "text": "HMM([π0::AbstractVector{T}, ]π::AbstractMatrix{T}, D::AbstractVector{<:Distribution{F}}) where F where T\n\nBuild an HMM with transition matrix π and observations distributions D.   If the initial state distribution π0 is not specified, a uniform distribution is assumed. \n\nObservations distributions can be of different types (for example Normal and Exponential).   However they must be of the same dimension (all scalars or all multivariates).\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\n\n\n\n\n\n"
},

{
    "location": "hmm/#HMMBase.StaticHMM",
    "page": "Types",
    "title": "HMMBase.StaticHMM",
    "category": "type",
    "text": "StaticHMM([π0::AbstractVector{T}, ]π::AbstractMatrix{T}, D::AbstractVector{<:Distribution{F}}) where {F,T}\n\nSee HMM.\n\n\n\n\n\n"
},

{
    "location": "hmm/#HMMBase.assert_hmm",
    "page": "Types",
    "title": "HMMBase.assert_hmm",
    "category": "function",
    "text": "assert_hmm(π0::AbstractVector{Float64}, π::AbstractMatrix{Float64}, D::AbstractVector{<:Distribution})\n\nThrow an AssertionError if the initial state distribution and the transition matrix rows does not sum to 1, and if the observations distributions does not have the same dimensions.\n\n\n\n\n\n"
},

{
    "location": "hmm/#Base.size-Tuple{AbstractHMM}",
    "page": "Types",
    "title": "Base.size",
    "category": "method",
    "text": "size(hmm::AbstractHMM)\n\nReturns the number of states in the HMM and the dimension of the observations.\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nsize(hmm) # (2,1)\n\n\n\n\n\n"
},

{
    "location": "hmm/#Types-1",
    "page": "Types",
    "title": "Types",
    "category": "section",
    "text": "AbstractHMM\nHMM\nStaticHMM\nassert_hmm\nsize(::AbstractHMM)"
},

{
    "location": "inference/#",
    "page": "Inference",
    "title": "Inference",
    "category": "page",
    "text": ""
},

{
    "location": "inference/#Inference-1",
    "page": "Inference",
    "title": "Inference",
    "category": "section",
    "text": ""
},

{
    "location": "inference/#HMMBase.messages_backwards",
    "page": "Inference",
    "title": "HMMBase.messages_backwards",
    "category": "function",
    "text": "messages_backwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)\n\nCompute backward probabilities, see Forward-backward algorithm.\n\n\n\n\n\nmessages_backwards(hmm, observations)\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nz, y = rand(hmm, 1000)\nbetas, logtot = messages_backwards(hmm, y)\n\n\n\n\n\n"
},

{
    "location": "inference/#HMMBase.messages_forwards",
    "page": "Inference",
    "title": "HMMBase.messages_forwards",
    "category": "function",
    "text": "messages_forwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)\n\nCompute forward probabilities, see Forward-backward algorithm.\n\n\n\n\n\nmessages_forwards(hmm, observations)\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nz, y = rand(hmm, 1000)\nalphas, logtot = messages_forwards(hmm, y)\n\n\n\n\n\n"
},

{
    "location": "inference/#HMMBase.forward_backward",
    "page": "Inference",
    "title": "HMMBase.forward_backward",
    "category": "function",
    "text": "forward_backward(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)\n\n\n\n\n\nforward_backward(hmm, observations)\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nz, y = rand(hmm, 1000)\ngammas = forward_backward(hmm, y)\n\n\n\n\n\n"
},

{
    "location": "inference/#Forward-backward-1",
    "page": "Inference",
    "title": "Forward-backward",
    "category": "section",
    "text": "messages_backwards\nmessages_forwards\nforward_backward"
},

{
    "location": "inference/#Baum–Welch-algorithm-1",
    "page": "Inference",
    "title": "Baum–Welch algorithm",
    "category": "section",
    "text": ""
},

{
    "location": "inference/#HMMBase.viterbi",
    "page": "Inference",
    "title": "HMMBase.viterbi",
    "category": "function",
    "text": "viterbi(init_distn::AbstractVector, trans_matrix::AbstractMatrix, likelihoods::AbstractMatrix)\n\nFind the most likely hidden state sequence, see Viterbi algorithm.\n\n\n\n\n\nviterbi(trans_matrix::AbstractMatrix, likelihoods::AbstractMatrix)\n\nAssume an uniform initial distribution.\n\n\n\n\n\nviterbi(hmm::HMM, observations::Vector)\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)]);\nz, y = rand(hmm, 1000);\nz_viterbi = viterbi(hmm, y)\nz == z_viterbi\n\n\n\n\n\n"
},

{
    "location": "inference/#Viterbi-1",
    "page": "Inference",
    "title": "Viterbi",
    "category": "section",
    "text": "viterbi"
},

{
    "location": "sampling/#",
    "page": "Sampling",
    "title": "Sampling",
    "category": "page",
    "text": ""
},

{
    "location": "sampling/#Base.rand-Tuple{AbstractHMM,Int64}",
    "page": "Sampling",
    "title": "Base.rand",
    "category": "method",
    "text": "rand(hmm::AbstractHMM, T::Int[, initial_state::Int])\n\nGenerate a random trajectory of hmm for T timesteps.\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nz, y = rand(hmm, 1000)\n\n\n\n\n\n"
},

{
    "location": "sampling/#Base.rand-Tuple{AbstractHMM,AbstractArray{Int64,1}}",
    "page": "Sampling",
    "title": "Base.rand",
    "category": "method",
    "text": "rand(hmm::AbstractHMM, z::AbstractVector{Int})\n\nGenerate observations from hmm according to trajectory z.\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\ny = rand(hmm, [1, 1, 2, 2, 1])\n\n\n\n\n\n"
},

{
    "location": "sampling/#Sampling-1",
    "page": "Sampling",
    "title": "Sampling",
    "category": "section",
    "text": "rand(hmm::AbstractHMM, T::Int)\nrand(hmm::AbstractHMM, z::AbstractVector{Int})"
},

{
    "location": "examples/continuous_obs/#",
    "page": "HMM with continuous observations",
    "title": "HMM with continuous observations",
    "category": "page",
    "text": "EditURL = \"https://github.com/maxmouchet/HMMBase.jl/blob/master/examples/continuous_obs.jl\""
},

{
    "location": "examples/continuous_obs/#HMM-with-continuous-observations-1",
    "page": "HMM with continuous observations",
    "title": "HMM with continuous observations",
    "category": "section",
    "text": "using Distributions\nusing HMMBase\nusing Plots\n\nπ0 = [0.6, 0.4]\nπ = [0.7 0.3; 0.4 0.6]\nD = [MvNormal([0.0,5.0],[1.0,1.0]), MvNormal([5.0,10.0],[1.0,1.0])]\nhmm = HMM(π0, π, D)z, y = rand(hmm, 250)\nz_viterbi = viterbi(hmm, y);plot(y, linetype=:steppre, label=[\"Observations (1)\", \"Observations (2)\"], size=(600,200))plot(z, linetype=:steppre, label=\"True hidden state\", size=(600,200))plot(z_viterbi, linetype=:steppre, label=\"Viterbi decoded hidden state\", size=(600,200))#-This page was generated using Literate.jl."
},

{
    "location": "examples/discrete_obs/#",
    "page": "HMM with discrete observations",
    "title": "HMM with discrete observations",
    "category": "page",
    "text": "EditURL = \"https://github.com/maxmouchet/HMMBase.jl/blob/master/examples/discrete_obs.jl\""
},

{
    "location": "examples/discrete_obs/#HMM-with-discrete-observations-1",
    "page": "HMM with discrete observations",
    "title": "HMM with discrete observations",
    "category": "section",
    "text": "using Distributions\nusing HMMBase\nusing Plotshttps://en.wikipedia.org/wiki/Viterbi_algorithm#Exampleπ0 = [0.6, 0.4]\nπ = [0.7 0.3; 0.4 0.6]\nD = [Categorical([0.5, 0.4, 0.1]), Categorical([0.1, 0.3, 0.6])]\nhmm = HMM(π0, π, D)z, y = rand(hmm, 250)\nz_viterbi = viterbi(hmm, y);plot(y, linetype=:steppre, label=\"Observations\", size=(600,200))plot(z, linetype=:steppre, label=\"True hidden state\", size=(600,200))plot(z_viterbi, linetype=:steppre, label=\"Viterbi decoded hidden state\", size=(600,200))#-This page was generated using Literate.jl."
},

{
    "location": "examples/static_arrays/#",
    "page": "Static arrays#-",
    "title": "Static arrays#-",
    "category": "page",
    "text": "EditURL = \"https://github.com/maxmouchet/HMMBase.jl/blob/master/examples/static_arrays.jl\""
},

{
    "location": "examples/static_arrays/#Static-arrays#-1",
    "page": "Static arrays#-",
    "title": "Static arrays#-",
    "category": "section",
    "text": "This page was generated using Literate.jl."
},

{
    "location": "_index/#",
    "page": "Index",
    "title": "Index",
    "category": "page",
    "text": ""
},

{
    "location": "_index/#Index-1",
    "page": "Index",
    "title": "Index",
    "category": "section",
    "text": ""
},

]}
