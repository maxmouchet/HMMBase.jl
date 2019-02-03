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
    "text": "(View project on GitHub)A lightweight and efficient hidden Markov model abstraction for Julia.# HMMs with arbitrary observations distributions\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Gamma(1,1)])\n\n# MLE estimation\n# TODOLogo: Blockchain by Pablo Rozenberg from the Noun Project."
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
    "location": "inference/#Forward-backward-1",
    "page": "Inference",
    "title": "Forward-backward",
    "category": "section",
    "text": ""
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
    "text": "viterbi(init_distn::Vector, trans_matrix::Matrix, likelihoods::Matrix)\n\nFind the most likely hidden state sequence, see Viterbi algorithm.\n\n\n\n\n\nviterbi(trans_matrix::Matrix, likelihoods::Matrix)\n\nAssume an uniform initial distribution.\n\n\n\n\n\nviterbi(hmm::HMM, observations::Vector)\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)]);\nz, y = rand(hmm, 1000);\nz_viterbi = viterbi(hmm, y[:])\nz == z_viterbi\n\n\n\n\n\n"
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
    "location": "sampling/#Sampling-1",
    "page": "Sampling",
    "title": "Sampling",
    "category": "section",
    "text": "rand(hmm::AbstractHMM, T::Int)\nrand(hmm::AbstractHMM, z::AbstractVector{Int})"
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
    "text": "using Distributions\nusing HMMBase\nusing Plots\n\nπ = [0.9 0.1; 0.2 0.8]\nD = [Categorical([0.9, 0.1]), Categorical([0.2, 0.8])]\nhmm = HMM(π, D)\n\nz, y = rand(hmm, 250)\nplot(y)#-This page was generated using Literate.jl."
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
