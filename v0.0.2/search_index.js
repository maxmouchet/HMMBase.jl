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
    "text": "(View project on GitHub)A lightweight and efficient hidden Markov model abstraction for Julia.Logo: Blockchain by Pablo Rozenberg from the Noun Project."
},

{
    "location": "hmm/#",
    "page": "HMM Type",
    "title": "HMM Type",
    "category": "page",
    "text": ""
},

{
    "location": "hmm/#HMMBase.HMM-Union{Tuple{F}, Tuple{Array{Float64,2},Array{Float64,1},Array{#s1,1} where #s1<:(Distribution{F,S} where S<:ValueSupport)}} where F",
    "page": "HMM Type",
    "title": "HMMBase.HMM",
    "category": "method",
    "text": "HMM(π::Matrix{Float64}, D::Vector{<:Distribution{F}}) where F\nHMM(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution{F}}) where F\n\nBuild an HMM with transition matrix π and observations distributions D.   If the initial state distribution π0 is not specified, a uniform distribution is assumed. \n\nObservations distributions can be of different types (for example Normal and Exponential).   However they must be of the same dimension (all scalars or all multivariates).\n\nExamples\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\n\n\n\n\n\n"
},

{
    "location": "hmm/#HMMBase.assert_hmm-Tuple{Array{Float64,2},Array{Float64,1},Array{#s1,1} where #s1<:Distribution}",
    "page": "HMM Type",
    "title": "HMMBase.assert_hmm",
    "category": "method",
    "text": "assert_hmm(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution})\n\nThrow an AssertionError if the initial state distribution and the transition matrix rows does not sum to 1, and if the observations distributions does not have the same dimensions.\n\n\n\n\n\n"
},

{
    "location": "hmm/#HMMBase.sample_hmm-Tuple{HMM{Univariate},Int64}",
    "page": "HMM Type",
    "title": "HMMBase.sample_hmm",
    "category": "method",
    "text": "sample_hmm(hmm::HMM{Univariate}, timesteps::Int)\nsample_hmm(hmm::HMM{Multivariate}, timesteps::Int)\n\nSample a trajectory from hmm.   Return a vector of observations for univariate HMMs and a matrix for multivariate HMMs.\n\n\n\n\n\n"
},

{
    "location": "hmm/#HMM-Type-1",
    "page": "HMM Type",
    "title": "HMM Type",
    "category": "section",
    "text": "HMM(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution{F}}) where F\nassert_hmm(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution})\nsample_hmm(hmm::HMM{Univariate}, timesteps::Int)"
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
    "text": "using Distributions\nusing HMMBase\nusing Plots\n\nπ = [0.9 0.1; 0.2 0.8]\nD = [Categorical([0.9, 0.1]), Categorical([0.2, 0.8])]\nhmm = HMM(π, D)\n\nz, y = sample_hmm(hmm, 250)\nplot(y)#-This page was generated using Literate.jl."
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
