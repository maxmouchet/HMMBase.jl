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
