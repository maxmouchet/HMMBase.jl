var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Home-1",
    "page": "Home",
    "title": "Home",
    "category": "section",
    "text": "(View project on GitHub)HMMBase provides a lightweight and efficient abstraction for hidden Markov models in Julia. Most HMMs libraries only support discrete (e.g. categorical) or normal distributions. In contrast HMMBase builds upon Distributions.jl to support arbitrary univariate and multivariate distributions.  The goal is to provide well-tested and fast implementations of the basic HMMs algorithms such as the forward-backward algorithm, the Viterbi algorithm, and the MLE estimator. More advanced models, such as Bayesian HMMs, can be built upon HMMBase."
},

{
    "location": "#Getting-Started-1",
    "page": "Home",
    "title": "Getting Started",
    "category": "section",
    "text": "The package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run:pkg> add HMMBaseHMMBase supports any observations distributions implementing the Distribution interface from Distributions.jl.using Distributions, HMMBase\n\n# Univariate continuous observations\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Gamma(1,1)])\n\n# Multivariate continuous observations\nhmm = HMM([0.9 0.1; 0.1 0.9], [MvNormal([0.,0.],[1.,1.]), MvNormal([0.,0.],[1.,1.])])\n\n# Univariate discrete observations\nhmm = HMM([0.9 0.1; 0.1 0.9], [Categorical([0.3, 0.7]), Categorical([0.8, 0.2])])\n\n# Multivariate discrete observations\nhmm = HMM([0.9 0.1; 0.1 0.9], [Multinomial(10, [0.3, 0.7]), Multinomial(10, [0.8, 0.2])])Logo: lego by jon trillana from the Noun Project."
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
    "location": "hmm/#HMMBase.assert_hmm",
    "page": "Types",
    "title": "HMMBase.assert_hmm",
    "category": "function",
    "text": "assert_hmm(π0, π, D)\n\nThrow an ArgumentError if the initial state distribution and the transition matrix rows does not sum to 1, and if the observations distributions does not have the same dimensions.\n\n\n\n\n\n"
},

{
    "location": "hmm/#Base.size-Tuple{AbstractHMM}",
    "page": "Types",
    "title": "Base.size",
    "category": "method",
    "text": "size(hmm::AbstractHMM, [dim])\n\nReturns the number of states in the HMM and the dimension of the observations.\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nsize(hmm) # (2,1)\n\n\n\n\n\n"
},

{
    "location": "hmm/#Types-1",
    "page": "Types",
    "title": "Types",
    "category": "section",
    "text": "AbstractHMM\nHMM\nassert_hmm\nsize(::AbstractHMM)"
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
    "text": "messages_forwards\nmessages_backwards\nforward_backward"
},

{
    "location": "inference/#HMMBase.fit_mle!",
    "page": "Inference",
    "title": "HMMBase.fit_mle!",
    "category": "function",
    "text": "fit_mle!(hmm::AbstractHMM, observations; eps=1e-3, max_iterations=100, verbose=false)\n\nPerform EM (Baum-Welch) steps until max_iterations is reached, or the change in the log-likelihood is smaller than eps.\n\nExample\n\nhmm, log_likelihood = fit_mle!(hmm, observations)\n\n\n\n\n\n"
},

{
    "location": "inference/#HMMBase.mle_step",
    "page": "Inference",
    "title": "HMMBase.mle_step",
    "category": "function",
    "text": "mle_step(hmm::AbstractHMM{F}, observations) where F\n\nPerform one step of the EM (Baum-Welch) algorithm.\n\nExample\n\nhmm, log_likelihood = mle_step(hmm, observations)\n\n\n\n\n\n"
},

{
    "location": "inference/#Baum–Welch-algorithm-1",
    "page": "Inference",
    "title": "Baum–Welch algorithm",
    "category": "section",
    "text": "fit_mle!\nmle_step"
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
    "text": "using Distributions\nusing HMMBase\nusing Plots\n\nπ0 = [0.6, 0.4]\nπ = [0.7 0.3; 0.4 0.6]\nD = [MvNormal([0.0,5.0],[1.0,1.0]), MvNormal([5.0,10.0],[1.0,1.0])]\nhmm = HMM(π0, π, D)z, y = rand(hmm, 250)\nz_viterbi = viterbi(hmm, y);plot(y, linetype=:steppre, label=[\"Observations (1)\", \"Observations (2)\"], size=(600,200))plot(z, linetype=:steppre, label=\"True hidden state\", size=(600,200))plot(z_viterbi, linetype=:steppre, label=\"Viterbi decoded hidden state\", size=(600,200))This page was generated using Literate.jl."
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
    "text": "using Distributions\nusing HMMBase\nusing Plotshttps://en.wikipedia.org/wiki/Viterbi_algorithm#Exampleπ0 = [0.6, 0.4]\nπ = [0.7 0.3; 0.4 0.6]\nD = [Categorical([0.5, 0.4, 0.1]), Categorical([0.1, 0.3, 0.6])]\nhmm = HMM(π0, π, D)z, y = rand(hmm, 250)\nz_viterbi = viterbi(hmm, y);plot(y, linetype=:steppre, label=\"Observations\", size=(600,200))plot(z, linetype=:steppre, label=\"True hidden state\", size=(600,200))plot(z_viterbi, linetype=:steppre, label=\"Viterbi decoded hidden state\", size=(600,200))This page was generated using Literate.jl."
},

{
    "location": "examples/mle/#",
    "page": "MLE Estimator",
    "title": "MLE Estimator",
    "category": "page",
    "text": "EditURL = \"https://github.com/maxmouchet/HMMBase.jl/blob/master/examples/mle.jl\""
},

{
    "location": "examples/mle/#MLE-Estimator-1",
    "page": "MLE Estimator",
    "title": "MLE Estimator",
    "category": "section",
    "text": "using Distributions\nusing HMMBase\nusing Plots\n\ny1 = rand(Normal(0,2), 1000)\ny2 = rand(Normal(10,1), 500)\ny = vcat(y1, y2, y1, y2)\n\nplot(y, linetype=:steppre, size=(600,200))For now HMMBase does not handle the initialization of the parameters. Hence we must instantiate an initial HMM by hand.hmm = HMM([0.5 0.5; 0.5 0.5], [Normal(-1,1), Normal(15,1)])\nhmm, log_likelihood = fit_mle!(hmm, y, verbose=true)z_viterbi = viterbi(hmm, y)\nplot(z_viterbi, linetype=:steppre, label=\"Viterbi decoded hidden state\", size=(600,200))We can also perform individual EM steps.hmm, log_likelihood = mle_step(hmm, y)This page was generated using Literate.jl."
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
