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
    "location": "notations/#",
    "page": "Notations",
    "title": "Notations",
    "category": "page",
    "text": ""
},

{
    "location": "notations/#Notations-1",
    "page": "Notations",
    "title": "Notations",
    "category": "section",
    "text": "Symbol Shape Description\nK - Number of states in an HMM\nT - Number of observations\na K Initial state distribution\nA KxK Transition matrix\nB K Vector of observations distributions\nα TxK Forward filter\nβ TxK Backward filter\nγ TxK Posteriors (α * β)Before version 1.0:Symbol Shape Description\nπ0 K Initial state distribution\nπ KxK Transition matrix\nD K Vector of observation distributions"
},

{
    "location": "types/#",
    "page": "Types",
    "title": "Types",
    "category": "page",
    "text": ""
},

{
    "location": "types/#Types-1",
    "page": "Types",
    "title": "Types",
    "category": "section",
    "text": "AbstractHMM\nHMM\nistransmat\nnparams\ncopy\nsize"
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
    "location": "inference/#HMMBase.forward",
    "page": "Inference",
    "title": "HMMBase.forward",
    "category": "function",
    "text": "HMMBase.forward(a, A, L)\n\nCompute HMMBase.forward probabilities using samples likelihoods. See Forward-backward algorithm.\n\n\n\n\n\nHMMBase.forward(hmm, observations)\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nz, y = rand(hmm, 1000)\nprobs, tot = HMMBase.forward(hmm, y)\n\n\n\n\n\n"
},

{
    "location": "inference/#HMMBase.backward",
    "page": "Inference",
    "title": "HMMBase.backward",
    "category": "function",
    "text": "HMMBase.backward(a, A, L)\n\nCompute HMMBase.backward probabilities using samples likelihoods. See Forward-backward algorithm.\n\n\n\n\n\nHMMBase.backward(hmm, observations)\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nz, y = rand(hmm, 1000)\nprobs, tot = HMMBase.backward(hmm, y)\n\n\n\n\n\n"
},

{
    "location": "inference/#HMMBase.posteriors",
    "page": "Inference",
    "title": "HMMBase.posteriors",
    "category": "function",
    "text": "posteriors(α, β)\n\nCompute posterior probabilities from α and β.\n\n\n\n\n\nposteriors(a, A, L)\n\nCompute posterior probabilities using samples likelihoods.\n\n\n\n\n\nposteriors(hmm, observations)\n\nCompute posterior probabilities using samples likelihoods.\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nz, y = rand(hmm, 1000)\nγ = posteriors(hmm, y)\n\n\n\n\n\n"
},

{
    "location": "inference/#HMMBase.forwardlog",
    "page": "Inference",
    "title": "HMMBase.forwardlog",
    "category": "function",
    "text": "HMMBase.forwardlog(a, A, LL)\n\nCompute HMMBase.forward probabilities using samples log-likelihoods. See HMMBase.forward.\n\n\n\n\n\nHMMBase.forwardlog(hmm, observations)\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nz, y = rand(hmm, 1000)\nprobs, tot = HMMBase.forwardlog(hmm, y)\n\n\n\n\n\n"
},

{
    "location": "inference/#HMMBase.backwardlog",
    "page": "Inference",
    "title": "HMMBase.backwardlog",
    "category": "function",
    "text": "HMMBase.backwardlog(a, A, LL)\n\nCompute HMMBase.backward probabilities using samples log-likelihoods. See HMMBase.backward.\n\n\n\n\n\nHMMBase.backwardlog(hmm, observations)\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nz, y = rand(hmm, 1000)\nprobs, tot = HMMBase.backwardlog(hmm, y)\n\n\n\n\n\n"
},

{
    "location": "inference/#HMMBase.posteriorslog",
    "page": "Inference",
    "title": "HMMBase.posteriorslog",
    "category": "function",
    "text": "posteriorslog(α, A, L)\n\nCompute posterior probabilities using samples log-likelihoods.\n\n\n\n\n\nposteriorslog(hmm, observations)\n\nCompute posterior probabilities using samples log-likelihoods.\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nz, y = rand(hmm, 1000)\nγ = posteriors(hmm, y)\n\n\n\n\n\n"
},

{
    "location": "inference/#Forward-Backward-1",
    "page": "Inference",
    "title": "Forward-Backward",
    "category": "section",
    "text": "forward\nbackward\nposteriors\nforwardlog\nbackwardlog\nposteriorslog"
},

{
    "location": "inference/#Baum–Welch-1",
    "page": "Inference",
    "title": "Baum–Welch",
    "category": "section",
    "text": "fit_mle"
},

{
    "location": "inference/#Viterbi-1",
    "page": "Inference",
    "title": "Viterbi",
    "category": "section",
    "text": "viterbi\nviterbilog"
},

{
    "location": "sampling/#",
    "page": "Sampling",
    "title": "Sampling",
    "category": "page",
    "text": ""
},

{
    "location": "sampling/#Base.rand",
    "page": "Sampling",
    "title": "Base.rand",
    "category": "function",
    "text": "rand(hmm::AbstractHMM, T::Int[, initial_state::Int])\n\nGenerate a random trajectory of hmm for T timesteps.\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\nz, y = rand(hmm, 1000)\n\n\n\n\n\nrand(hmm::AbstractHMM, z::AbstractVector{Int})\n\nGenerate observations from hmm according to trajectory z.\n\nExample\n\nhmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])\ny = rand(hmm, [1, 1, 2, 2, 1])\n\n\n\n\n\n"
},

{
    "location": "sampling/#Sampling-1",
    "page": "Sampling",
    "title": "Sampling",
    "category": "section",
    "text": "rand"
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
    "page": "Maximum Likelihood Estimation",
    "title": "Maximum Likelihood Estimation",
    "category": "page",
    "text": "EditURL = \"https://github.com/maxmouchet/HMMBase.jl/blob/master/examples/mle.jl\""
},

{
    "location": "examples/mle/#Maximum-Likelihood-Estimation-1",
    "page": "Maximum Likelihood Estimation",
    "title": "Maximum Likelihood Estimation",
    "category": "section",
    "text": "using Distributions\nusing HMMBase\nusing Plots\n\ny1 = rand(Normal(0,2), 1000)\ny2 = rand(Normal(10,1), 500)\ny = vcat(y1, y2, y1, y2)\n\nplot(y, linetype=:steppre, size=(600,200))For now HMMBase does not handle the initialization of the parameters. Hence we must instantiate an initial HMM by hand.hmm = HMM([0.5 0.5; 0.5 0.5], [Normal(-1,2), Normal(15,2)])\nhmm = fit_mle(hmm, y, verbose=true)z_viterbi = viterbi(hmm, y)\nplot(z_viterbi, linetype=:steppre, label=\"Viterbi decoded hidden state\", size=(600,200))This page was generated using Literate.jl."
},

{
    "location": "internals/#",
    "page": "Internals",
    "title": "Internals",
    "category": "page",
    "text": ""
},

{
    "location": "internals/#Internals-1",
    "page": "Internals",
    "title": "Internals",
    "category": "section",
    "text": ""
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
