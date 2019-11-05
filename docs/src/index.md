# Home

*([View project on GitHub](https://github.com/maxmouchet/HMMBase.jl))*

HMMBase provides a lightweight and efficient abstraction for hidden Markov models in Julia. Most HMMs libraries only support discrete (e.g. categorical) or normal distributions. In contrast HMMBase builds upon [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) to support arbitrary univariate and multivariate distributions.  

The goal is to provide well-tested and fast implementations of the basic HMMs algorithms such as the forward-backward algorithm, the Viterbi algorithm, and the MLE estimator. More advanced models, such as Bayesian HMMs, can be built upon HMMBase.

## Getting Started

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add HMMBase
```

HMMBase supports any observations distributions implementing the `Distribution` interface from Distributions.jl.

```julia
using Distributions, HMMBase

# Univariate continuous observations
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Gamma(1,1)])

# Multivariate continuous observations
hmm = HMM([0.9 0.1; 0.1 0.9], [MvNormal([0.,0.],[1.,1.]), MvNormal([0.,0.],[1.,1.])])

# Univariate discrete observations
hmm = HMM([0.9 0.1; 0.1 0.9], [Categorical([0.3, 0.7]), Categorical([0.8, 0.2])])

# Multivariate discrete observations
hmm = HMM([0.9 0.1; 0.1 0.9], [Multinomial(10, [0.3, 0.7]), Multinomial(10, [0.8, 0.2])])
```

See the [Manual](@ref manual) section for more details on the models and algorithms, or jump directly to the [Examples](@ref examples).

*Logo: lego by jon trillana from the Noun Project.*
