# HMMBase.jl

*([View project on GitHub](https://github.com/maxmouchet/HMMBase.jl))*

A lightweight and efficient hidden Markov model abstraction for Julia.

```julia
# HMMBase supports any observations distributions implementing
# the `Distribution` interface from Distributions.jl.

# Univariate continuous observations
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Gamma(1,1)])

# Multivariate continuous observations
hmm = HMM([0.9 0.1; 0.1 0.9], [MvNormal([0.,0.],[1.,1.]), MvNormal([0.,0.],[1.,1.])])

# Univariate discrete observations
hmm = HMM([0.9 0.1; 0.1 0.9], [Categorical([0.3, 0.7]), Categorical([0.8, 0.2])])

# Multivariate discrete observations
hmm = HMM([0.9 0.1; 0.1 0.9], [Multinomial(10, [0.3, 0.7]), Multinomial(10, [0.8, 0.2])])

# Read the manual for more information.
```

*Logo: Blockchain by Pablo Rozenberg from the Noun Project.*