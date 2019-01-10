```@meta
EditURL = "https://github.com/TRAVIS_REPO_SLUG/blob/master/"
```

# HMM with discrete observations

```@example discrete_obs
using Distributions
using HMMBase
using Plots

π = [0.9 0.1; 0.2 0.8]
D = [Categorical([0.9, 0.1]), Categorical([0.2, 0.8])]
hmm = HMM(π, D)

z, y = sample_hmm(hmm, 250)
plot(y)#-
```

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

