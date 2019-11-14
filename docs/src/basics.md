# Basics

## [Common Options](@id common_options)

### Arguments

- `a::AbstractVector`: initial probabilities vector.
- `A::AbstractMatrix`: transition matrix.
- `L::AbstractMatrix`: (log-)likelihoods.


- `hmm::AbstractHMM`: an HMM.
- `observations`: T or Txdim(obs)


- `rng::AbstractRNG` (`GLOBAL_RNG` by default): random number generator to use.

### Keyword Arguments

- `logl::Bool` (`false` by default): whether to use samples likelihoods, or log-likelihoods.
- `robust::Bool` (`false` by default): truncates `[-Inf, +Inf]` to `[eps(), prevfloat(Inf)]` or `[eps(), log(prevfloat(Inf))]` in the log case.

## Notations

Symbol | Shape |             Description
:----- | :---- | :-----------------------------------
K      | -     | Number of states in an HMM
T      | -     | Number of observations
a      | K     | Initial state distribution
A      | KxK   | Transition matrix
B      | K     | Vector of observations distributions
α      | TxK   | Forward filter
β      | TxK   | Backward filter
γ      | TxK   | Posteriors (α * β)

**Before version 1.0:**

Symbol | Shape |             Description
:----- | :---- | :----------------------------------
π0     | K     | Initial state distribution
π      | KxK   | Transition matrix
D      | K     | Vector of observation distributions