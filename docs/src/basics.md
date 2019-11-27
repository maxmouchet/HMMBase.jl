# Basics

## [Common Options](@id common_options)

### Arguments

Name         |                 Type                 |   Default    |                     Description
:----------- | :----------------------------------- | :----------- | :--------------------------------------------------
a            | `AbstractVector`                     | -            | Initial probabilities vector
A            | `AbstractMatrix`                     | -            | Transition matrix
L            | `AbstractMatrix`                     | -            | (Log-)likelihoods
rng          | `AbstractRNG`                        | `GLOBAL_RNG` | Random number generator to use
hmm          | `AbstractHMM`                        | -            | HMM model
observations | `AbstractVector` or `AbstractMatrix` | -            | `T` or `T x dim(obs)`

### Keyword Arguments

Name   |  Type  | Default |                                             Description
:----- | :----- | :------ | :--------------------------------------------------------------------------------------------------
logl   | `Bool` | `false` | Use log-likelihoods instead of likelihoods, if set to true
robust | `Bool` | `false` | Truncate `-Inf` to `eps()` and `+Inf` to `prevfloat(Inf)` (`log(prevfloat(Inf))` in the log. case)


## Notations

Symbol |  Size  |             Description              |          Definition           
:----- | :----- | :----------------------------------- | :----------------------------
K      | -      | Number of states in an HMM           | _                            
T      | -      | Number of observations               | _                            
a      | K      | Initial state distribution           | $\sum_i a_i = 1$                            
A      | (K, K) | Transition matrix                    | $\sum_j A_{i,j} = 1, \forall i$                            
B      | K      | Vector of observation distributions  | _
z      | T      | Hidden states vector                 | $z_1 \sim a$, $z_t \sim A_{z_{t-1}\bullet}$
y      | (T, .) | Observations vector                  | $y_t \sim B_{z_t}$       
L      | (T, K) | Observations (log-)likelihoods       | $L(t,i) = p_{B_i}(y_t)$    
α      | (T, K) | Forward (filter) probabilities       | $\alpha(i) = \mathbb{P}(y_{1:t}, z_t = i)$
β      | (T, K) | Backward (smoothed) probabilities    | $\beta(i) = \mathbb{P}(y_{t+1:T} \,\|\, z_t = i)$
γ      | (T, K) | Posterior probabilities (α * β)      | $\gamma(i) = \mathbb{P}(z_t = i \,\|\, y_{1:T})$

**Before version 1.0:**

Symbol | Shape |             Description
:----- | :---- | :----------------------------------
π0     | K     | Initial state distribution
π      | KxK   | Transition matrix
D      | K     | Vector of observation distributions