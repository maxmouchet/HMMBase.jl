# Algorithms

## [Common Options](@id common_options)

- `logl::Bool` (false by default): whether to use samples likelihoods, or log-likelihoods.
- `robust::Bool` (false by default): truncates [-Inf, +Inf] to [eps(), prevfloat(Inf)] or [eps(), log(prevfloat(Inf))] in the log. case.

## Forward-Backward

```@docs
forward
backward
posteriors
```

## Baum–Welch

```@docs
fit_mle
```

## Viterbi

```@docs
viterbi
```
