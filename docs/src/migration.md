# Migrating to v1.0

Notations, and most notably:
- `messages_forwards` -> `forward/forwardlog`
- `messages_backward` -> `backward/backwardlog`
- `forward_backward` -> `posteriors/posteriorslog`

See the bottom of [`HMMBase.jl`](src/HMMBase.jl), in addition:

```julia
hmm, logtot = fit_mle!(hmm, observations, eps=1e-3) # Old
hmm = fit_mle(hmm, observations, tol=1e-3)          # New
```