# Migrating to v1.0

Notations, and most notably:
- `messages_forwards` -> `forward/forwardlog`
- `messages_backward` -> `backward/backwardlog`
- `forward_backward` -> `posteriors/posteriorslog`

```julia
@deprecate log_likelihoods(hmm, observations) loglikelihoods(hmm, observations)

@deprecate forward_backward(init_distn, trans_matrix, log_likelihoods) posteriorslog(init_distn, trans_matrix, log_likelihoods)
@deprecate messages_forwards(init_distn, trans_matrix, log_likelihoods) forwardlog(init_distn, trans_matrix, log_likelihoods)
@deprecate messages_backwards(init_distn, trans_matrix, log_likelihoods) backwardlog(init_distn, trans_matrix, log_likelihoods)

@deprecate forward_backward(hmm, observations) posteriorslog(hmm, observations)
@deprecate messages_forwards(hmm, observations) forwardlog(hmm, observations)
@deprecate messages_backwards(hmm, observations) backwardlog(hmm, observations)

@deprecate messages_forwards_log(init_distn, trans_matrix, log_likelihoods) log.(forwardlog(init_distn, trans_matrix, log_likelihoods)[1])
@deprecate messages_backwards_log(trans_matrix, log_likelihoods) log.(backwardlog(init_distn, trans_matrix, log_likelihoods)[1])

@deprecate n_parameters(hmm) nparams(hmm)
@deprecate compute_transition_matrix(seq) gettransmat(seq, relabel = true)
@deprecate rand_transition_matrix(K, α = 1.0) randtransmat(K, α)
```

```julia
hmm, logtot = fit_mle!(hmm, observations, eps=1e-3) # Old
hmm = fit_mle(hmm, observations, tol=1e-3)          # New
```