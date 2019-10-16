log_likelihoods(hmm, observations) = loglikelihoods(hmm, observations)

messages_forwards_log(init_distn, trans_matrix, log_likelihoods) = forward_loglog(init_distn, trans_matrix, log_likelihoods)

messages_backwards_log(trans_matrix, log_likelihoods) = backward_loglog(trans_matrix, log_likelihoods)

"""
    messages_forwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)

Compute forward probabilities, see [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
"""
messages_forwards(init_distn, trans_matrix, log_likelihoods) = forward_log(init_distn, trans_matrix, log_likelihoods)

"""
    messages_backwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)

Compute backward probabilities, see [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
"""
messages_backwards(init_distn, trans_matrix, log_likelihoods) = backward_log(init_distn, trans_matrix, log_likelihoods)


"""
    forward_backward(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
"""
forward_backward(init_distn, trans_matrix, log_likelihoods) = posteriors_log(init_distn, trans_matrix, log_likelihoods)

"""
    messages_forwards(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
alphas, logtot = messages_forwards(hmm, y)
```
"""
messages_forwards(hmm, observations) = forward_log(hmm, observations)

"""
    messages_backwards(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
betas, logtot = messages_backwards(hmm, y)
```
"""
messages_backwards(hmm, observations) = backward_log(hmm, observations)

"""
    forward_backward(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
gammas = forward_backward(hmm, y)
```
"""
forward_backward(hmm, observations) = posteriors_log(hmm, observations)