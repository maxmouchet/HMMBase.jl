# logsumexp implementations inspired by pyhsmm
# https://github.com/mattjj/pyhsmm/blob/master/pyhsmm/internals/hmm_states.py

"""
    forward_logsumexp(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)

Compute forward probabilities, see [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
"""
@views function forward_logsumexp(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
    alphas = zeros(size(log_likelihoods))
    logtot = 0.0

    ll = log_likelihoods[1,:]
    c = vec_maximum(ll)

    alpha = @. init_distn * exp(ll - c)
    norm = normalize!(alpha)

    alphas[1,:] = alpha
    logtot += c + log(norm)

    @inbounds for t = 2:size(alphas)[1]
        ll = log_likelihoods[t,:]
        c = vec_maximum(ll)

        # Cut down allocations by T, instead of *
        mul!(alpha, transpose(trans_matrix), alphas[t-1,:])
        alpha .= @. alpha * exp(ll - c)
        norm = normalize!(alpha)
    
        alphas[t,:] = alpha
        logtot += c + log(norm)
    end

    alphas, logtot
end

"""
    backward_logsumexp(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)

Compute backward probabilities, see [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
"""
@views function backward_logsumexp(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
    betas = zeros(size(log_likelihoods))
    betas[end,:] .= 1
    
    # Allows to reduce memory allocs. by T
    beta = zeros(size(betas)[2])
    tmp = zeros(size(betas)[2])
    logtot = 0.0

    @inbounds for t = size(betas)[1]-1:-1:1
        ll = log_likelihoods[t+1,:]
        c = vec_maximum(ll)

        tmp .= betas[t+1,:] .* exp.(ll .- c)
        mul!(beta, trans_matrix, tmp)
        norm = normalize!(beta)

        betas[t,:] = beta
        logtot += c + log(norm)
    end

    ll = log_likelihoods[1,:]
    c = vec_maximum(ll)
    logtot += c + log(sum(exp.(ll .- c) .* init_distn .* betas[1,:]))

    betas, logtot
end

"""
    messages_forwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)

Compute forward probabilities, see [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
"""
messages_forwards(init_distn, trans_matrix, log_likelihoods) = forward_logsumexp(init_distn, trans_matrix, log_likelihoods)
# Deprecated

"""
    messages_backwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)

Compute backward probabilities, see [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
"""
messages_backwards(init_distn, trans_matrix, log_likelihoods) = backward_logsumexp(init_distn, trans_matrix, log_likelihoods)
# Deprecated

"""
    posteriors(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
"""
function posteriors_logsumexp(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
    alphas, _ = forward_logsumexp(init_distn, trans_matrix, log_likelihoods)
    betas, _ = backward_logsumexp(init_distn, trans_matrix, log_likelihoods)
    gammas = alphas .* betas
    gammas ./ sum(gammas, dims=2)
end

"""
    forward_backward(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
"""
forward_backward(init_distn, trans_matrix, log_likelihoods)
# Deprecated

# Convenience functions

"""
    forward_logsumexp(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
alphas, logtot = forward_logsumexp(hmm, y)
```
"""
function forward_logsumexp(hmm, observations)
    forward_logsumexp(hmm.π0, hmm.π, loglikelihoods(hmm, observations))
end

"""
    messages_forwards(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
alphas, logtot = messages_forwards(hmm, y)
```
"""
messages_forwards(hmm, observations) = forward_logsumexp(hmm, observations)
# Deprecated

"""
    backward_logsumexp(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
betas, logtot = backward_logsumexp(hmm, y)
```
"""
function backward_logsumexp(hmm, observations)
    backward_logsumexp(hmm.π0, hmm.π, loglikelihoods(hmm, observations))
end

"""
    messages_backwards(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
betas, logtot = messages_backwards(hmm, y)
```
"""
messages_backwards(hmm, observations) = backward_logsumexp(hmm, observations)
# Deprecated

"""
    posteriors_logsumexp(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
gammas = posteriors_logsumexp(hmm, y)
```
"""
function posteriors_logsumexp(hmm, observations)
    posteriors_logsumexp(hmm.π0, hmm.π, loglikelihoods(hmm, observations))
end

"""
    forward_backward(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
gammas = forward_backward(hmm, y)
```
"""
forward_backward(hmm, observations) = posteriors_logsumexp(hmm, observations)
# Deprecated