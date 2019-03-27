# Implementations inspired by pyhsmm
# https://github.com/mattjj/pyhsmm/blob/master/pyhsmm/internals/hmm_states.py

# TODO: See softmax, implems : https://zenodo.org/record/1284341/files/main_pdf.pdf?download=1

@inline function normalize!(v::AbstractVector)
    norm = sum(v)
    v ./= norm
    norm
end

# ~2x times faster than Base.maximum
# v = rand(25)
# @btime maximum(v)
# @btime vec_maximum(v)
#   63.909 ns (1 allocation: 16 bytes)
#   30.307 ns (1 allocation: 16 bytes)
function vec_maximum(v::AbstractVector)
    m = v[1]
    @inbounds for i = Base.OneTo(length(v))
        if v[i] > m
            m = v[i]
        end
    end
    m
end

# Scaled implementations

"""
    messages_forwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)

Compute forward probabilities, see [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
"""
@views function messages_forwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
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
    messages_backwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)

Compute backward probabilities, see [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
"""
@views function messages_backwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
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
    forward_backward(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
"""
function forward_backward(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
    alphas, _ = messages_forwards(init_distn, trans_matrix, log_likelihoods)
    betas, _ = messages_backwards(init_distn, trans_matrix, log_likelihoods)
    gammas = alphas .* betas
    gammas ./ sum(gammas, dims=2)
end

# Convenience functions

"""
    messages_forwards(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
alphas, logtot = messages_forwards(hmm, y)
```
"""
function messages_forwards(hmm, observations)
    messages_forwards(hmm.π0, hmm.π, log_likelihoods(hmm, observations))
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
function messages_backwards(hmm, observations)
    messages_backwards(hmm.π0, hmm.π, log_likelihoods(hmm, observations))
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
function forward_backward(hmm, observations)
    forward_backward(hmm.π0, hmm.π, log_likelihoods(hmm, observations))
end
