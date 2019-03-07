# Implementations inspired by pyhsmm
# https://github.com/mattjj/pyhsmm/blob/master/pyhsmm/internals/hmm_states.py

# TODO: See softmax, implems : https://zenodo.org/record/1284341/files/main_pdf.pdf?download=1

@inline function normalize!(v::AbstractVector)
    norm = sum(v)
    v ./= norm
    norm
end

# Scaled implementations

@views function messages_forwards(init_distn::AbstractVector{Float64}, trans_matrix::AbstractMatrix{Float64}, log_likelihoods::AbstractMatrix{Float64})
    alphas = zeros(size(log_likelihoods))
    logtot = 0.0

    ll = log_likelihoods[1,:]
    c = maximum(ll)

    alpha = @. init_distn * exp(ll - c)
    norm = normalize!(alpha)

    alphas[1,:] = alpha
    logtot += c + log(norm)

    @inbounds for t = 2:size(alphas)[1]
        ll = log_likelihoods[t,:]
        c = maximum(ll)

        alpha .= trans_matrix' * alphas[t-1,:] .* exp.(ll .- c)
        norm = normalize!(alpha)
    
        alphas[t,:] = alpha
        logtot += c + log(norm)
    end

    alphas, logtot
end

@views function messages_backwards(init_distn::AbstractVector{Float64}, trans_matrix::AbstractMatrix{Float64}, log_likelihoods::AbstractMatrix{Float64})
    betas = zeros(size(log_likelihoods))
    betas[end,:] .= 1
    
    # Allows to reduce memory allocs. by T
    tmp = zeros(size(betas)[2])
    logtot = 0.0

    @inbounds for t = size(betas)[1]-1:-1:1
        ll = log_likelihoods[t+1,:]
        c = maximum(ll)

        tmp .= betas[t+1,:] .* exp.(ll .- c)
        beta = trans_matrix * tmp
        norm = normalize!(beta)

        betas[t,:] = beta
        logtot += c + log(norm)
    end

    ll = log_likelihoods[1,:]
    c = maximum(ll)
    logtot += c + log(sum(exp.(ll .- c) .* init_distn .* betas[1,:]))

    betas, logtot
end

function forward_backward(init_distn::AbstractVector{Float64}, trans_matrix::AbstractMatrix{Float64}, log_likelihoods::AbstractMatrix{Float64})
    alphas, _ = messages_forwards(init_distn, trans_matrix, log_likelihoods)
    betas, _ = messages_backwards(init_distn, trans_matrix, log_likelihoods)
    gammas = alphas .* betas
    gammas ./ sum(gammas, dims=2)
end

# Convenience functions

function messages_forwards(hmm, observations)
    messages_forwards(hmm.π0, hmm.π, log_likelihoods(hmm, observations))
end

function messages_backwards(hmm, observations)
    messages_backwards(hmm.π0, hmm.π, log_likelihoods(hmm, observations))
end

function forward_backward(hmm, observations)
    forward_backward(hmm.π0, hmm.π, log_likelihoods(hmm, observations))
end
