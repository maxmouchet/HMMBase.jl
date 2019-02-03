# Implementations inspired by pyhsmm
# https://github.com/mattjj/pyhsmm/blob/master/pyhsmm/internals/hmm_states.py

# TODO: See softmax, implems : https://zenodo.org/record/1284341/files/main_pdf.pdf?download=1

# Scaled implementations

@views function messages_forwards(init_distn::AbstractVector{Float64}, trans_matrix::AbstractMatrix{Float64}, log_likelihoods::AbstractMatrix{Float64})
    alphas = zeros(size(log_likelihoods))
    logtot = 0.0

    ll = log_likelihoods[1,:]
    c = maximum(ll)

    alpha = init_distn .* exp.(ll .- c)
    norm = sum(alpha)

    alphas[1,:] = alpha / norm
    logtot += c + log(norm)

    @inbounds for t = 2:size(alphas)[1]
        ll = log_likelihoods[t,:]
        c = maximum(ll)

        alpha .= trans_matrix' * alphas[t-1,:] .* exp.(ll .- c)
        norm = sum(alpha)
    
        alphas[t,:] = alpha / norm
        logtot += c + log(norm)
    end

    alphas, logtot
end

@views function messages_backwards(init_distn::AbstractVector{Float64}, trans_matrix::AbstractMatrix{Float64}, log_likelihoods::AbstractMatrix{Float64})
    betas = zeros(size(log_likelihoods))
    betas[end,:] .= 1
    logtot = 0.0

    @inbounds for t = size(betas)[1]-1:-1:1
        ll = log_likelihoods[t+1,:]
        c = maximum(ll)

        beta = trans_matrix * (betas[t+1,:] .* exp.(ll .- c))
        norm = sum(beta)

        betas[t,:] = beta / norm
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

# Log implementations

function messages_forwards_log(init_distn::AbstractVector{Float64}, trans_matrix::AbstractMatrix{Float64}, log_likelihoods::AbstractMatrix{Float64})
    # OPTIMIZE
    log_alphas = zeros(size(log_likelihoods))
    log_trans_matrix = log.(trans_matrix)
    log_alphas[1,:] = log.(init_distn) .+ log_likelihoods[1,:]
    @inbounds for t = 2:size(log_alphas)[1]
        for i in 1:size(log_alphas)[2]
            # NOTE: log_trans_matrix[:,i] instead of log_trans_matrix.T[i,:]
            log_alphas[t,i] = logsumexp(log_alphas[t-1,:] .+ log_trans_matrix[:,i]) + log_likelihoods[t,i]
        end
    end
    log_alphas
end

function messages_backwards_log(trans_matrix::AbstractMatrix{Float64}, log_likelihoods::AbstractMatrix{Float64})
    # OPTIMIZE
    log_betas = zeros(size(log_likelihoods))
    log_trans_matrix = log.(trans_matrix)
    @inbounds for t = size(log_betas)[1]-1:-1:1
        tmp = view(log_betas, t+1, :) .+ view(log_likelihoods, t+1, :)
        @inbounds for i in 1:size(log_betas)[2]
            log_betas[t,i] = logsumexp(view(log_trans_matrix, i, :) .+ tmp)
        end
    end
    log_betas
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