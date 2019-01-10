# Implementations inspired by pyhsmm
# https://github.com/mattjj/pyhsmm/blob/master/pyhsmm/internals/hmm_states.py

# Scaled implementations

function messages_forward(init_distn::Vector{Float64}, trans_matrix::Matrix{Float64}, log_likelihoods::Matrix{Float64})
    alphas = zeros(size(log_likelihoods))

    ll = view(log_likelihoods, 1, :)
    c = maximum(ll)
    alpha = init_distn .* exp.(ll .- c)
    alphas[1,:] = alpha / sum(alpha)

    @inbounds for t = 2:size(alphas)[1]
        ll = view(log_likelihoods, t, :)
        c = maximum(ll)
        alpha = trans_matrix' * view(alphas, t-1, :) .* exp.(ll .- c)
        alphas[t,:] = alpha / sum(alpha)
    end

    alphas
end

function messages_backward(trans_matrix::Matrix{Float64}, log_likelihoods::Matrix{Float64})
    betas = zeros(size(log_likelihoods))
    betas[end,:] .= 1

    @inbounds for t = size(betas)[1]-1:-1:1
        ll = view(log_likelihoods, t+1, :)
        c = maximum(ll)
        beta = trans_matrix * (view(betas, t+1, :) .* exp.(ll .- c))
        betas[t,:] = beta / sum(beta);
    end

    betas
end

function forward_backward(init_distn::Vector{Float64}, trans_matrix::Matrix{Float64}, log_likelihoods::Matrix{Float64})
    alphas = messages_forward(init_distn, trans_matrix, log_likelihoods)
    betas = messages_backward(trans_matrix, log_likelihoods)
    gammas = alphas .* betas
    gammas ./ sum(gammas, dims=2)
end

# Log implementations

function messages_forward_log(init_distn::Vector{Float64}, trans_matrix::Matrix{Float64}, log_likelihoods::Matrix{Float64})
    log_alphas = zeros(size(log_likelihoods))
    log_trans_matrix = log.(trans_matrix)
    log_alphas[1,:] = log.(init_distn) .+ log_likelihoods[1,:]
    # OPTIMIZE
    @inbounds for t = 2:size(log_alphas)[1]
        for i in 1:size(log_alphas)[2]
            # NOTE: log_trans_matrix[:,i] instead of log_trans_matrix.T[i,:]
            log_alphas[t,i] = logsumexp(log_alphas[t-1,:] .+ log_trans_matrix[:,i]) + log_likelihoods[t,i]
        end
    end
    log_alphas
end

function messages_backward_log(trans_matrix::Matrix{Float64}, log_likelihoods::Matrix{Float64})
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

function messages_forward(hmm, observations)
    likelihoods = hcat(map(d -> pdf.(d, observations), hmm.D)...)
    messages_forward(hmm.π0, hmm.π, log.(likelihoods))
end

function messages_backward(hmm, observations)
    likelihoods = hcat(map(d -> pdf.(d, observations), hmm.D)...)
    messages_backward(hmm.π, log.(likelihoods))
end

function forward_backward(hmm, observations)
    likelihoods = hcat(map(d -> pdf.(d, observations), hmm.D)...)
    forward_backward(hmm.π0, hmm.π, log.(likelihoods))
end
