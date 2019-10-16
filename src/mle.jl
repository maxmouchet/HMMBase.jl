"""
    mle_step(hmm::AbstractHMM{F}, observations) where F

Perform one step of the EM (Baum-Welch) algorithm.

# Example
```julia
hmm, log_likelihood = mle_step(hmm, observations)
```
"""
function mle_step(hmm::AbstractHMM{F}, observations) where F
    # NOTE: This function works but there is room for improvement.

    log_likelihoods = HMMBase.loglikelihoods(hmm, observations)

    log_α = forward_loglog(hmm.π0, hmm.π, log_likelihoods)
    log_β = backward_loglog(hmm.π, log_likelihoods)
    log_π = log.(hmm.π)

    normalizer = logsumexp(log_α[1,:] + log_β[1,:])

    # E-step

    T, K = size(log_likelihoods)
    log_ξ = zeros(T-1, K, K)

    @inbounds for t = 1:T-1, i = 1:K, j = 1:K
        log_ξ[t,i,j] = log_α[t,i] + log_π[i,j] + log_β[t+1,j] + log_likelihoods[t+1,j] - normalizer
    end

    ξ = exp.(log_ξ)
    ξ ./= sum(ξ, dims=[2,3])

    # M-step

    new_π = sum(ξ, dims=1)[1,:,:]
    new_π ./= sum(new_π, dims=2)

    new_π0 = exp.((log_α[1,:] + log_β[1,:]) .- normalizer)
    new_π0 ./= sum(new_π0)

    # TODO: Cleanup/optimize this part
    γ = exp.((log_α .+ log_β) .- normalizer)

    D = Distribution{F}[]
    for (i, d) in enumerate(hmm.D)
        # Super hacky...
        # https://github.com/JuliaStats/Distributions.jl/issues/809
        push!(D, fit_mle(eval(typeof(d).name.name), permutedims(observations), γ[:,i]))
    end

    typeof(hmm)(new_π0, new_π, D), normalizer
end

# function fit_mle(::Type{U}, y; initialization=) where U <: AbstractHMM
# TODO
# end

"""
    fit_mle!(hmm::AbstractHMM, observations; eps=1e-3, max_iterations=100, verbose=false)

Perform EM (Baum-Welch) steps until `max_iterations` is reached, or the change in the log-likelihood is smaller than `eps`.

# Example
```julia
hmm, log_likelihood = fit_mle!(hmm, observations)
```
"""
function fit_mle!(hmm::AbstractHMM, observations; eps=1e-3, max_iterations=100, verbose=false)
    new_hmm, last_norm = mle_step(hmm, observations)
    for i = 2:max_iterations
        new_hmm, norm = mle_step(new_hmm, observations)
        if abs(last_norm - norm) < eps
            verbose && println("Converged after $(i) iterations")
            break
        end
        last_norm = norm
    end
    new_hmm, last_norm
end
