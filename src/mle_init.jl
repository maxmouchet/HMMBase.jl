# Functions for initializing HMM parameters from the observations

function kmeans_init!(hmm::AbstractHMM, observations; kwargs...)
    K = size(hmm, 1)

    res = kmeans(permutedims(observations), size(hmm, 1); kwargs...)
    seq = res.assignments

    # Initialize A
    copyto!(hmm.A, gettransmat(seq)[2])
    @check istransmat(hmm.A)

    # Initialize B
    for i in OneTo(K)
        observations_ = view(observations, seq .== i, :)
        if length(observations_) > 0
            hmm.B[i] = fit_mle(typeof(hmm.B[i]), permutedims(observations_))
        end
    end
end
