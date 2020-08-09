# Functions for initializing HMM parameters from the observations
function kmeans_init!(hmm::AbstractHMM{Univariate}, observations; kwargs...)
    K = size(hmm, 1)
    N = last(size(observations))
    seq = Array{Union{Nothing, Int}}(nothing, size(observations))

    for n in OneTo(N)
        res = kmeans(filter(!isnothing, observations[:, n])', size(hmm, 1))
        T = length(filter(!isnothing, observations[:, n]))
        seq[1:T, n] .= res.assignments
    end

    # # Initialize A
    copyto!(hmm.A, gettransmat(seq)[2])
    @check istransmat(hmm.A)

    # # Initialize B
    for i in OneTo(K)
        y_ = vcat(view(observations, seq .== i)...)
        if length(y_) > 0
            hmm.B[i] = fit_mle(typeof(hmm.B[i]), permutedims(y_))
        end
    end
end

function kmeans_init!(hmm::AbstractHMM{Multivariate}, observations; kwargs...)
    K = size(hmm, 1)
    N = last(size(observations))
    seq = Array{Union{Nothing, Int}}(nothing, size(observations, 1), last(size(observations)))

    for n in OneTo(N)
        y_ = remove_nothing(observations[:, :, n])
        res = kmeans(y_', size(hmm, 1))
        T = size(y_, 1)
        seq[1:T, n] .= res.assignments
    end

    # # Initialize A
    copyto!(hmm.A, gettransmat(seq)[2])
    @check istransmat(hmm.A)

    # # Initialize B
    for i in OneTo(K)
        x1 = vcat(view(view(observations, :, 1, :), seq .== i)...)
        x2 = vcat(view(view(observations, :, 2, :), seq .== i)...)
        y_ = hcat(x1, x2)
        if length(y_) > 0
            hmm.B[i] = fit_mle(typeof(hmm.B[i]), permutedims(y_))
        end
    end
end