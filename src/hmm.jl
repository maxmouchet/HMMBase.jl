struct HMM{F<:VariateForm}
    # Transition matrix
    π::Matrix{Float64}

    # Initial state distribution
    π0::Vector{Float64}

    # Observations distributions
    # Distributions can be differents but they must be of the same dimension
    # (all scalar or all multivariate)
    D::Vector{Distribution{F}}
end

function HMM(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution{F}}) where F
    assert_hmm(π, π0, D)
    HMM{F}(π, π0, D)
end

# HMM with a uniform initial state distribution
function HMM(π::Matrix{Float64}, D::Vector{<:Distribution{F}}) where F
    π0 = ones(size(π)[1]) / size(π)[1]
    assert_hmm(π, π0, D)
    HMM{F}(π, π0, D)
end

function assert_hmm(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution})
    # Initial state distribution and transition matrix rows must sum to 1
    @assert isprobvec(π0)
    @assert unique(mapslices(isprobvec, π, dims=2)) == [true]
    # All distributions must have the same dimensions
    @assert length(unique(map(length, D))) == 1
end

"""
    sample_hmm(hmm::HMM{Univariate}, timesteps::Int)

Sample a trajectory...
Returns a vector
"""
function sample_hmm(hmm::HMM{Univariate}, timesteps::Int)
    z = zeros(Int, timesteps)
    y = zeros(timesteps)

    z[1] = rand(Categorical(hmm.π0))
    y[1] = rand(hmm.D[z[1]])
    
    for t = 2:timesteps
        z[t] = rand(Categorical(hmm.π[z[t-1],:]))
        y[t] = rand(hmm.D[z[t]])
    end
    
    z, y
end

"""
    sample_hmm(hmm::HMM{Multivariate}, timesteps::Int)

Sample a trajectory...
Returns a matrix
"""
function sample_hmm(hmm::HMM{Multivariate}, timesteps::Int)
    z = zeros(Int, timesteps)
    y = zeros(timesteps, length(hmm.D[1]))

    z[1] = rand(Categorical(hmm.π0))
    y[1,:] = rand(hmm.D[z[1]])

    for t = 2:timesteps
        z[t] = rand(Categorical(hmm.π[z[t-1],:]))
        y[t,:] = rand(hmm.D[z[t]])
    end

    z, y
end

function compute_transition_matrix(seq::Vector{Int64})
    mapping = Dict([(x[2], x[1]) for x in enumerate(unique(seq))])
    transmat = zeros(length(mapping), length(mapping))
    for i in 1:length(seq)-1
        transmat[mapping[seq[i]], mapping[seq[i+1]]] += 1
    end
    transmat = transmat ./ sum(transmat, dims=2)
    mapping, transmat
end
