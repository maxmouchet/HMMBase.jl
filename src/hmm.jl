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

"""
    HMM(π::Matrix{Float64}, D::Vector{<:Distribution{F}}) where F
    HMM(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution{F}}) where F

Build an HMM with transition matrix π and observations distributions D.  
If the initial state distribution π0 is not specified, a uniform distribution is assumed. 

Observations distributions can be of different types (for example `Normal` and `Exponential`).  
However they must be of the same dimension (all scalars or all multivariates).

# Examples
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
```
"""
function HMM(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution{F}}) where F
    assert_hmm(π, π0, D)
    HMM{F}(π, π0, D)
end

function HMM(π::Matrix{Float64}, D::Vector{<:Distribution{F}}) where F
    π0 = ones(size(π)[1]) / size(π)[1]
    assert_hmm(π, π0, D)
    HMM{F}(π, π0, D)
end

"""
    assert_hmm(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution})

Throw an `AssertionError` if the initial state distribution and the transition matrix rows does not sum to 1,
and if the observations distributions does not have the same dimensions.
"""
function assert_hmm(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution})
    # Initial state distribution and transition matrix rows must sum to 1
    @assert isprobvec(π0)
    @assert unique(mapslices(isprobvec, π, dims=2)) == [true]
    # All distributions must have the same dimensions
    @assert length(unique(map(length, D))) == 1
end

"""
    sample_hmm(hmm::HMM{Univariate}, timesteps::Int)
    sample_hmm(hmm::HMM{Multivariate}, timesteps::Int)

Sample a trajectory from `hmm`.  
Return a vector of observations for univariate HMMs and a matrix for multivariate HMMs.
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
