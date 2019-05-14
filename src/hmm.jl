"""
    AbstractHMM{F<:VariateForm}

An HMM type must at-least implement the following interface:
```julia
struct CustomHMM{F,T} <: AbstractHMM{F}
    π0::AbstractVector{T}              # Initial state distribution
    π::AbstractMatrix{T}               # Transition matrix
    D::AbstractVector{Distribution{F}} # Observations distributions
    # Custom fields ....
end
```
"""
abstract type AbstractHMM{F<:VariateForm} end

"""
    HMM([π0::AbstractVector{T}, ]π::AbstractMatrix{T}, D::AbstractVector{<:Distribution{F}}) where F where T

Build an HMM with transition matrix `π` and observations distributions `D`.  
If the initial state distribution `π0` is not specified, a uniform distribution is assumed. 

Observations distributions can be of different types (for example `Normal` and `Exponential`).  
However they must be of the same dimension (all scalars or all multivariates).

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
```
"""
struct HMM{F,T} <: AbstractHMM{F}
    π0::Vector{T}
    π::Matrix{T}
    D::Vector{Distribution{F}}
    HMM{F,T}(π0, π, D) where {F,T} = assert_hmm(π0, π, D) ? new(π0, π, D) : error()
end

HMM(π0::AbstractVector{T}, π::AbstractMatrix{T}, D::AbstractVector{<:Distribution{F}}) where {F,T} = HMM{F,T}(π0, π, D)
HMM(π::AbstractMatrix{T}, D::AbstractVector{<:Distribution{F}}) where {F,T} = HMM{F,T}(ones(size(π)[1])/size(π)[1], π, D)

"""
    StaticHMM([π0::AbstractVector{T}, ]π::AbstractMatrix{T}, D::AbstractVector{<:Distribution{F}}) where {F,T}

See [`HMM`](@ref).
"""
struct StaticHMM{F,T,K} <: AbstractHMM{F}
    π0::SVector{K,T}
    π::SMatrix{K,K,T}
    D::SVector{K,Distribution{F}}
    StaticHMM{F,T,K}(π0, π, D) where {F,T,K} = assert_hmm(π0, π, D) ? new(π0, π, D) : error()
end

StaticHMM(π0::AbstractVector{T}, π::AbstractMatrix{T}, D::AbstractVector{<:Distribution{F}}) where {F,T} = StaticHMM{F,T,size(π)[1]}(π0, π, D)
StaticHMM(π::AbstractMatrix{T}, D::AbstractVector{<:Distribution{F}}) where {F,T} = StaticHMM{F,T,size(π)[1]}(ones(size(π)[1])/size(π)[1], π, D)

"""
    assert_hmm(π0::AbstractVector{Float64}, π::AbstractMatrix{Float64}, D::AbstractVector{<:Distribution})

Throw an `AssertionError` if the initial state distribution and the transition matrix rows does not sum to 1,
and if the observations distributions does not have the same dimensions.
"""
function assert_hmm(π0::AbstractVector{Float64}, π::AbstractMatrix{Float64}, D::AbstractVector{<:Distribution})
    # Initial state distribution and transition matrix rows must sum to 1
    @assert isprobvec(π0)
    @assert unique([isprobvec(π[i,:]) for i in 1:size(π)[1]]) == [true]
    # All distributions must have the same dimensions
    @assert length(unique(map(length, D))) == 1
    # There must be one distribution per state
    @assert length(π0) == size(π)[1] == size(π)[2] == length(D)
    true
end

"""
    rand(hmm::AbstractHMM, T::Int[, initial_state::Int])

Generate a random trajectory of `hmm` for `T` timesteps.

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
```
"""
function rand(hmm::AbstractHMM, T::Int; initial_state=nothing)
    z = zeros(Int, T)
    y = zeros(T, length(hmm.D[1]))

    z[1] = initial_state == nothing ? rand(Categorical(hmm.π0)) : initial_state
    y[1,:] = rand(hmm.D[z[1]], 1)

    for t = 2:T
        z[t] = rand(Categorical(hmm.π[z[t-1],:]))
        y[t,:] = rand(hmm.D[z[t]], 1)
    end

    z, y
end

"""
    rand(hmm::AbstractHMM, z::AbstractVector{Int})

Generate observations from `hmm` according to trajectory `z`.

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, [1, 1, 2, 2, 1])
```
"""
rand(hmm::AbstractHMM, z::AbstractVector{Int}) = hcat(transpose(map(x -> rand(hmm.D[x], 1), z))...)

"""
    size(hmm::AbstractHMM)

Returns the number of states in the HMM and the dimension of the observations.

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
size(hmm) # (2,1)
```
"""
size(hmm::AbstractHMM) = (length(hmm.D), length(hmm.D[1]))

# TODO: Naming ?
function likelihoods(hmm::AbstractHMM{Univariate}, observations)
    hcat(map(d -> pdf.(d, observations), hmm.D)...)
end

function likelihoods(hmm::AbstractHMM{Multivariate}, observations)
    # OPTIMIZE ?
    ls = zeros(size(observations)[1], size(hmm)[1])
    @inbounds for i = 1:size(hmm)[1], t = 1:size(observations)[1]
        ls[t,i] = pdf(hmm.D[i], view(observations,t,:))
    end
    ls
end

# TODO: Naming ?
function log_likelihoods(hmm::AbstractHMM{Univariate}, observations)
    hcat(map(d -> logpdf.(d, observations), hmm.D)...)
end

function log_likelihoods(hmm::AbstractHMM{Multivariate}, observations)
    # OPTIMIZE ?
    lls = zeros(size(observations)[1], size(hmm)[1])
    @inbounds for i = 1:size(hmm)[1], t = 1:size(observations)[1]
        lls[t,i] = logpdf(hmm.D[i], view(observations,t,:))
    end
    lls
end

"""
    n_parameters(hmm::AbstractHMM)

Returns the number of parameters in `hmm`.  

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
n_parameters(hmm) # 6
```
"""
function n_parameters(hmm::AbstractHMM)
    length(hmm.π) - size(hmm.π)[1] + sum(d -> length(params(d)), hmm.D)
end