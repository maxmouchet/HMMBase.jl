"""
    AbstractHMM{F<:VariateForm}

A custom HMM type must at-least implement the following interface:
```julia
struct CustomHMM{F,T} <: AbstractHMM{F}
    a::AbstractVector{T}               # Initial state distribution
    A::AbstractMatrix{T}               # Transition matrix
    B::AbstractVector{Distribution{F}} # Observations distributions
    # Custom fields ....
end
```
"""
abstract type AbstractHMM{F<:VariateForm} end

"""
    HMM([a, ]A, B) -> HMM

Build an HMM with transition matrix `A` and observations distributions `B`.  
If the initial state distribution `a` is not specified, a uniform distribution is assumed. 

Observations distributions can be of different types (for example `Normal` and `Exponential`).  
However they must be of the same dimension (all scalars or all multivariates).

# Arguments
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractMatrix{T}`: transition matrix.
- `B::AbstractVector{<:Distribution{F}}`: observations distributions.

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
```
"""
struct HMM{F,T} <: AbstractHMM{F}
    a::Vector{T}
    A::Matrix{T}
    B::Vector{Distribution{F}}
    HMM{F,T}(a, A, B) where {F,T} = assert_hmm(a, A, B) && new(a, A, B) 
end

HMM(a::AbstractVector{T}, A::AbstractMatrix{T}, B::AbstractVector{<:Distribution{F}}) where {F,T} = HMM{F,T}(a, A, B)
HMM(A::AbstractMatrix{T}, B::AbstractVector{<:Distribution{F}}) where {F,T} = HMM{F,T}(ones(size(A)[1])/size(A)[1], A, B)

"""
    assert_hmm(a, A, B)

Throw an `ArgumentError` if the initial state distribution and the transition matrix rows does not sum to 1,
and if the observations distributions does not have the same dimensions.
"""
function assert_hmm(a::AbstractVector, 
                    A::AbstractMatrix, 
                    B::AbstractVector{<:Distribution})
    @argcheck isprobvec(a)
    @argcheck istransmat(A)
    @argcheck all(length.(B) .== length(B[1])) ArgumentError("All distributions must have the same dimensions")
    @argcheck length(a) == size(A,1) == length(B)
    return true
end

"""
    issquare(A) -> Bool

Return true if `A` is a square matrix.
"""
issquare(A::AbstractMatrix) = size(A,1) == size(A,2)

"""
    istransmat(A) -> Bool

Return true if `A` is square and its rows sums to 1.
"""
istransmat(A::AbstractMatrix) = issquare(A) && all([isprobvec(A[i,:]) for i in 1:size(A,1)])

==(h1::AbstractHMM, h2::AbstractHMM) = (h1.a == h2.a) && (h1.A == h2.A) && (h1.B == h2.B)

"""
    rand([rng, ]hmm, T; ...) -> Matrix | (Vector, Matrix)

Generate a random trajectory of `hmm` for `T` timesteps.

# Arguments
- `rng::AbstractRNG` (`GLOBAL_RNG` by default): random number generator to use.
- `init::Integer` (`rand(Categorical(hmm.a))` by default): initial state.
- `seq::Bool` (false by default): whether to return the hidden state sequence.

# Output
- `Vector{Int}`: hidden state sequence.
- `Matrix{Float64}` (if `seq` is true): observations (T x dim(obs)).

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000)
```

```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000, seq = true)
```
"""
function rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer; init = rand(rng, Categorical(hmm.a)), seq = false)
    z = Vector{Int}(undef, T)
    y = Matrix{Float64}(undef, T, size(hmm, 2))
    (T < 1) && return z, y

    z[1] = init
    y[1,:] = rand(rng, hmm.B[z[1]], 1)

    for t = 2:T
        z[t] = rand(rng, Categorical(hmm.A[z[t-1],:]))
        y[t,:] = rand(rng, hmm.B[z[t]], 1)
    end

    seq ? (z, y) : y
end

rand(hmm::AbstractHMM, T::Integer; kwargs...) = rand(GLOBAL_RNG, hmm, T; kwargs...)

"""
    rand([rng, ]hmm::AbstractHMM, z::AbstractVector{Int}) -> Matrix

Generate observations from `hmm` according to trajectory `z`.

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, [1, 1, 2, 2, 1])
```
"""
rand(rng::AbstractRNG, hmm::AbstractHMM, z::AbstractVector{Int}) = hcat(transpose(map(x -> rand(rng, hmm.B[x], 1), z))...)

rand(hmm::AbstractHMM, z::AbstractVector{Int}) = rand(GLOBAL_RNG, hmm, z)

"""
    size(hmm, [dim]) -> Int | Tuple

Returns the number of states in the HMM and the dimension of the observations.

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
size(hmm) # (2,1)
```
"""
size(hmm::AbstractHMM, dim=:) = (length(hmm.B), length(hmm.B[1]))[dim]

"""
    copy(hmm) -> HMM

Returns a copy of `hmm`.
"""
copy(hmm::HMM) = HMM(copy(hmm.a), copy(hmm.A), copy(hmm.B))

"""
    permute(hmm) -> HMM

Permute the states of `hmm` according to `perm`.

# Example
```julia
hmm = HMM([0.8 0.2; 0.1 0.9], [Normal(0,1), Normal(10,1)])
hmm = permute(hmm, [2, 1])
hmm.A # [0.9 0.1; 0.2 0.8]
hmm.B # [Normal(10,1), Normal(0,1)]
```
"""
function permute(hmm::HMM, perm::Vector{<:Integer})
    a = hmm.a[perm]
    B = hmm.B[perm]
    A = zeros(size(hmm.A))
    for i in 1:size(A,1), j in 1:size(A,2)
        A[i,j] = hmm.A[perm[i],perm[j]]
    end
    HMM(a, A, B)
end

"""
    nparams(hmm) -> Int

Return the number of _free_ parameters in `hmm`.

!!! warning
    Does not work, currently, for observations distributions with non-scalar parameters.

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
nparams(hmm) # 6
```
"""
function nparams(hmm::AbstractHMM)
    length(hmm.A) - size(hmm.A)[1] + sum(d -> length(params(d)), hmm.B)
end

"""
    statdists(hmm)

Return the stationnary distribution(s) of `hmm`.  
That is, the eigenvectors of transpose(hmm.A) with eigenvalues 1.
"""
function statdists(hmm::AbstractHMM)
    eig = eigen(collect(transpose(hmm.A)))
    dists = []
    for (i, val) in enumerate(eig.values)
        if val == 1.0
            dist = eig.vectors[:,i]
            dist /= sum(dist)
            push!(dists, dist)
        end
    end
    dists
end