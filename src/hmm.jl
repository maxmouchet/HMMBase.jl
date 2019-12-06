"""
    AbstractHMM{F<:VariateForm}

A custom HMM type must at-least implement the following interface:
```julia
struct CustomHMM{F,T} <: AbstractHMM{F}
    a::AbstractVector{T}               # Initial state distribution
    A::AbstractMatrix{T}               # Transition matrix
    B::AbstractVector{Distribution{F}} # Observations distributions
    # Optional, custom, fields ....
end
```
"""
abstract type AbstractHMM{F<:VariateForm} end

"""
    HMM([a, ]A, B) -> HMM

Build an HMM with transition matrix `A` and observation distributions `B`.  
If the initial state distribution `a` is not specified, a uniform distribution is assumed. 

Observations distributions can be of different types (for example `Normal` and `Exponential`),  
but they must be of the same dimension.

**Arguments**
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractMatrix{T}`: transition matrix.
- `B::AbstractVector{<:Distribution{F}}`: observations distributions.

**Example**
```julia
using Distributions, HMMBase
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
and if the observation distributions do not have the same dimensions.
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
    rand([rng, ]hmm, T; init, seq) -> Array | (Vector, Array)

Sample a trajectory of `T` timesteps from `hmm`.

**Keyword Arguments**
- `init::Integer = rand(Categorical(hmm.a))`: initial state.
- `seq::Bool = false`: whether to return the hidden state sequence or not.

**Output**
- `Vector{Int}` (if `seq == true`): hidden state sequence.
- `Vector{Float64}` (for `Univariate` HMMs): observations (`T`).
- `Matrix{Float64}` (for `Multivariate` HMMs): observations (`T x dim(obs)`).

**Examples**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000) # or
z, y = rand(hmm, 1000, seq = true)
size(y) # (1000,)
```

```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [MvNormal(ones(2)), MvNormal(ones(2))])
y = rand(hmm, 1000) # or
z, y = rand(hmm, 1000, seq = true)
size(y) # (1000, 2)
```
"""
function rand(rng::AbstractRNG, hmm::AbstractHMM, T::Integer; init = rand(rng, Categorical(hmm.a)), seq = false)
    z = Vector{Int}(undef, T)
    (T >= 1) && (z[1] = init)
    for t in 2:T
        z[t] = rand(rng, Categorical(hmm.A[z[t-1],:]))
    end
    y = rand(rng, hmm, z)
    seq ? (z, y) : y
end

"""
    rand([rng, ]hmm, z) -> Array

Sample observations from `hmm` according to trajectory `z`.

**Output**
- `Vector{Float64}` (for `Univariate` HMMs): observations (`T`).
- `Matrix{Float64}` (for `Multivariate` HMMs): observations (`T x dim(obs)`).

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, [1, 1, 2, 2, 1])
```
"""
function rand(rng::AbstractRNG, hmm::AbstractHMM{Univariate}, z::AbstractVector{<:Integer})
    y = Vector{Float64}(undef, length(z))
    for t in eachindex(z)
        y[t] = rand(rng, hmm.B[z[t]])
    end
    y
end

function rand(rng::AbstractRNG, hmm::AbstractHMM{Multivariate}, z::AbstractVector{<:Integer})
    y = Matrix{Float64}(undef, length(z), size(hmm, 2))
    for t in eachindex(z)
        y[t,:] = rand(rng, hmm.B[z[t]])
    end
    y
end

rand(hmm::AbstractHMM, T::Integer; kwargs...) = rand(GLOBAL_RNG, hmm, T; kwargs...)

rand(hmm::AbstractHMM, z::AbstractVector{<:Integer}) = rand(GLOBAL_RNG, hmm, z)

"""
    size(hmm, [dim]) -> Int | Tuple

Return the number of states in `hmm` and the dimension of the observations.

**Example**
```jldoctest
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
size(hmm)
# output
(2, 1)
```
"""
size(hmm::AbstractHMM, dim=:) = (length(hmm.B), length(hmm.B[1]))[dim]

"""

    copy(hmm) -> HMM

Return a copy of `hmm`.
"""
copy(hmm::HMM) = HMM(copy(hmm.a), copy(hmm.A), copy(hmm.B))

"""
    permute(hmm, perm) -> HMM

Permute the states of `hmm` according to `perm`.

**Arguments**

- `perm::Vector{<:Integer}`: permutation of the states.

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.8 0.2; 0.1 0.9], [Normal(0,1), Normal(10,1)])
hmm = permute(hmm, [2, 1])
hmm.A # [0.9 0.1; 0.2 0.8]
hmm.B # [Normal(10,1), Normal(0,1)]
```
"""
function permute(hmm::AbstractHMM, perm::Vector{<:Integer})
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

Return the number of _free_ parameters in `hmm`, without counting the
observation distributions parameters.

**Example**
```jldoctest
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
nparams(hmm)
# output
3
```
"""
function nparams(hmm::AbstractHMM)
    (length(hmm.a) - 1) + (length(hmm.A) - size(hmm.A, 1))
end

"""
    statdists(hmm) -> Vector{Vector}

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