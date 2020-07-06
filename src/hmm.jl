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

Alternatively, `B` can be an emission matrix where `B[i,j]` is the probability of observing symbol `j` in state `i`.

**Arguments**
- `a::AbstractVector{T}`: initial probabilities vector.
- `A::AbstractMatrix{T}`: transition matrix.
- `B::AbstractVector{<:Distribution{F}}`: observations distributions.
- or `B::AbstractMatrix`: emission matrix.

**Example**
```julia
using Distributions, HMMBase
# from distributions
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
# from an emission matrix
hmm = HMM([0.9 0.1; 0.1 0.9], [0. 0.5 0.5; 0.25 0.25 0.5])
```
"""
struct HMM{F,T} <: AbstractHMM{F}
    a::Vector{T}
    A::Matrix{T}
    B::Vector{Distribution{F}}
    HMM{F,T}(a, A, B) where {F,T} = assert_hmm(a, A, B) && new(a, A, B)
end

HMM(a::AbstractVector{T}, A::AbstractMatrix{T}, B::AbstractVector{<:Distribution{F}}) where {F,T} =
    HMM{F,T}(a, A, B)

HMM(A::AbstractMatrix{T}, B::AbstractVector{<:Distribution{F}}) where {F,T} =
    HMM{F,T}(ones(size(A)[1]) / size(A)[1], A, B)

function HMM(a::AbstractVector{T}, A::AbstractMatrix{T}, B::AbstractMatrix) where {T}
    B = map(i -> Categorical(B[i,:]), 1:size(B,1))
    HMM{Univariate,T}(a, A, B)
end

HMM(A::AbstractMatrix{T}, B::AbstractMatrix) where {T} =
    HMM(ones(size(A)[1]) / size(A)[1], A, B)


"""
    assert_hmm(a, A, B)

Throw an `ArgumentError` if the initial state distribution and the transition matrix rows does not sum to 1,
and if the observation distributions do not have the same dimensions.
"""
function assert_hmm(a::AbstractVector, A::AbstractMatrix, B::AbstractVector{<:Distribution})
    @argcheck isprobvec(a)
    @argcheck istransmat(A)
    @argcheck all(length.(B) .== length(B[1])) ArgumentError("All distributions must have the same dimensions")
    @argcheck length(a) == size(A, 1) == length(B)
    return true
end

"""
    issquare(A) -> Bool

Return true if `A` is a square matrix.
"""
issquare(A::AbstractMatrix) = size(A, 1) == size(A, 2)

"""
    istransmat(A) -> Bool

Return true if `A` is square and its rows sums to 1.
"""
istransmat(A::AbstractMatrix) = issquare(A) && all([isprobvec(A[i, :]) for i = 1:size(A, 1)])

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
function rand(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    T::Integer;
    init = rand(rng, Categorical(hmm.a)),
    seq = false,
)
    z = Vector{Int}(undef, T)
    (T >= 1) && (z[1] = init)
    for t = 2:T
        z[t] = rand(rng, Categorical(hmm.A[z[t-1], :]))
    end
    y = rand(rng, hmm, z)
    seq ? (z, y) : y
end

"""
    rand([rng, ]hmm, T, N; init, seq) -> Array | (Vector, Array)

Sample a trajectory of `T` timesteps from `hmm`.

**Keyword Arguments**
- `init::Integer = rand(Categorical(hmm.a), N)`: initial state.
- `seq::Bool = false`: whether to return the hidden state sequence or not.

**Output**
- `Array{Int}` (if `seq == true`): hidden state sequence.
- `Array{Float64}` (for `Univariate` HMMs): observations (`T x N`).
- `Array{Float64}` (for `Multivariate` HMMs): observations (`T x dim(obs) x N`).

**Examples**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000) # or
z, y = rand(hmm, 1000, 2, seq = true)
size(y) # (1000, 2)
```

```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [MvNormal(ones(2)), MvNormal(ones(2))])
y = rand(hmm, 1000) # or
z, y = rand(hmm, 1000, 3, seq = true)
size(y) # (1000, 2, 3)
```
"""
function rand(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    T::Integer,
    N::Integer;
    init = rand(rng, Categorical(hmm.a), N),
    seq = false,
)
    z = Matrix{Int}(undef, T, N)
    (N == 1) && (z = vec(z))
    (T >= 1) && (z[1, :] = init)
    for n = 1:N
        for t = 2:T
            z[t, n] = rand(rng, Categorical(hmm.A[z[t-1, n], :]))
        end
    end
    y = rand(rng, hmm, z)
    seq ? (z, y) : y
end

"""
    rand([rng, ]hmm, z) -> Array

Sample observations from `hmm` according to trajectory `z`.

**Output**
- `Array{Float64}` (for `Univariate` HMMs): observations (`T x N`).
- `Array{Float64}` (for `Multivariate` HMMs): observations (`T x dim(obs) x N`).

**Examples**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, [1, 1, 2, 2, 1])
```
"""
function rand(
    rng::AbstractRNG,
    hmm::AbstractHMM{Univariate},
    z::AbstractArray{<:Integer},
    )
    T, N = size(z, 1), size(z, 2)
    y = Array{Float64}(undef, T, N)
    for n in 1:N
        for t in 1:T
            y[t, n] = rand(rng, hmm.B[z[t, n]])
        end
    end
    y
end

function rand(
    rng::AbstractRNG,
    hmm::AbstractHMM{Multivariate},
    z::AbstractArray{<:Integer},
    )
    T, N = size(z, 1), size(z, 2)
    y = Array{Float64}(undef, size(z, 1), size(hmm, 2), size(z, 2))
    for n in 1:N
        for t in 1:T
            y[t, :, n] = rand(rng, hmm.B[z[t, n]])
        end
    end
    y
end

"""
    rand([rng, ]hmm, d, N; seq) -> Array

Sample observations from `hmm` according to random trajectory sampled from `d`.

**Output**
- `Array{Float64}` (for `Univariate` HMMs): observations (`maximun(rand(d, N)) x N`).

**Examples**
```julia
using Distributions, HMMBase, Random
Random.seed!(1234)
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, Poisson(10), 2) # or
z, y = rand(hmm, Poisson(10), 2, seq = true)
size(y) #(12, 2)
```

```julia
using Distributions, HMMBase, Random
Random.seed!(1234)
hmm = HMM([0.9 0.1; 0.1 0.9], [MvNormal(ones(2)), MvNormal(ones(2))])
y = rand(hmm, Poisson(10), 3) # or
z, y = rand(hmm, Poisson(10), 3, seq = true)
size(y) #(10, 2, 3)
```
"""
function rand(
    rng::AbstractRNG,
    hmm::AbstractHMM{Univariate},
    d::DiscreteUnivariateDistribution,
    N::Integer;
    seq = false,
    )
    length_observations = generate_random_lengths(d, N)
    T = maximum(length_observations)
    z = Matrix{Union{Nothing, Int}}(nothing, T, N)
    y = Matrix{Union{Nothing, Float64}}(nothing, T, N)
    for n = 1:N
        z[1, n] = rand(rng, Categorical(hmm.a))
        y[1, n] = rand(rng, hmm.B[z[1, n]])
        for t = 2:T
            if t <= length_observations[n]
                z[t, n] = rand(rng, Categorical(hmm.A[z[t, n], :]))
                y[t, n] = rand(rng, hmm.B[z[t-1, n]])
            end
        end
    end
    seq ? (z, y) : y
end

function rand(
    rng::AbstractRNG,
    hmm::AbstractHMM{Multivariate},
    d::DiscreteUnivariateDistribution,
    N::Integer;
    seq = false,
    )
    length_observations = generate_random_lengths(d, N)
    T = maximum(length_observations)
    z = Matrix{Union{Nothing, Int}}(nothing, T, N)
    dimension = size(hmm, 2)
    y = Array{Union{Nothing, Float64}}(nothing, T, dimension, N)
    for n = 1:N
        z[1, n] = rand(rng, Categorical(hmm.a))
        y[1, :, n] = rand(rng, hmm.B[z[1, n]])
        for t = 2:T
            if t <= length_observations[n]
                z[t, n] = rand(rng, Categorical(hmm.A[z[t-1, n], :]))
                y[t, :, n] = rand(rng, hmm.B[z[t, n]])
            end
        end
    end
    seq ? (z, y) : y
end

rand(hmm::AbstractHMM, T::Integer; kwargs...) = rand(GLOBAL_RNG, hmm, T; kwargs...)

rand(hmm::AbstractHMM, T::Integer, N::Integer; kwargs...) =
    rand(GLOBAL_RNG, hmm, T, N; kwargs...)

rand(hmm::AbstractHMM, z::AbstractArray{<:Integer}) =
    rand(GLOBAL_RNG, hmm, size(z, 1), size(z, 2))

rand(hmm::AbstractHMM, d::DiscreteUnivariateDistribution, N::Integer; kwargs...) =
    rand(GLOBAL_RNG, hmm, d, N; kwargs...)

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
size(hmm::AbstractHMM, dim = :) = (length(hmm.B), length(hmm.B[1]))[dim]

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
    for i = 1:size(A, 1), j = 1:size(A, 2)
        A[i, j] = hmm.A[perm[i], perm[j]]
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
        if val ≈ 1.0
            dist = eig.vectors[:, i]
            dist /= sum(dist)
            push!(dists, dist)
        end
    end
    dists
end

function generate_random_lengths(
    d::DiscreteUnivariateDistribution,
    N::Integer
    )
    observations_length = rand(d, N)
end