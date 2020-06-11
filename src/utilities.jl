"""
    gettransmat(seq; relabel = false) -> (Dict, Matrix)

Return the transition matrix associated to the label sequence `seq`.  
The labels must be positive integer.  

**Arguments**
- `seq::Vector{<:Integer}`: positive label sequence.

**Keyword Arguments**
- `relabel::Bool = false`: if set to true the sequence
  will be made contiguous. E.g. `[7,7,9,9,1,1]` will become `[2,2,3,3,1,1]`.

**Output**
- `Dict{Integer,Integer}`: the mapping between the original and the new labels.
- `Matrix{Float64}`: the transition matrix.
"""
function gettransmat(seq::Vector{<:Integer}; relabel = false)
    @argcheck all(seq .>= 0)

    if relabel
        # /!\ Sort is important here, so that we don't relabel already contiguous states.
        mapping = Dict(old => new for (new, old) in enumerate(sort(unique(seq))))
    else
        mapping = Dict(old => old for old in unique(seq))
    end

    (length(mapping) == 0) && return mapping, Float64[]
    K = maximum(values(mapping))

    transmat = zeros(K, K)
    for i = 1:length(seq)-1
        transmat[mapping[seq[i]], mapping[seq[i+1]]] += 1
    end
    transmat = transmat ./ sum(transmat, dims = 2)
    transmat[isnan.(transmat)] .= 0.0

    mapping, transmat
end

"""
    randtransmat([rng,] prior) -> Matrix{Float64}

Generate a transition matrix where each row is sampled from `prior`.  
The prior must be a multivariate probability distribution, such as a
Dirichlet distribution.

**Arguments**
- `prior::MultivariateDistribution`: distribution over the transition matrix rows.

**Example**
```julia
A = randtransmat(Dirichlet([0.1, 0.1, 0.1]))
```
"""
function randtransmat(rng::AbstractRNG, prior::MultivariateDistribution)
    K = length(prior)
    A = Matrix{Float64}(undef, K, K)
    for i in OneTo(K)
        A[i, :] = rand(rng, prior)
    end
    @check istransmat(A)
    A
end

randtransmat(prior::MultivariateDistribution) = randtransmat(GLOBAL_RNG, prior)

"""
    randtransmat([rng, ]K, α = 1.0) -> Matrix{Float64}

Generate a transition matrix where each row is sampled from
a Dirichlet distribution of dimension `K` and concentration
parameter `α`.

**Arguments**
- `K::Integer`: number of states.
- `α::Float64 = 1.0`: concentration parameter of the Dirichlet distribution.

**Example**
```julia
A = randtransmat(4)
```
"""
randtransmat(rng::AbstractRNG, K::Integer, α = 1.0) = randtransmat(rng, Dirichlet(K, α))

randtransmat(K::Integer, args...) = randtransmat(GLOBAL_RNG, K, args...)

"""
    remapseq(seq, ref) -> Vector{Integer}

Find the permutations of `seq` indices that maximize the overlap with `ref`.

**Arguments**
- `seq::Vector{Integer}`: sequence to be remapped.
- `ref::Vector{Integer}`: reference sequence.

**Example**
```julia
ref = [1,1,2,2,3,3]
seq = [2,2,3,3,1,1]
remapseq(seq, ref)
# [1,1,2,2,3,3]
```
"""
function remapseq(seq::Vector{<:Integer}, ref::Vector{<:Integer})
    seqlabels, reflabels = unique(seq), unique(ref)
    @argcheck all(seqlabels .> 0) && all(reflabels .> 0)

    # C[i,j]: cost of assigning seq. label `i` to ref. label `j`
    C = zeros(maximum(seqlabels), maximum(reflabels))
    for i in seqlabels, j in reflabels
        C[i, j] = -sum((seq .== i) .& (ref .== j))
    end

    # TODO: Own implementation of the hungarian alg.,
    # to avoid pulling another dependency ?
    assignment, _ = hungarian(C)
    [assignment[x] for x in seq]
end


# ~2x times faster than Base.maximum
# v = rand(25)
# @btime maximum(v)
# @btime vec_maximum(v)
#   63.909 ns (1 allocation: 16 bytes)
#   30.307 ns (1 allocation: 16 bytes)
function vec_maximum(v::AbstractVector)
    m = v[1]
    @inbounds for i in OneTo(length(v))
        if v[i] > m
            m = v[i]
        end
    end
    m
end
