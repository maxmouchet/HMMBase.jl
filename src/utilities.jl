"""
    gettransmat(seq; relabel = false) -> (Dict, Matrix)

Return the transition matrix associated to the label sequence `seq`.  
The labels must be positive integer.  

# Arguments
- `relabel::Bool` (false by default): if set to true the sequence
  will be made contiguous. E.g. `[7,7,9,9,1,1]` will become `[2,2,3,3,1,1]`.

# Output
- `Dict{Integer,Integer}`: the mapping between the original and the new labels.
- `Matrix{Float64}`: the transition matrix.
"""
function gettransmat(seq::Vector{<:Integer}; relabel = false)
    @argcheck all(seq .>= 0)

    if relabel
        # /!\ Sort is important here, so that we don't relabel already contiguous states.
        mapping = Dict([(x[2], x[1]) for x in enumerate(sort(unique(seq)))])
    else
        mapping = Dict([(x, x) for x in unique(seq)])
    end

    K = maximum(values(mapping))

    transmat = zeros(K, K)
    for i in 1:length(seq)-1
        transmat[mapping[seq[i]], mapping[seq[i+1]]] += 1
    end
    transmat = transmat ./ sum(transmat, dims=2)
    transmat[isnan.(transmat)] .= 0.0

    mapping, transmat
end

"""
    randtransmat(prior) -> Matrix{Float64}

Generate a transition matrix where each row is sampled from `prior`.  
The prior must be a multivariate probability distribution, such as a
Dirichlet distribution.
"""
function randtransmat(prior)
    A = Matrix{Float64}(undef, K, K)
    for i in OneTo(K)
        A[i,:] = rand(prior)
    end
    A
end

"""
    randtransmat(K, α = 1.0) -> Matrix{Float64}

Generate a transition matrix where each row is sampled from
a Dirichlet distribution of dimension `K` and concentration
parameter `α`.
"""
randtransmat(K::Integer, α = 1.0) = randtransmat(Dirichlet(K, α))



# function rand(::HMM, K::Integer; A_prior = Dirichlet(K, 1.0), B_prior = )

# Align sequences

# Generate random HMM of size K

# ...

# ~2x times faster than Base.maximum
# v = rand(25)
# @btime maximum(v)
# @btime vec_maximum(v)
#   63.909 ns (1 allocation: 16 bytes)
#   30.307 ns (1 allocation: 16 bytes)
function vec_maximum(v::AbstractVector)
    m = v[1]
    @inbounds for i = OneTo(length(v))
        if v[i] > m
            m = v[i]
        end
    end
    m
end
