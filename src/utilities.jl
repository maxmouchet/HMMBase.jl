function gettransmat(seq::Vector{Int64}; relabel = false)
    if relabel
        # /!\ Sort is important here, so that we don't relabel already contiguous states.
        mapping = Dict([(x[2], x[1]) for x in enumerate(sort(unique(seq)))])
    else
        mapping = Dict([(x, x) for x in unique(seq)])
    end

    transmat = zeros(length(mapping), length(mapping))
    for i in 1:length(seq)-1
        transmat[mapping[seq[i]], mapping[seq[i+1]]] += 1
    end
    transmat = transmat ./ sum(transmat, dims=2)

    mapping, transmat
end

function randtransmat(K::Integer, α = 1.0)
    prior = Dirichlet(K, α)
    A = Matrix{Float64}(undef, K, K)
    for i in OneTo(K)
        A[i,:] = rand(prior)
    end
    A
end


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
