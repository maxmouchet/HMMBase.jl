# Compute transition matrix

function compute_transition_matrix(seq::Vector{Int64})
    # /!\ Sort is important here, so that we don't relabel already contiguous states.
    mapping = Dict([(x[2], x[1]) for x in enumerate(sort(unique(seq)))])
    transmat = zeros(length(mapping), length(mapping))
    for i in 1:length(seq)-1
        transmat[mapping[seq[i]], mapping[seq[i+1]]] += 1
    end
    transmat = transmat ./ sum(transmat, dims=2)
    mapping, transmat
end

# Align sequences

# Generate random HMM of size K

# ...