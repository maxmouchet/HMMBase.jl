function viterbi(init_distn::Vector{Float64}, trans_matrix::Matrix{Float64}, likelihoods::Matrix{Float64})
    likelihoods = likelihoods' # Swap dims for better mem. perf ?
    K, T = size(likelihoods)
    
    T1 = zeros(K,T)
    T2 = zeros(Int,K,T)

    T1[:,1] = init_distn.*likelihoods[:,1]
    T1[:,1] /= sum(T1[:,1])

    @inbounds for t = 2:T
        for j in 1:K
            # TODO: Naming
            tmp = map(k -> T1[k,t-1]*trans_matrix[k,j]*likelihoods[j,t], 1:K)
            vmax, amax = findmax(tmp)
            T1[j,t] = vmax
            T2[j,t] = amax
        end
        T1[:,t] /= sum(T1[:,t])
    end

    z = zeros(Int,T)
    _, z[T] = findmax(T1[:,T])
    @inbounds for t in T:-1:2
        z[t-1] = T2[z[t],t]
    end
    
    z
end

function viterbi(trans_matrix::Matrix{Float64}, likelihoods::Matrix{Float64})
    init_distn = ones(size(trans_matrix)[1])/size(trans_matrix)[1]
    viterbi(init_distn, trans_matrix, likelihoods)
end

function viterbi(hmm::HMM, observations::Vector{T}) where T
    lls = hcat(map(d -> (pdf.(d, observations)), hmm.D)...)
    viterbi(hmm.π0, hmm.π, lls)
end
