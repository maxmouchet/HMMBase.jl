#"""
#    viterbi(init_distn::AbstractVector, trans_matrix::AbstractMatrix, likelihoods::AbstractMatrix)
#
#Find the most likely hidden state sequence, see [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm).
#"""
#function viterbi(init_distn::AbstractVector, trans_matrix::AbstractMatrix, likelihoods::AbstractMatrix)
#    # TODO: Benchmark/optimize and cleanup code
#    likelihoods = likelihoods' # Swap dims for better mem. perf ?
#    K, T = size(likelihoods)
#    
#    T1 = zeros(K,T)
#    T2 = zeros(Int,K,T)
#
#    T1[:,1] = init_distn.*likelihoods[:,1]
#    T1[:,1] /= sum(T1[:,1])
#
#    @inbounds for t = 2:T
#        for j in 1:K
#            amax = 0
#            vmax = -Inf
#            for k in 1:K
#                v = T1[k,t-1]*trans_matrix[k,j]*likelihoods[j,t]
#                if v > vmax
#                    amax = k
#                    vmax = v
#                end
#            end
#            T1[j,t] = vmax
#            T2[j,t] = amax
#        end
#        T1[:,t] /= sum(T1[:,t])
#    end
#
#    z = zeros(Int,T)
#    _, z[T] = findmax(T1[:,T])
#    @inbounds for t in T:-1:2
#        z[t-1] = T2[z[t],t]
#    end
#    
#    z
#end
#
#"""
#    viterbi(trans_matrix::AbstractMatrix, likelihoods::AbstractMatrix)
#
#Assume an uniform initial distribution.
#"""
#function viterbi(trans_matrix::AbstractMatrix, likelihoods::AbstractMatrix)
#    init_distn = ones(size(trans_matrix)[1])/size(trans_matrix)[1]
#    viterbi(init_distn, trans_matrix, likelihoods)
#end
#
#"""
#    viterbi(hmm::HMM, observations::Vector)
#
## Example
#```julia
#hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)]);
#z, y = rand(hmm, 1000);
#z_viterbi = viterbi(hmm, y)
#z == z_viterbi
#```
#"""
#function viterbi(hmm::AbstractHMM, observations)
#    viterbi(hmm.π0, hmm.π, likelihoods(hmm, observations))
#end
