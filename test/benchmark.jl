using BenchmarkTools
using HMMBase
using StaticArrays

K, T = 10, 2500

init_distn = rand(K)
trans_matrix = rand(K,K)
log_likelihoods = rand(T,K)

bench = @benchmark messages_forwards(init_distn, trans_matrix, log_likelihoods)
show(stdout, "text/plain", bench)

bench = @benchmark messages_backwards(init_distn, trans_matrix, log_likelihoods)
show(stdout, "text/plain", bench)

bench = @benchmark forward_backward(init_distn, trans_matrix, log_likelihoods)
show(stdout, "text/plain", bench)

# init_distn = @SVector rand(K)
# trans_matrix = @SMatrix rand(K,K)
# log_likelihoods = @SMatrix rand(T,K)

# bench = @benchmark messages_forwards(init_distn, trans_matrix, log_likelihoods)
# show(stdout, "text/plain", bench)

# bench = @benchmark messages_backwards(init_distn, trans_matrix, log_likelihoods)
# show(stdout, "text/plain", bench)

# bench = @benchmark forward_backward(init_distn, trans_matrix, log_likelihoods)
# show(stdout, "text/plain", bench)