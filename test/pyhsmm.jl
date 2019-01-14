using Test
using Random
using PyCall
using HMMBase

@pyimport pyhsmm

# TODO: Benchmarks

Random.seed!(42)

trans_matrix = rand(10,10)
trans_matrix = trans_matrix ./ sum(trans_matrix, dims=2)
init_distn = ones(10) / 10
log_likelihoods = rand(2500,10)

# Scaled
ref = pyhsmm.internals[:hmm_states][:HMMStatesPython][:_messages_forwards_normalized](trans_matrix, init_distn, log_likelihoods)
res = messages_forwards(init_distn, trans_matrix, log_likelihoods)

@test sum(abs.(ref[1]-res[1])) < 1e-11
@test sum(abs.(ref[2]-res[2])) < 1e-11

ref = pyhsmm.internals[:hmm_states][:HMMStatesPython][:_messages_backwards_normalized](trans_matrix, init_distn, log_likelihoods)
res = messages_backwards(init_distn, trans_matrix, log_likelihoods)

@test sum(abs.(ref[1]-res[1])) < 1e-11
@test sum(abs.(ref[2]-res[2])) < 1e-11

# Log
ref = pyhsmm.internals[:hmm_states][:HMMStatesPython][:_messages_forwards_log](trans_matrix, init_distn, log_likelihoods)
res = messages_forwards_log(init_distn, trans_matrix, log_likelihoods)

@test sum(abs.(ref-res)) < 1e-8

ref = pyhsmm.internals[:hmm_states][:HMMStatesPython][:_messages_backwards_log](trans_matrix, log_likelihoods)
res = messages_backwards_log(trans_matrix, log_likelihoods)

@test sum(abs.(ref-res)) < 1e-8