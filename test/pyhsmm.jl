using Test
using Random
using PyCall

push!(LOAD_PATH, "src/")
using HMMBase

@pyimport pyhsmm

# TODO: Benchmarks

Random.seed!(42)

trans_matrix = rand(10,10)
trans_matrix = trans_matrix ./ sum(trans_matrix, dims=2)
init_dist = ones(10) / 10
log_likelihoods = rand(2500,10)

ref, _ = pyhsmm.internals[:hmm_states][:HMMStatesPython][:_messages_backwards_normalized](trans_matrix, init_dist, log_likelihoods)
res1 = messages_backward(trans_matrix, log_likelihoods)

@test sum(abs.(ref-res1)) < 1e-11

ref = pyhsmm.internals[:hmm_states][:HMMStatesPython][:_messages_backwards_log](trans_matrix, log_likelihoods)
res1 = messages_backward_log(trans_matrix, log_likelihoods)

@test sum(abs.(ref-res1)) < 1e-8

res1a = messages_forward(init_dist, trans_matrix, log_likelihoods)
res2a, res2b = pyhsmm.internals[:hmm_states][:HMMStatesPython][:_messages_forwards_normalized](trans_matrix, init_dist, log_likelihoods)

@test sum(abs.(res1a-res2a)) < 1e-11
# @test abs.(res1b-res2b) < 1e-11

res1 = messages_forward_log(init_dist, trans_matrix, log_likelihoods)
res2 = pyhsmm.internals[:hmm_states][:HMMStatesPython][:_messages_forwards_log](trans_matrix, init_dist, log_likelihoods)

@test sum(abs.(res1-res2)) < 1e-8
