using Test
using Random
using PyCall
using HMMBase
using Distributions

np = pyimport("numpy")
pyhsmm = pyimport("pyhsmm")
pyhsmmi = pyimport("pyhsmm.internals.hmm_messages_interface")

Random.seed!(2019)

function rand_hmm(K)
    A = rand_transition_matrix(K)
    B = [Normal(rand()*100, rand()*10) for _ in 1:K]
    HMM(A, B)
end

@testset "Messages #$k" for k in 2:10
    hmm = rand_hmm(k)
    z, y = rand(hmm, 2500)
    LL = loglikelihoods(hmm, y)

    ref = pyhsmm.internals.hmm_states.HMMStatesPython._messages_forwards_normalized(hmm.A, hmm.a, LL)
    res = forwardlog(hmm.a, hmm.A, LL)

    @test sum(abs.(ref[1]-res[1])) < 1e-11
    @test abs(ref[2]-res[2]) < 1e-10

    ref = pyhsmm.internals.hmm_states.HMMStatesPython._messages_backwards_normalized(hmm.A, hmm.a, LL)
    res = backwardlog(hmm.a, hmm.A, LL)

    @test sum(abs.(ref[1]-res[1])) < 1e-11
    @test abs(ref[2]-res[2]) < 1e-10
end

@testset "Viterbi #$k" for k in 2:10
    hmm = rand_hmm(k)
    z, y = rand(hmm, 2500)

    L = likelihoods(hmm, y)
    LL = loglikelihoods(hmm, y)

    ref = pyhsmmi.viterbi(
        PyReverseDims(permutedims(hmm.A)),
        PyReverseDims(permutedims(LL)),
        hmm.a,
        zeros(Int32, size(LL,1))
    ) 
    
    res1 = viterbi(hmm.a, hmm.A, L)
    res2 = viterbilog(hmm.a, hmm.A, LL)

    # Python indices are off by 1
    @test res1 == (ref .+ 1)
    @test res2 == (ref .+ 1)
end
