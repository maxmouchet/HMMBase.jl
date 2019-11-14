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
    A = randtransmat(K)
    B = [Normal(rand()*100, rand()*10) for _ in 1:K]
    HMM(A, B)
end

@testset "Messages #$k" for k in 2:10
    hmm = rand_hmm(k)
    y = rand(hmm, 2500)
    LL = likelihoods(hmm, y, logl = true)

    ref = pyhsmm.internals.hmm_states.HMMStatesPython._messages_forwards_normalized(hmm.A, hmm.a, LL)
    res = forward(hmm.a, hmm.A, LL, logl = true)

    @test all(res[1] .≈ ref[1])
    @test res[2] ≈ ref[2]

    ref = pyhsmm.internals.hmm_states.HMMStatesPython._messages_backwards_normalized(hmm.A, hmm.a, LL)
    res = backward(hmm.a, hmm.A, LL, logl = true)

    @test all(res[1] .≈ ref[1])
    @test res[2] ≈ ref[2]
end

@testset "Viterbi #$k" for k in 2:10
    hmm = rand_hmm(k)
    y = rand(hmm, 2500)

    L = likelihoods(hmm, y)
    LL = likelihoods(hmm, y, logl = true)

    ref = pyhsmmi.viterbi(
        PyReverseDims(permutedims(hmm.A)),
        PyReverseDims(permutedims(LL)),
        hmm.a,
        zeros(Int32, size(LL,1))
    ) 
    
    res1 = viterbi(hmm.a, hmm.A, L)
    res2 = viterbi(hmm.a, hmm.A, LL, logl = true)

    # Python indices are off by 1
    @test res1 == (ref .+ 1)
    @test res2 == (ref .+ 1)
end
