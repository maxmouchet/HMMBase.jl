using Test
using HMMBase
using Distributions
using LinearAlgebra
using Random

import HMMBase: issquare

Random.seed!(2019)

@testset "Base" begin
    hmm1 = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    hmm2 = HMM([0.9 0.1; 0.1 0.9], [MvNormal([0.0,0.0], [1.0,1.0]), MvNormal([10.0,10.0], [1.0,1.0])])

    @test size(hmm1) == (2, 1)
    @test size(hmm2) == (2, 2)

    @test nparams(hmm1) == 6
    # 2 free parameters for the transition matrix, 2x4 for the covariance matrices,
    # and 2x2 for the means vectors.
    @test_broken nparams(hmm2) == 14    
end

@testset "Base (2)" begin
    hmm1 = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    hmm2 = HMM([0.9 0.1; 0.1 0.9], [MvNormal([0.0,0.0], [1.0,1.0]), MvNormal([10.0,10.0], [1.0,1.0])])

    @test issquare(hmm1.A)
    @test istransmat(hmm1.A)

    @test hmm1 != hmm2 
    @test hmm1 == copy(hmm1)
    @test hmm1 !== copy(hmm1) # !== (identity) not != (equality)
end

@testset "Base (3)" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])

    perm = [1, 2]
    hmmp = permute(hmm, perm)
    @test hmmp == hmm

    perm = [2, 1]
    hmmp = permute(hmm, perm)
    @test hmmp != hmm
    @test hmmp.B == hmm.B[perm]
    @test diag(hmmp.A) == diag(hmm.A)[perm]
end

@testset "Constructors" begin
    # Test that errors are raised
    # Wrong trans. matrix
    @test_throws ArgumentError HMM(ones(2, 2), [Normal();Normal()])
    # Wrong trans. matrix dimensions
    @test_throws ArgumentError HMM([0.8 0.1 0.1; 0.1 0.1 0.8], [Normal(0, 1), Normal(10, 1)])
    # Wrong number of distributions
    @test_throws ArgumentError HMM([0.8 0.2; 0.1 0.9], [Normal(0, 1), Normal(10, 1), Normal()])
    # Wrong distributions size
    @test_throws ArgumentError HMM([0.8 0.2; 0.1 0.9], [MvNormal(randn(3)), MvNormal(randn(10))])
    # Wrong initial state 
    @test_throws ArgumentError HMM([0.1;0.1], [0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    # Wrong initial state length
    @test_throws ArgumentError HMM([0.1;0.1;0.8], [0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
end

@testset "Stationnary Distributions" begin
    hmm1 = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    hmm2 = HMM([1.0 0.0; 0.0 1.0], [Normal(0, 1), Normal(10, 1)])

    dists1 = statdists(hmm1)
    dists2 = statdists(hmm2)

    @test length(dists1) == 1
    @test length(dists2) == 2

    @test permutedims(dists2[1]) ≈ [1.0 0.0] * (hmm2.A^1000)
    @test permutedims(dists2[2]) ≈ [0.0 1.0] * (hmm2.A^1000)
end

@testset "Messages (1)" begin
    # Example from https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
    A = [0.7 0.3; 0.3 0.7]
    B = [Categorical([0.9, 0.1]), Categorical([0.2, 0.8])]
    hmm = HMM(A, B)

    O = [1,1,2,1,1]

    α, logtot1 = forward(hmm, O)
    α = round.(α, digits = 4)

    β, logtot2 = backward(hmm, O)
    β = round.(β, digits = 4)

    γ = posteriors(hmm, O)
    γ = round.(γ, digits = 4)

    @test α == [
        0.8182 0.1818;
        0.8834 0.1166;
        0.1907 0.8093;
        0.7308 0.2692;
        0.8673 0.1327;
    ]

    @test β == [
        0.5923 0.4077;
        0.3763 0.6237;
        0.6533 0.3467;
        0.6273 0.3727;
        1.0    1.0;
    ]

    @test γ == [
        0.8673 0.1327;
        0.8204 0.1796;
        0.3075 0.6925;
        0.8204 0.1796;
        0.8673 0.1327;
    ]

    @test logtot1 ≈ logtot2
end

@testset "Messages (2)" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    z, y = rand(hmm, 1000)

    α1, logtot1 = forward(hmm, y)
    α2, logtot2 = forwardlog(hmm, y)

    @test logtot1 ≈ logtot2
    @test α1 ≈ α2

    β1, logtot3 = backward(hmm, y)
    β2, logtot4 = backwardlog(hmm, y)

    @test logtot3 ≈ logtot4
    @test β1 ≈ β2

    @test size(α1) == size(α2) == size(β1) == size(β2)
    @test logtot1 ≈ logtot2 ≈ logtot3 ≈ logtot4

    γ1 = posteriors(hmm, y)
    γ2 = posteriorslog(hmm, y)

    @test γ1 ≈ γ2
end

@testset "Viterbi" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(100, 1)])
    z, y = rand(hmm, 1000)

    zv1 = viterbi(hmm, y)
    zv2 = viterbilog(hmm, y)

    @test zv1 == z
    @test zv1 == zv2
end

@testset "MLE" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    z, y = rand(hmm, 1000)

    # Likelihood should not decrease
    hmmp = fit_mle(hmm, y, display = :final)
    @test forward(hmmp, y)[2] >= forward(hmm, y)[2]
end

@testset "Utilities" begin
    # Make sure that we do not relabel the states if they are in 1...K
    mapping, _  = gettransmat([3,3,1,1,2,2], relabel = true)
    for (k, v) in mapping
        @test mapping[k] == v
    end

    mapping, transmat = gettransmat([3,3,8,8,3,3], relabel = true)
    @test mapping[3] == 1
    @test mapping[8] == 2
    @test transmat == [2 / 3 1 / 3; 1 / 2 1 / 2]

    transmat = randtransmat(10)
    @test issquare(transmat)
    @test istransmat(transmat)
end