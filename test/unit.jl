using Distributions
using JSON
using HMMBase
using LinearAlgebra
using Test
using Random

using HMMBase: from_dict, issquare

Random.seed!(2019)

@testset "Base" begin
    hmm1 = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    hmm2 = HMM(
        [0.9 0.1; 0.1 0.9],
        [MvNormal([0.0, 0.0], [1.0, 1.0]), MvNormal([10.0, 10.0], [1.0, 1.0])],
    )

    @test size(hmm1) == (2, 1)
    @test size(hmm2) == (2, 2)

    @test nparams(hmm1) == 3
    @test nparams(hmm2) == 3
end

@testset "Base (2)" begin
    hmm1 = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    hmm2 = HMM(
        [0.9 0.1; 0.1 0.9],
        [MvNormal([0.0, 0.0], [1.0, 1.0]), MvNormal([10.0, 10.0], [1.0, 1.0])],
    )

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

@testset "Base (4)" begin
    hmm1 = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    hmm2 = HMM(
        [0.9 0.1; 0.1 0.9],
        [MvNormal([0.0, 0.0], [1.0, 1.0]), MvNormal([10.0, 10.0], [1.0, 1.0])],
    )
    hmm3 = HMM([0.9 0.1; 0.1 0.9], [MvNormal(ones(3)), MvNormal(ones(3))])

    # Univariate HMMs should return observations vectors
    # (consistent with Distributions.jl)
    z1, y1 = rand(hmm1, 1000, seq = true)
    y11 = rand(hmm1, z1)

    @test size(z1) == (1000,)
    @test size(y1) == (1000,)
    @test size(y11) == size(y1)

    # Multivariate HMMs should return a `TxK` matrix
    # (different from Distributions.jl which returns `KxT`)
    z2, y2 = rand(hmm2, 1000, seq = true)
    y22 = rand(hmm2, z2)

    @test size(z2) == (1000,)
    @test size(y2) == (1000, 2)
    @test size(y22) == size(y2)

    # Rand called with T < 1 should return empty arrays
    z3, y3 = rand(hmm2, 0, seq = true)
    y33 = rand(hmm2, z3)
    @test size(z3) == (0,)
    @test size(y3) == (0, 2)
    @test size(y33) == size(y3)

    # Multivariate HMM should work with more observations than states:
    # related to the issue: https://github.com/maxmouchet/HMMBase.jl/issues/12
    y = rand(hmm3, 1000)
    @test size(y) == (1000, 3)
end

@testset "Base (5)" begin
    # Emission matrix constructor
    hmm1 = HMM([0.9 0.1; 0.1 0.9], [0. 0.5 0.5; 0.25 0.25 0.5])
    hmm2 = HMM([0.9 0.1; 0.1 0.9], [Categorical([0., 0.5, 0.5]), Categorical([0.25, 0.25, 0.5])])
    @test hmm1 == hmm2
end

@testset "Constructors" begin
    # Test that errors are raised
    # Wrong trans. matrix
    @test_throws ArgumentError HMM(ones(2, 2), [Normal(); Normal()])
    # Wrong trans. matrix dimensions
    @test_throws ArgumentError HMM([0.8 0.1 0.1; 0.1 0.1 0.8], [Normal(0, 1), Normal(10, 1)])
    # Wrong number of distributions
    @test_throws ArgumentError HMM([0.8 0.2; 0.1 0.9], [Normal(0, 1), Normal(10, 1), Normal()])
    # Wrong distributions size
    @test_throws ArgumentError HMM([0.8 0.2; 0.1 0.9], [MvNormal(randn(3)), MvNormal(randn(10))])
    # Wrong initial state 
    @test_throws ArgumentError HMM([0.1; 0.1], [0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    # Wrong initial state length
    @test_throws ArgumentError HMM(
        [0.1; 0.1; 0.8],
        [0.9 0.1; 0.1 0.9],
        [Normal(0, 1), Normal(10, 1)],
    )
end

@testset "Stationnary Distributions" begin
    hmm1 = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    hmm2 = HMM([1.0 0.0; 0.0 1.0], [Normal(0, 1), Normal(10, 1)])
    hmm3 = HMM([0.0 0.8 0.2; 0.6 0.0 0.4; 0.0 1.0 0.0], [Normal(), Normal(), Normal()])

    dists1 = statdists(hmm1)
    dists2 = statdists(hmm2)
    dists3 = statdists(hmm3)

    @test length(dists1) == 1
    @test length(dists2) == 2
    @test length(dists3) == 1

    @test permutedims(dists2[1]) ≈ [1.0 0.0] * (hmm2.A^1000)
    @test permutedims(dists2[2]) ≈ [0.0 1.0] * (hmm2.A^1000)

    @test dists3[1] ≈ [15/53, 25/53, 13/53]
end

@testset "Messages (1)" begin
    # Example from https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
    A = [0.7 0.3; 0.3 0.7]
    B = [Categorical([0.9, 0.1]), Categorical([0.2, 0.8])]
    hmm = HMM(A, B)

    O = [1, 1, 2, 1, 1]

    α, logtot1 = forward(hmm, O)
    α = round.(α, digits = 4)

    β, logtot2 = backward(hmm, O)
    β = round.(β, digits = 4)

    γ = posteriors(hmm, O)
    γ = round.(γ, digits = 4)

    @test α == [
        0.8182 0.1818
        0.8834 0.1166
        0.1907 0.8093
        0.7308 0.2692
        0.8673 0.1327
    ]

    @test β == [
        0.5923 0.4077
        0.3763 0.6237
        0.6533 0.3467
        0.6273 0.3727
        1.0 1.0
    ]

    @test γ == [
        0.8673 0.1327
        0.8204 0.1796
        0.3075 0.6925
        0.8204 0.1796
        0.8673 0.1327
    ]

    @test logtot1 ≈ logtot2
end

@testset "Messages (2)" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    y = rand(hmm, 1000)

    α1, logtot1 = forward(hmm, y)
    α2, logtot2 = forward(hmm, y, logl = true)

    @test logtot1 ≈ logtot2
    @test α1 ≈ α2

    β1, logtot3 = backward(hmm, y)
    β2, logtot4 = backward(hmm, y, logl = true)

    @test logtot3 ≈ logtot4
    @test β1 ≈ β2

    @test size(α1) == size(α2) == size(β1) == size(β2)
    @test logtot1 ≈ logtot2 ≈ logtot3 ≈ logtot4

    γ1 = posteriors(hmm, y)
    γ2 = posteriors(hmm, y, logl = true)

    @test γ1 ≈ γ2
end

@testset "Messages (3)" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 0), Normal(10, 0)])
    y = rand(hmm, 1000)

    # The likelihood of a Normal distribution with std. = 0
    # equals either 0 or +Inf (-Inf, +Inf in log domain).
    # This cause the forward/backward algorithms to return NaNs,
    # We mark the tests as broken, since I don't know if this can be fixed.
    # The workaround is to set `robust = true`.
    _, logtot1 = forward(hmm, y)
    _, logtot2 = forward(hmm, y, logl = true)
    _, logtot3 = backward(hmm, y)
    _, logtot4 = backward(hmm, y, logl = true)

    @test_broken !isnan(logtot1)
    @test_broken !isnan(logtot2)
    @test_broken !isnan(logtot3)
    @test_broken !isnan(logtot4)

    @test_nowarn viterbi(hmm, y, logl = true)

    _, logtot5 = forward(hmm, y, robust = true)
    _, logtot6 = forward(hmm, y, logl = true, robust = true)
    _, logtot7 = backward(hmm, y, robust = true)
    _, logtot8 = backward(hmm, y, logl = true, robust = true)

    @test !isnan(logtot5)
    @test !isnan(logtot6)
    @test !isnan(logtot7)
    @test !isnan(logtot8)

    @test logtot5 ≈ logtot6 ≈ logtot7 ≈ logtot8

    @test viterbi(hmm, y, robust = true) == viterbi(hmm, y, logl = true, robust = true)
end

@testset "Viterbi" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(100, 1)])
    z, y = rand(hmm, 1000, seq = true)

    zv1 = viterbi(hmm, y)
    zv2 = viterbi(hmm, y, logl = true)

    @test zv1 == z
    @test zv1 == zv2
end

@testset "MLE" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    y = rand(hmm, 1000)

    # Likelihood should not decrease
    _, hist = fit_mle(hmm, y)
    @test issorted(round.(hist.logtots, digits = 9))

    _, hist = fit_mle(hmm, y, robust = true)
    @test issorted(round.(hist.logtots, digits = 9))

    _, hist = fit_mle(hmm, y, maxiter = 0)
    @test hist.iterations == 0
    @test !hist.converged
end

@testset "Utilities (1)" begin
    # Make sure that we do not relabel the states if they are in 1...K
    mapping, _ = gettransmat([3, 3, 1, 1, 2, 2], relabel = true)
    for (k, v) in mapping
        @test mapping[k] == v
    end

    mapping, transmat = gettransmat([3, 3, 8, 8, 3, 3], relabel = true)
    @test mapping[3] == 1
    @test mapping[8] == 2
    @test transmat == [2 / 3 1 / 3; 1 / 2 1 / 2]

    transmat = randtransmat(10)
    @test issquare(transmat)
    @test istransmat(transmat)
end

@testset "Utilities (2)" begin
    ref = [1, 1, 2, 2, 3, 3]
    seq1 = [2, 2, 3, 3, 1, 1]
    seq2 = [1, 1, 1, 1, 2, 2]

    @test remapseq(seq1, ref) == ref
    @test remapseq(seq2, ref) == [1, 1, 1, 1, 3, 3]
end

@testset "Reproducibility" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])

    z1, y1 = rand(MersenneTwister(0), hmm, 1000, seq = true)
    z2, y2 = rand(MersenneTwister(0), hmm, 1000, seq = true)
    z3, y3 = rand(hmm, 1000, seq = true)
    @test z1 == z2 != z3
    @test y1 == y2 != y3

    A1 = randtransmat(MersenneTwister(0), Dirichlet(4, 1.0))
    A2 = randtransmat(MersenneTwister(0), Dirichlet(4, 1.0))
    A3 = randtransmat(Dirichlet(4, 1.0))
    @test A1 == A2 != A3

    A4 = randtransmat(MersenneTwister(0), 4)
    A5 = randtransmat(MersenneTwister(0), 4)
    A6 = randtransmat(4)
    @test A4 == A5 != A6
end

@testset "Experimental" begin
    # from_dict(...)
    hmm1 = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    d = JSON.parse(json(hmm1))
    @test from_dict(HMM{Univariate,Float64}, Normal, d) == hmm1

    hmm2 = HMM(
        [0.9 0.1; 0.1 0.9],
        [MvNormal([0.0, 0.0], [1.0, 1.0]), MvNormal([10.0, 10.0], [1.0, 1.0])],
    )
    d = JSON.parse(json(hmm2))
    @test_broken from_dict(HMM{Multivariate,Float64}, MvNormal, d) == hmm2

    hmm3 = HMM([0.9 0.1; 0.1 0.9], [MixtureModel([Normal(0, 1)]), MixtureModel([Normal(5, 2), Normal(10, 1)], [0.25, 0.75])])
    d = JSON.parse(json(hmm3))
    @test_broken from_dict(HMM{Univariate,Float64}, MixtureModel{Univariate,Continuous,Normal,Float64}, d) == hmm3

    # MixtureModel <-> HMM (stationnary distribution)
    a = [0.4, 0.6]
    B = [Normal(0, 1), Exponential(2)]
    m = MixtureModel(B, a)

    @test MixtureModel(HMM(m)).prior == m.prior
    @test MixtureModel(HMM(m)).components == m.components

    # TODO: Assert error if #distns != 1
end
