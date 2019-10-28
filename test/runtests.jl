using Test
using HMMBase
using Distributions
using LinearAlgebra
using Random

Random.seed!(2018)

function rand_hmm(K)
    A = rand_transition_matrix(K)
    B = [Normal(rand()*100, rand()*10) for _ in 1:K]
    HMM(A, B)
end

targets = [
    HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)]),
    HMM([0.9 0.1; 0.1 0.9], [MvNormal([0.0,0.0], [1.0,1.0]), MvNormal([10.0,10.0], [1.0,1.0])]),
]

@testset "Constructors" begin
    # Test error are raised
    # wrong Tras Matrix
    @test_throws ArgumentError HMM(ones(2,2), [Normal();Normal()])
    # wrong Tras Matrix dimensions
    @test_throws ArgumentError HMM([0.8 0.1 0.1; 0.1 0.1 0.8], [Normal(0,1), Normal(10,1)])
    # wrong number of Distributions
    @test_throws ArgumentError HMM([0.8 0.2; 0.1 0.9], [Normal(0,1), Normal(10,1), Normal()])
    # wrong distribution size
    @test_throws ArgumentError HMM([0.8 0.2; 0.1 0.9], [MvNormal(randn(3)), MvNormal(randn(10))])
    # wrong initial state 
    @test_throws ArgumentError HMM([0.1;0.1],[0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
    # wrong initial state length
    @test_throws ArgumentError HMM([0.1;0.1;0.8],[0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
end

@testset "Random" begin
    # Test random observations generation with a fixed sequence
    hmm = HMM([0.9 0.1; 0.1 0.9], [Categorical([1.0, 0.0]), Categorical([0.0, 1.0])])
    z = [1,1,2,2,1,1]
    y = rand(hmm, z)
    @test y[:] == z
end

@testset "Messages $f" for f in [
    (forward, backward, posteriors),
    (forwardlog, backwardlog, posteriorslog)
]
    # Example from https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
    π = [0.7 0.3; 0.3 0.7]
    D = [Categorical([0.9, 0.1]), Categorical([0.2, 0.8])]
    hmm = HMM(π, D)

    O = [1,1,2,1,1]

    α, logtot_alpha = f[1](hmm, O)
    α = round.(α, digits=4)

    β, logtot_beta = f[2](hmm, O)
    β = round.(β, digits=4)

    γ = f[3](hmm, O)
    γ = round.(γ, digits=4)

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

    @test logtot_alpha ≈ logtot_beta atol=1e-12
end

@testset "Viterbi $(typeof(hmm))" for hmm in targets
    # TODO: Better viterbi tests....
    z, y = rand(hmm, 1000);
    z_viterbi = viterbi(hmm, y)
    z_viterbilog = viterbilog(hmm, y)
    @test z == z_viterbi
    # @test z == z_viterbilog
end

# Test high-level API interfaces (types compatibility, ...)
# to ensure that there is no exceptions.

@testset "Integration $(typeof(hmm))" for hmm in targets
    z, y = rand(hmm, 1000)
    z_viterbi = viterbi(hmm, y)
    α, _ = forwardlog(hmm, y)
    β, _ = backwardlog(hmm, y)
    γ = posteriorslog(hmm, y)
    @test size(z) == size(z_viterbi)
    @test size(α) == size(β) == size(γ)

    new_hmm = fit_mle(hmm, y)
    @test size(new_hmm) == size(hmm)
    @test typeof(new_hmm) == typeof(hmm)
    @test typeof(copy(hmm)) == typeof(hmm)
end

@testset "Utilities" begin
    # Make sure we do not relabel states if they are in 1...K
    mapping, _  = compute_transition_matrix([3,3,1,1,2,2])
    for (k, v) in mapping
        @test mapping[k] == v
    end

    mapping, transmat = compute_transition_matrix([3,3,8,8,3,3])
    @test mapping[3] == 1
    @test mapping[8] == 2
    @test transmat == [2/3 1/3; 1/2 1/2]

    transmat = rand_transition_matrix(10)
    @test HMMBase.issquare(transmat)
    @test istransmat(transmat)

    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
    @test nparams(hmm) == 6

    perm = [3,4,2,1]
    hmm1 = rand_hmm(4)
    hmm2 = permute(hmm1, perm)
    @test hmm2.D[1] == hmm1.D[3]
    @test hmm2.D[2] == hmm1.D[4]
    @test hmm2.D[3] == hmm1.D[2]
    @test hmm2.D[4] == hmm1.D[1]
    @test diag(hmm2.π) == diag(hmm1.π)[perm]
end
