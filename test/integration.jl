using Test
using HMMBase
using Distributions
using Random

Random.seed!(2019)

hmms = [
    HMM([0.9 0.1; 0.1 0.9], [Normal(10, 1), Gamma(1, 1)]),
    HMM([0.9 0.1; 0.1 0.9], [Categorical([0.1, 0.2, 0.7]), Categorical([0.5, 0.5])]),
    HMM([0.9 0.1; 0.1 0.9], [MvNormal([0.0, 0.0], [1.0, 1.0]), MvNormal([10.0, 10.0], [1.0, 1.0])]),
]

@testset "Integration $(typeof(hmm))" for hmm in hmms, T in [0, 1, 1000]
    K = size(hmm, 1)

    # HMM API
    @test hmm !== copy(hmm)

    z, y = rand(hmm, T, seq = true)
    @test size(z, 1) == size(y, 1)
    @test size(y, 2) == size(hmm, 2)

    yp = rand(hmm, z)
    @test size(z, 1) == size(y, 1)
    @test size(y, 2) == size(hmm, 2)

    LL = loglikelihoods(hmm, y)
    @test size(LL) == (T, K)

    # Forward/Backward
    α1, logtot1 = forward(hmm, y)
    β1, logtot2 = backward(hmm, y)
    γ1 = posteriors(hmm, y)
    logtot3 = loglikelihood(hmm, y)

    @test size(α1) == size(β1) == size(γ1)
    @test logtot1 ≈ logtot2 ≈ logtot3

    # Viterbi
    zv1 = viterbi(hmm, y)
    @test size(zv1) == size(z)

    # MLE
    if T > 2
        hmm2, _ = fit_mle(hmm, y, maxiter = 1, display = :iter)
        @test size(hmm2) == size(hmm)
        @test typeof(hmm2) == typeof(hmm)

        hmm2, _ = fit_mle(hmm, y, init = nothing)
        @test size(hmm2) == size(hmm)
        @test typeof(hmm2) == typeof(hmm)

        hmm2, _ = fit_mle(hmm, y, init = :kmeans, robust = true)
        @test size(hmm2) == size(hmm)
        @test typeof(hmm2) == typeof(hmm)
    end
end
