using Test
using HMMBase
using Distributions
using Random

Random.seed!(2019)

hmms = [
    HMM([0.8 0.2; 0.2 0.8], [Normal(10, 1), Gamma(1, 1)]),
    HMM([0.8 0.2; 0.2 0.8], [Categorical([0.1, 0.2, 0.7]), Categorical([0.5, 0.5])]),
    HMM([0.8 0.2; 0.2 0.8], [MvNormal([0.0, 0.0], [1.0, 1.0]), MvNormal([10.0, 10.0], [1.0, 1.0])]),
]

@testset "Integration $(typeof(hmm))" for hmm in hmms, T in [0, 1, 100], N in [0, 1, 100]
    K = size(hmm, 1)

    # HMM API
    @test hmm !== copy(hmm)

    z, y = rand(hmm, T, N, seq = true)
    @test size(z, 1) == size(y, 1)
    @test size(z, 2) == last(size(y))
    ((ndims(y) > 2) && (N >= 1)) && (@test size(y, 2) == size(hmm, 2))

    y = rand(hmm, z)
    @test size(z, 1) == size(y, 1)
    ((ndims(y) > 2) && (N >= 1)) && (@test size(y, 2) == size(hmm, 2))

    LL = loglikelihoods(hmm, y)
    @test size(LL) == (T, K, N)

    # Forward/Backward
    α1, logtot1 = forward(hmm, y)
    β1, logtot2 = backward(hmm, y)
    γ1 = posteriors(hmm, y)
    logtot3 = loglikelihood(hmm, y)

    @test size(α1) == size(β1) == size(γ1)
    @test logtot1 ≈ logtot2 ≈ logtot3

    # Viterbi
    zv1, _ = viterbi(hmm, y)
    @test size(zv1) == size(z)

    # MLE
    if ((T > 2)&&(N > 2))
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