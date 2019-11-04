using Test
using HMMBase
using Distributions
using Random

Random.seed!(2019)

hmms = [
    HMM([0.9 0.1; 0.1 0.9], [Normal(10,1), Gamma(1,1)]),
    HMM([0.9 0.1; 0.1 0.9], [Categorical([0.1, 0.2, 0.7]), Categorical([0.5, 0.5])]),
    HMM([0.9 0.1; 0.1 0.9], [MvNormal([0.0,0.0], [1.0,1.0]), MvNormal([10.0,10.0], [1.0,1.0])])
]

@testset "Integration $(typeof(hmm))" for hmm in hmms
    # HMM API
    @test hmm !== copy(hmm)

    z, y = rand(hmm, 1000)
    @test size(z, 1) == size(y, 1)
    @test size(y, 2) == size(hmm, 2)

    yp = rand(hmm, z)
    @test size(z, 1) == size(y, 1)
    @test size(y, 2) == size(hmm, 2)

    L = likelihoods(hmm, y)
    LL = likelihoods(hmm, y, logl = true)
    @test size(L) == size(LL)

    # Forward/Backward
    α1, logtot1 = forward(hmm, y)
    α2, logtot2 = forward(hmm, y, logl = true)

    @test logtot1 ≈ logtot2
    @test α1 ≈ α2

    β1, logtot3 = backward(hmm, y)
    β2, logtot4 = backward(hmm, y, logl = true)

    @test logtot3 ≈ logtot4
    @test β1 ≈ β2

    logtot5 = likelihood(hmm, y)
    logtot6 = likelihood(hmm, y, logl = true)

    @test logtot5 ≈ logtot6

    @test size(α1) == size(α2) == size(β1) == size(β2)
    @test logtot1 ≈ logtot2 ≈ logtot3 ≈ logtot4 ≈ logtot5 ≈ logtot6

    γ1 = posteriors(hmm, y)
    γ2 = posteriors(hmm, y, logl = true)

    @test γ1 ≈ γ2

    # Viterbi
    zv1 = viterbi(hmm, y)
    zv2 = viterbi(hmm, y; logl = true)
    @test size(zv1) == size(zv2) == size(z)

    # MLE
    hmm2 = fit_mle(hmm, y, display = :final, init = nothing)
    @test size(hmm2) == size(hmm)
    @test typeof(hmm2) == typeof(hmm)

    hmm2 = fit_mle(hmm, y, display = :final, init = :kmeans)
    @test size(hmm2) == size(hmm)
    @test typeof(hmm2) == typeof(hmm)
end