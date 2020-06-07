using Distributions
using HMMBase
using Test

@testset "< v1.1" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
    y = rand(hmm, 2500)
    @test likelihoods(hmm, y) == exp.(loglikelihoods(hmm, y))
    @test likelihoods(hmm, y, logl = true) == loglikelihoods(hmm, y)
end

@testset "< v1.0" begin
    @test size(rand_transition_matrix(5, 2.0)) == (5, 5)
    # TODO: More
end
