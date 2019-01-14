using Test
using HMMBase
using Distributions

@testset "Messages" begin
    # Example from https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
    π = [0.7 0.3; 0.3 0.7]
    D = [Categorical([0.9, 0.1]), Categorical([0.2, 0.8])]
    hmm = HMM(π, D)

    O = [1,1,2,1,1]

    α, logtot_alpha = messages_forwards(hmm, O)
    α = round.(α, digits=4)

    β, logtot_beta = messages_backwards(hmm, O)
    β = round.(β, digits=4)

    γ = forward_backward(hmm, O)
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