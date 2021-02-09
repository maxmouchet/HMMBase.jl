@testset "< v1.1" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    z, y = rand(hmm, 2500, seq = true)
    @test (@test_logs (:warn,) match_mode=:any forward(hmm, y, logl = true)) == (@test_logs (:warn,) match_mode=:any forward(hmm, y, logl = false)) == forward(hmm, y)
    @test (@test_logs (:warn,) match_mode=:any backward(hmm, y, logl = true)) ==
          (@test_logs (:warn,) match_mode=:any backward(hmm, y, logl = false)) ==
          backward(hmm, y)
    @test (@test_logs (:warn,) match_mode=:any posteriors(hmm, y, logl = true)) ==
          (@test_logs (:warn,) match_mode=:any posteriors(hmm, y, logl = false)) ==
          posteriors(hmm, y)
    @test (@test_logs (:warn,) match_mode=:any viterbi(hmm, y, logl = false)) == (@test_logs (:warn,) match_mode=:any viterbi(hmm, y, logl = true)) == z
    @test (@test_logs (:warn,) match_mode=:any likelihoods(hmm, y)) == exp.(loglikelihoods(hmm, y))
    @test (@test_logs (:warn,) match_mode=:any likelihoods(hmm, y, logl = true)) == loglikelihoods(hmm, y)
end

@testset "< v1.0" begin
    hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0, 1), Normal(10, 1)])
    z, y = rand(hmm, 2500, seq = true)
    @test (@test_deprecated n_parameters(hmm)) == nparams(hmm)
    @test (@test_deprecated log_likelihoods(hmm, y)) == loglikelihoods(hmm, y)
    @test (@test_deprecated forward_backward(hmm, y)) == posteriors(hmm, y)
    @test (@test_deprecated messages_forwards(hmm, y)) == forward(hmm, y)
    @test (@test_deprecated messages_backwards(hmm, y)) == backward(hmm, y)
    @test (@test_deprecated compute_transition_matrix(z)) == gettransmat(z, relabel = true)
    @test size((@test_deprecated rand_transition_matrix(5, 2.0))) == (5, 5)
end
