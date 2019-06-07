using Test
using HMMBase
using Distributions
using Random
using LinearAlgebra, Combinatorics

Random.seed!(2018)

@testset "HMMBase" begin

  @testset "Constructors" begin
    # Test error are raised
    # wrong Tras Matrix
    @test_throws ErrorException HMM(ones(2,2), [Normal();Normal()])
    # wrong Tras Matrix dimensions
    @test_throws ErrorException HMM([0.8 0.1 0.1; 0.1 0.1 0.8], [Normal(0,1), Normal(10,1)])
    # wrong number of Distributions
    @test_throws ErrorException HMM([0.8 0.2; 0.1 0.9], [Normal(0,1), Normal(10,1), Normal()])
    # wrong distribution size
    @test_throws ErrorException HMM([0.8 0.2; 0.1 0.9], [MvNormal(randn(3)), MvNormal(randn(10))])
    # wrong initial state 
    @test_throws ErrorException HMM([0.1;0.1],[0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
    # wrong initial state length
    @test_throws ErrorException HMM([0.1;0.1;0.8],[0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
  end

  @testset "Generative HHM" begin

    # Test random observations generation for T time steps
    T = 10
    hmm = HMM([0.9 0.1; 0.1 0.9], [Categorical([1.0, 0.0]), Categorical([0.0, 1.0])])
    y, s = rand(hmm, T)
    @test y == s

    # Test random observations generation for T time steps using Multivariate
    T = 10
    b1 = MvNormal( convert(Array,Diagonal([1;1e-12]) ))
    b2 = MvNormal( convert(Array,Diagonal([1e-12;1]) ))
    hmm = HMM([0.9 0.1; 0.1 0.9], [b1,b2])
    y, s = rand(hmm, T)
    @test all([argmax(abs.(y[i])) == s[i] for i in eachindex(s)])

    # Test random observations generation for T time steps with initial state
    T = 2
    hmm = HMM([0.9 0.1; 0.1 0.9], [Categorical([1.0, 0.0]), Categorical([0.0, 1.0])])
    y, s = rand(hmm, T; s0 = 2 )
    @test s[1] == 2

    ## Test random observations generation with a fixed sequence
    hmm = HMM([0.9 0.1; 0.1 0.9], [Categorical([1.0, 0.0]), Categorical([0.0, 1.0])])
    s = [1,1,2,2,1,1]
    y = rand(hmm, s)
    @test y == s

    ## Test random observations generation with a fixed sequence (Multivariate)
    b1 = MvNormal( convert(Array,Diagonal([1;1e-12;1e-12]) ))
    b2 = MvNormal( convert(Array,Diagonal([1e-12;1;1e-12]) ))
    b3 = MvNormal( convert(Array,Diagonal([1e-12;1e-12;1]) ))
    hmm = HMM([0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9], [b1,b2,b3])
    s = [1,2,3,3,2,1]
    y = rand(hmm, s)
    @test all([argmax(abs.(y[i])) == s[i] for i in eachindex(s)])

  end

  @testset "Likelihoods" begin

    ## Testing Likelihood
    hmm = HMM([0.9 0.1; 0.1 0.9], [Categorical([1.0-1e-8, 1e-8]), Categorical([1e-8, 1.0-1e-8])])
    s = [1,1,2,2,1,1]
    y = rand(hmm, s)
    L = HMMBase.likelihoods(hmm,y)
    @test all([argmax(L[t,:]) == s[t] for t in eachindex(s)])
    @test all([maximum(L[t,:]) == 1.0-1e-8 for t in eachindex(s)])
    L = HMMBase.likelihoods(hmm, y; log = true)
    @test all([maximum(L[t,:]) == log(1-1e-8) for t in eachindex(s)])

    ## Test random observations generation with a fixed sequence (Multivariate)
    b1 = MvNormal( convert(Array,Diagonal([1;1e-12;1e-12]) ))
    b2 = MvNormal( convert(Array,Diagonal([1e-12;1;1e-12]) ))
    b3 = MvNormal( convert(Array,Diagonal([1e-12;1e-12;1]) ))
    hmm = HMM([0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9], [b1,b2,b3])
    s = [1,2,3,3,2,1]
    y = rand(hmm, s)
    L = HMMBase.likelihoods(hmm,y)
    @test all([argmax(L[t,:]) == s[t] for t in eachindex(s)])

  end

  @testset "Messages" begin
    #    # Example from https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
    # define Markov model's parameters (λ)
    S = 2                                                  
    # number of states
    a = [0.5; 0.5]                                         
    # initial state probability  
    A = [0.7 0.3; 0.3 0.7]                                 
    # transmission Matrix  [a_11 a21; a12 a_22]

    B = [Categorical([0.9, 0.1]), Categorical([0.2, 0.8])] 
    # observation distribution [b_1; b_2]
    T = 5                     
    # time window length
    o = [1;1;2;1;1]           
    # observations
    s_all = collect(multiset_permutations([1;2],[5;5],5))
    # this has all possible state sequences e.g. [1;1;1;1;1], [1;1;1;1;2] ...
    # these are S^T permutations

    # Probability of sequence of states
    Pλ_s = zeros(length(s_all)) 
    for z = 1:length(s_all)
      Pλ_s[z] = a[1] 
      # initial distribution, since they are equal so we don't need to do this twice
      for t = 2:T
        Pλ_s[z] *= A[ s_all[z][t-1], s_all[z][t] ]
      end
    end
    @assert sum(Pλ_s) ≈ 1 # check it's a probability 

    # Likelihood of O given a sequence of states
    Lλ_o_s = ones(length(s_all)) 
    for z = 1:length(s_all)
      for t = 1:T
        Lλ_o_s[z] *= pdf(B[ s_all[z][t] ], o[t])
      end
    end

    Lkh = sum(Pλ_s .* Lλ_o_s) #likelihood
    hmm = HMM(a,A,B)
    L = HMMBase.likelihoods(hmm,o)

    # Baum's original messages
    alphas = HMMBase.baum_forward(hmm,L)
    betas = HMMBase.baum_backward(hmm,L)
    @test sum(alphas[end,:]) ≈ Lkh
    @test all(sum(alphas.*betas,dims=2) .≈ Lkh)

    # normalized messages (data from Wikipedia)
    alphas, c = HMMBase.forward(hmm,L)
    @test round.(alphas, digits = 4) == [
                                         0.8182 0.1818;
                                         0.8834 0.1166;
                                         0.1907 0.8093;
                                         0.7308 0.2692;
                                         0.8673 0.1327;
                                        ]

    betas = HMMBase.backward(hmm,L)
    betas2 = HMMBase.backward(hmm,L,c)
    @test round.(betas, digits=4) == [
                                      0.5923 0.4077;
                                      0.3763 0.6237;
                                      0.6533 0.3467;
                                      0.6273 0.3727;
                                      1.0    1.0;
                                     ]

    gammas = alphas .* betas
    gammas2 = alphas .* betas2
    @test round.(gammas ./ sum(gammas, dims=2) , digits=4) == [
                                                               0.8673 0.1327;
                                                               0.8204 0.1796;
                                                               0.3075 0.6925;
                                                               0.8204 0.1796;
                                                               0.8673 0.1327;
                                                              ]

    # println( gammas  )  # gammas by Wikipedia definitions are not prob!
    # println( gammas2 )  # gammas by Devijver definition are:
    @test all(sum(gammas2, dims=2) .≈ 1)

  end

  @testset "Decoding" begin 

    # test with large random HMM
    T = 200       # time steps 
    S = 100       # number of states
    A = rand(S,S) # random trans matrix
    for i = 1:S
      A[i,:] ./= sum(A[i,:]) 
    end
    B = [ Normal(i,1/(4*S)) for i = range(-1, stop=1, length=S)] # emissin prob

    hmm = HMM(A, B)
    y, z = rand(hmm, T)

    L = HMMBase.likelihoods(hmm,y)
    z_viterbi0 = HMMBase.viterbi(hmm, y)
    z_viterbi = HMMBase.viterbi(hmm, L)
    @test all(z_viterbi0 .== z_viterbi)
    # without normalization
    z_unviterbi = HMMBase.viterbi(hmm, L; normalize=false)

    # with normalization (Wiki)
    alphas, c = HMMBase.forward(hmm, L)
    betas = HMMBase.backward(hmm, L)
    @test all(sum(betas[1:end-1,:], dims=2) .≈ 1)

    # with normalization (Devijver)
    betas2 = HMMBase.backward(hmm, L, c)

    # without normalization
    alphas_un = HMMBase.baum_forward(hmm, L)
    betas_un = HMMBase.baum_backward(hmm, L)

    gammas = alphas .* betas
    gammas2 = alphas .* betas2
    @test all(sum(gammas2, dims=2) .≈ 1)
    gammas_un = alphas_un .* betas_un

    z_gammas = [ argmax(gammas[t,:]) for t = 1:T]
    z_gammas2 = [ argmax(gammas2[t,:]) for t = 1:T]
    z_gammas_un = [ argmax(gammas_un[t,:]) for t = 1:T]

    @test all(z_viterbi .== z)
    @test all(z_unviterbi .== z)
    @test all(z_viterbi .== z_gammas) 
    @test all(z_viterbi .== z_gammas2) 
    @test all(z_viterbi .== z_gammas_un) 

  end


  @testset "Learn HMM - Categorical" begin

    verb = false
    Nt = 100
    Ns = 10
    A = rand(Ns,Ns) # random trans matrix
    for i = 1:Ns
      A[i,:] ./= sum(A[i,:]) 
    end
    B = [ Categorical(normalize(rand(Ns),1)) for i = 1:Ns] # emissin prob

    hmm = HMM(A, B)
    y, z = rand(hmm,Nt)
    A0 = diagm(0 => 0.5 .*ones(Ns), 1 => 0.5 .*ones(Ns-1))  # trans matrix
    A0[end,end-1] = 0.5
    B0 = [ Categorical(normalize(rand(Ns),1)) for i = 1:Ns] # emissin prob
    hmm0 = HMM(deepcopy(A0), deepcopy(B0))

    L = HMMBase.likelihoods(hmm0, y, log = false)
    alpha = HMMBase.baum_forward(hmm0, L)
    beta  = HMMBase.baum_backward(hmm0, L)
    gamma = HMMBase.posteriors(alpha,beta)
    epsilon = zeros(Nt,Ns,Ns)

    HMMBase.update_A!(hmm0, epsilon, alpha, beta, gamma, L)
    @test HMMBase.isTMatrix(hmm0.A)

    HMMBase.update_a!(hmm0,gamma)
    @test sum(hmm0.a) ≈ 1 

    HMMBase.update_B!(hmm0, gamma, y)
    @test all( [sum(b.p) ≈ 1 for b in hmm0.B] )

    hmm0 = HMM(deepcopy(A0), deepcopy(B0))
    nlogL = HMMBase.baum_welch!(hmm0, y; maxit = 100, normalize = false)

    @test HMMBase.isTMatrix(hmm0.A)
    @test sum(hmm0.a) ≈ 1 
    @test all([sum(hmm0.A[i,:]) ≈ 1 for i = 1:size(hmm0.A,1)])
    # checking negative log likelihood decreases
    @test issorted(nlogL; rev = true)

    hmm0 = HMM(deepcopy(A0), deepcopy(B0))
    nlogL2 = HMMBase.baum_welch!(hmm0, y; maxit = 100, verbose = verb)
    @test issorted(nlogL; rev = true)
    # checking with scaling beta and gamma same path is taken
    @test norm(nlogL2 - nlogL) < 1e-7

    Nt = 1000
    hmm0 = deepcopy(hmm)
    y, z = rand(hmm,Nt)
    nlogL1 = HMMBase.baum_welch!(hmm0, y; maxit = 100, verbose = verb)
    @test issorted(nlogL; rev = true)
    # training HMM on data it generated: 
    # cost should not decrease much
    @test abs(nlogL2[1]-nlogL2[end]) < abs(nlogL[1]-nlogL1[end])

  end

  @testset "Learn HMM - Normal" begin

    verb = true
    Nt = 500
    Ns = 30
    A = rand(Ns,Ns) # random trans matrix
    for i = 1:Ns
      A[i,:] ./= sum(A[i,:]) 
    end
    B = [ Normal(i,1/(4*Ns)) for i = range(-1, stop=1, length=Ns)] # emissin prob

    hmm = HMM(A, B)
    y, z = rand(hmm,Nt)
    A0 = rand(Ns,Ns) # random trans matrix
    for i = 1:Ns
      A0[i,:] ./= sum(A0[i,:]) 
    end
    B0 = [ Normal(i,1/(4*Ns)) for i = range(-1, stop=1, length=Ns)] # emissin prob
    hmm0 = HMM(deepcopy(A0), deepcopy(B0))

    L = HMMBase.likelihoods(hmm0, y, log = false)
    alpha = HMMBase.baum_forward(hmm0, L)
    beta  = HMMBase.baum_backward(hmm0, L)
    gamma = HMMBase.posteriors(alpha,beta)

    hmm0 = HMM(deepcopy(A0), deepcopy(B0))
    nlogL = HMMBase.baum_welch!(hmm0, y; verbose = verb)
    @test issorted(nlogL; rev = true)
    @test sum(hmm0.a) ≈ 1 
    @test all([sum(hmm0.A[i,:]) ≈ 1 for i = 1:size(hmm0.A,1)])

    Nt = 1000
    hmm0 = deepcopy(hmm)
    y, z = rand(hmm,Nt)
    nlogL2 = HMMBase.baum_welch!(hmm0, y; maxit = 100, verbose = verb)
    @test issorted(nlogL; rev = true)
    # training HMM on data it generated: 
    # cost should not decrease much
    @test abs(nlogL2[1]-nlogL2[end]) < abs(nlogL[1]-nlogL[end])

  end

  @testset "Learn HMM - Exponential + Gamma" begin

    verb = false
    Nt = 500
    Ns = 30
    A = rand(Ns,Ns) # random trans matrix
    for i = 1:Ns
      A[i,:] ./= sum(A[i,:]) 
    end
    B = [ [ Gamma(rand()) for i = 1:div(Ns,2) ]..., 
          [ Exponential(rand()) for i = 1:div(Ns,2) ]...]# emissin prob

    hmm = HMM(A, B)
    y, z = rand(hmm,Nt)
    A0 = rand(Ns,Ns) # random trans matrix
    for i = 1:Ns
      A0[i,:] ./= sum(A0[i,:]) 
    end
    B0 = [ [ Gamma(rand()) for i = 1:div(Ns,2) ]..., 
          [ Exponential(rand()) for i = 1:div(Ns,2) ]...]# emissin prob
    hmm0 = HMM(deepcopy(A0), deepcopy(B0))

    L = HMMBase.likelihoods(hmm0, y, log = false)
    alpha = HMMBase.baum_forward(hmm0, L)
    beta  = HMMBase.baum_backward(hmm0, L)
    gamma = HMMBase.posteriors(alpha,beta)

    hmm0 = HMM(deepcopy(A0), deepcopy(B0))
    nlogL = HMMBase.baum_welch!(hmm0, y; verbose = verb)
    @test issorted(nlogL; rev = true)
    @test sum(hmm0.a) ≈ 1 
    @test all([sum(hmm0.A[i,:]) ≈ 1 for i = 1:size(hmm0.A,1)])

    Nt = 1000
    hmm0 = deepcopy(hmm)
    y, z = rand(hmm,Nt)
    nlogL2 = HMMBase.baum_welch!(hmm0, y; maxit = 100, verbose = verb)
    @test issorted(nlogL; rev = true)
    # training HMM on data it generated: 
    # cost should not decrease much
    @test abs(nlogL2[1]-nlogL2[end]) < abs(nlogL[1]-nlogL[end])

  end

  @testset "Learn HMM - MvNormal" begin

    verb = false
    Nt = 500
    Ns = 30
    Ny = 3
    A = rand(Ns,Ns) # random trans matrix
    for i = 1:Ns
      A[i,:] ./= sum(A[i,:]) 
    end
    B = [ MvNormal(randn(Ny),randn(Ny)) for i = 1:Ns] # emissin prob

    hmm = HMM(A, B)
    y, z = rand(hmm,Nt)
    A0 = rand(Ns,Ns) # random trans matrix
    for i = 1:Ns
      A0[i,:] ./= sum(A0[i,:]) 
    end
    B0 = [ MvNormal(randn(Ny),randn(Ny)) for i = 1:Ns] # emissin prob
    hmm0 = HMM(deepcopy(A0), deepcopy(B0))

    L = HMMBase.likelihoods(hmm0, y, log = false)
    alpha = HMMBase.baum_forward(hmm0, L)
    beta  = HMMBase.baum_backward(hmm0, L)
    gamma = HMMBase.posteriors(alpha,beta)

    hmm0 = HMM(deepcopy(A0), deepcopy(B0))
    nlogL = HMMBase.baum_welch!(hmm0, y; verbose = verb)
    @test issorted(nlogL; rev = true)
    @test sum(hmm0.a) ≈ 1 
    @test all([sum(hmm0.A[i,:]) ≈ 1 for i = 1:size(hmm0.A,1)])

    Nt = 1000
    hmm0 = deepcopy(hmm)
    y, z = rand(hmm,Nt)
    nlogL2 = HMMBase.baum_welch!(hmm0, y; maxit = 100, verbose = verb)
    @test issorted(nlogL; rev = true)
    # training HMM on data it generated: 
    # cost should not decrease much
    @test abs(nlogL2[1]-nlogL2[end]) < abs(nlogL[1]-nlogL[end])

  end

end
