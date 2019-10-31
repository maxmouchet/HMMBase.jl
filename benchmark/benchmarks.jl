# https://juliaci.github.io/PkgBenchmark.jl/stable/run_benchmarks/
# https://github.com/JuliaCI/PkgBenchmark.jl/blob/master/benchmark/benchmarks.jl
using BenchmarkTools
using Distributions
using HMMBase

# Data generation

function rand_hmm(K)
    A = randtransmat(K)
    B = [Normal(rand()*100, rand()*10) for _ in 1:K]
    HMM(A, B)
end

HMMs = Dict([K => rand_hmm(K) for K in 2:2:10])
Ys = Dict([K => rand(HMMs[K], 5000)[2] for K in keys(HMMs)])

# Suite

const SUITE = BenchmarkGroup()

SUITE["messages"] = BenchmarkGroup()

for f in (forward, forwardlog, backward, backwardlog)
    SUITE["messages"][string(f)] = BenchmarkGroup()
    for (K, hmm) in HMMs
        y = Ys[K]
        SUITE["messages"][string(f)][K] = @benchmarkable ($f)($hmm, $y)
    end
end