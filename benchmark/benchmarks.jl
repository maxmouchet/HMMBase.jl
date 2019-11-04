# https://juliaci.github.io/PkgBenchmark.jl/stable/run_benchmarks/
# https://github.com/JuliaCI/PkgBenchmark.jl/blob/master/benchmark/benchmarks.jl
using BenchmarkTools
using Distributions
using HMMBase

# Data generation

function rand_hmm(K)
    A = randtransmat(K)
    B = [Normal(rand() * 100, rand() * 10) for _ in 1:K]
    HMM(A, B)
end

HMMs = Dict([K => rand_hmm(K) for K in 2:2:10])
Ys = Dict([K => rand(HMMs[K], 5000)[2] for K in keys(HMMs)])

# Suite

const SUITE = BenchmarkGroup()

SUITE["messages"] = BenchmarkGroup()

for f in (forward, backward, viterbi)
    SUITE["messages"][string(f)] = BenchmarkGroup()
    for logl in [true, false]
        SUITE["messages"][string(f)][string(logl)] = BenchmarkGroup()
        for (K, hmm) in HMMs
            L = likelihoods(hmm, Ys[K], logl = logl)
            SUITE["messages"][string(f)][string(logl)][K] = @benchmarkable ($f)($hmm.a, $hmm.A, $L, logl = $logl)
        end
    end
end

if "BENCHMARK_PYTHON" in keys(ENV)
    using PyCall
    pyhsmm = pyimport("pyhsmm")

    # hmmlearn
    # TODO

    # pyhsmm
    SUITE["pyhsmm"] = BenchmarkGroup()

    for f in (
        pyhsmm.internals.hmm_states.HMMStatesPython._messages_forwards_normalized,
        pyhsmm.internals.hmm_states.HMMStatesPython._messages_backwards_normalized
    )
        SUITE["pyhsmm"][string(f)] = BenchmarkGroup()
        for (K, hmm) in HMMs
            L = likelihoods(hmm, Ys[K], logl = true)
            SUITE["pyhsmm"][string(f)][K] = @benchmarkable ($f)($hmm.A, $hmm.a, $L)
        end
    end
end