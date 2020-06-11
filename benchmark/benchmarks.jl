# https://juliaci.github.io/PkgBenchmark.jl/stable/run_benchmarks/
# https://github.com/JuliaCI/PkgBenchmark.jl/blob/master/benchmark/benchmarks.jl
using BenchmarkTools
using Distributions
using HMMBase
using PyCall

include("helpers.jl")

# hmmlearn
hmmlearn = pyimport("hmmlearn.hmm")

# pyhsmm
pyhsmm_viterbi = pyimport("pyhsmm.internals.hmm_messages_interface").viterbi
HMMStatesPython = pyimport("pyhsmm.internals.hmm_states").HMMStatesPython

const SUITE = BenchmarkGroup()

function randhmm(K)
    A = randtransmat(K)
    B = [Normal(rand() * 100, rand() * 10) for _ = 1:K]
    HMM(A, B)
end

HMMs = Dict([K => randhmm(K) for K = 2:2:10])
Ys = Dict([K => rand(HMMs[K], 5000) for K in keys(HMMs)])

# HMMBase

for f in (forward, backward, viterbi), (K, hmm) in HMMs
    LL = loglikelihoods(hmm, Ys[K])
    mkbench!(SUITE, ["hmmbase", f, K]) do
        @benchmarkable ($f)($hmm.a, $hmm.A, $LL)
    end
end

# pyhsmm

fs = [
    HMMStatesPython._messages_forwards_normalized,
    HMMStatesPython._messages_backwards_normalized,
]

for f in fs, (K, hmm) in HMMs
    LL = loglikelihoods(hmm, Ys[K])
    mkbench!(SUITE, ["pyhsmm", f, K]) do
        @benchmarkable ($f)($hmm.A, $hmm.a, $LL)
    end
end

for (K, hmm) in HMMs
    LL = loglikelihoods(hmm, Ys[K])
    buf = zeros(Int32, size(LL, 1))

    a = hmm.a
    A = PyReverseDims(permutedims(hmm.A))
    LL = PyReverseDims(permutedims(LL))

    mkbench!(SUITE, ["pyhsmm", "viterbi", K]) do
        @benchmarkable pyhsmm_viterbi($A, $LL, $a, $buf)
    end
end

# hmmlearn

function tohmmlearn(hmm::AbstractHMM)
    K = size(hmm, 1)
    model = hmmlearn.GaussianHMM(n_components = K, covariance_type = :diag)
    model.startprob_ = hmm.a
    model.transmat_ = hmm.A # permutedims(...) ?
    model.means_ = [d.μ for d in hmm.B]
    model._covars_ = reshape([d.σ^2 for d in hmm.B], (1, 1, :))
    model
end

fs = [:_do_forward_pass, :_do_backward_pass, :_do_viterbi_pass]

for f in fs, (K, hmm) in HMMs
    model = tohmmlearn(hmm)
    LL = loglikelihoods(hmm, Ys[K])
    mkbench!(SUITE, ["hmmlearn", f, K]) do
        @benchmarkable $(model).$(f)($LL)
    end
end
