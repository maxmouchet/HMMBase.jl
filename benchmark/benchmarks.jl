# https://juliaci.github.io/PkgBenchmark.jl/stable/run_benchmarks/
# https://github.com/JuliaCI/PkgBenchmark.jl/blob/master/benchmark/benchmarks.jl
using BenchmarkTools
using Distributions
using HMMBase
using PyCall

# Helpers

# Recursively create benchmark groups
function mkgrp!(base, path)
    path = map(string, path)
    curr = base
    for k in path
        if !(k in keys(curr))
            curr[k] = BenchmarkGroup()
        end
        curr = curr[k]
    end
    curr
end

function mkbench!(f, base, path)
    grp = mkgrp!(base, path[1:end-1])
    grp[path[end]] = f()
end

# Dataset

function randhmm(K)
    A = randtransmat(K)
    B = [Normal(rand() * 100, rand() * 10) for _ in 1:K]
    HMM(A, B)
end

HMMs = Dict([K => randhmm(K) for K in 2:2:10])
Ys = Dict([K => rand(HMMs[K], 5000) for K in keys(HMMs)])

# Suite
const SUITE = BenchmarkGroup()

# HMMBase

for f in (forward, backward, viterbi), logl in [true, false], (K, hmm) in HMMs
    L = likelihoods(hmm, Ys[K], logl = logl)
    mkbench!(SUITE, ["hmmbase", f, logl, K]) do
        @benchmarkable ($f)($hmm.a, $hmm.A, $L, logl = $logl)
    end
end

# pyhsmm

pyhsmm = pyimport("pyhsmm")
pyhsmmi = pyimport("pyhsmm.internals.hmm_messages_interface")

fs = [
    pyhsmm.internals.hmm_states.HMMStatesPython._messages_forwards_normalized,
    pyhsmm.internals.hmm_states.HMMStatesPython._messages_backwards_normalized
]

for f in fs, (K, hmm) in HMMs
    L = likelihoods(hmm, Ys[K], logl = true)
    mkbench!(SUITE, ["pyhsmm", f, K]) do
        @benchmarkable ($f)($hmm.A, $hmm.a, $L)
    end
end

for (K, hmm) in HMMs
    L = likelihoods(hmm, Ys[K], logl = true)
    buf = zeros(Int32, size(L, 1))

    a = hmm.a
    A = PyReverseDims(permutedims(hmm.A))
    L = PyReverseDims(permutedims(L))

    mkbench!(SUITE, ["pyhsmm", "viterbi", K]) do
        @benchmarkable pyhsmmi.viterbi($A, $L, $a, $buf)
    end
end

# hmmlearn

hmml = pyimport("hmmlearn.hmm")

function tohmmlearn(hmm::AbstractHMM)
    K = size(hmm, 1)
    model = hmml.GaussianHMM(n_components = K, covariance_type = :diag)
    model.startprob_ = hmm.a
    model.transmat_ = hmm.A # TODO: PermuteDims ?
    model.means_ = [d.μ for d in hmm.B]
    model.covars_ = reshape([d.σ^2 for d in hmm.B], (1, 1, :))
    model
end

fs = [:_do_forward_pass, :_do_backward_pass, :_do_viterbi_pass]

for f in fs, (K, hmm) in HMMs
    model = tohmmlearn(hmm)
    L = likelihoods(hmm, Ys[K], logl = true)
    mkbench!(SUITE, ["hmmlearn", f, K]) do
        @benchmarkable $(model).$(f)($L)
    end
end
