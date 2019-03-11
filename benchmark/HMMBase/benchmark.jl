Base.eval(:(have_color = true))
@info "Loading benchmark..."

using BenchmarkTools
using DataFrames, CSV
using Distributions
using StaticArrays
using HMMBase
using Random

import Pkg

rand_init_distn(K) = rand(Dirichlet(K, 1))
rand_trans_matrix(K) = collect(rand(Dirichlet(K, 1), K)')
rand_distn(dim) = MvNormal(rand(dim), diagm(0 => ones(dim)))

function run_benchmark(Ks, Ts)
    @info "Running benchmark..."
    @info "Ks = $Ks"
    @info "Ts = $Ts"

    results = DataFrame(fn = String[], container = String[], K = Int[], T = Int[], time_ns = Float64[])

    for K in Ks, T in Ts
        @info "K = $K, T = $T"

        # HMM with Julia arrays
        π0  = rand_init_distn(K)
        π   = rand_trans_matrix(K)
        D   = [rand_distn(2) for _ in 1:K]
        hmm = HMM(π0, π, D)
        
        # HMM with static arrays
        π0_s  = SVector{K}(π0)
        π_s   = SMatrix{K,K}(π)
        hmm_s = HMM(π0_s, π_s, D)

        z, y = rand(hmm, T)
        likelihoods = HMMBase.likelihoods(hmm, y)
        log_likelihoods = HMMBase.log_likelihoods(hmm, y)

        # Commented functions that are not yet working...
        benchmarks = [
            ("rand", "Array", K, T, @benchmarkable rand($hmm, $T)),
            ("rand", "StaticArray", K, T, @benchmarkable rand($hmm_s, $T)),
            ("log_likelihoods", "Array", K, T, @benchmarkable HMMBase.log_likelihoods($hmm, $y)),
            ("log_likelihoods", "StaticArray", K, T, @benchmarkable HMMBase.log_likelihoods($hmm_s, $y)),
            ("messages_forwards", "Array", K, T, @benchmarkable messages_forwards($π0, $π, $log_likelihoods)),
            ("messages_forwards", "StaticArray", K, T, @benchmarkable messages_forwards($π0_s, $π_s, $log_likelihoods)),
            ("messages_backwards", "Array", K, T, @benchmarkable messages_backwards($π0, $π, $log_likelihoods)),
            # ("messages_backwards", "StaticArray", K, T, @benchmarkable messages_backwards($π0_s, $π_s, $log_likelihoods)),
            ("viterbi", "Array", K, T, @benchmarkable viterbi($π0, $π, $likelihoods)),
            ("viterbi", "StaticArray", K, T, @benchmarkable viterbi($π0, $π, $likelihoods))
        ]

        for (fn, container, K, T, benchmark) in benchmarks
            result = run(benchmark)
            for i = 1:length(result.times)
                push!(results, (fn, container, K, T, result.times[i]))
            end
        end
    end

    results
end

BenchmarkTools.DEFAULT_PARAMETERS.evals = 10
BenchmarkTools.DEFAULT_PARAMETERS.samples = 500
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
@info BenchmarkTools.DEFAULT_PARAMETERS

seed = 2019
Random.seed!(seed)

version = string(Pkg.installed()["HMMBase"])
@info "HMMBase v$(version)"

results = run_benchmark(2:2:20, [10, 100, 1000])
out_fp = "HMMBase_v$(version)_$(Int64(round(time())))_$(seed).csv"

@info "Writing results to $(out_fp)..."
CSV.write(out_fp, results)