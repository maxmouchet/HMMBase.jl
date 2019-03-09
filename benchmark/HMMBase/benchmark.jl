Base.eval(:(have_color = true))
@info "Loading benchmark..."

using BenchmarkTools
using DataFrames, CSV
using Distributions
using StaticArrays
using HMMBase

import Pkg

rand_init_distn(K) = rand(Dirichlet(K, 1))
rand_trans_matrix(K) = collect(rand(Dirichlet(K, 1), K)')

function run_benchmark(Ks, Ts)
    @info "Running benchmark..."
    @info "Ks = $Ks"
    @info "Ts = $Ts"

    results = DataFrame(fn = String[], container = String[], K = Int[], T = Int[], time_ns = Float64[])

    for K in Ks, T in Ts
        @info "K = $K, T = $T"

        π0 = rand_init_distn(K)
        π = rand_trans_matrix(K)

        π0_s = SVector{K}(π0)
        π_s = SMatrix{K,K}(π)

        log_likelihoods = rand(T, K)

        benchmarks = [
            ("messages_forwards", "Array", K, T, @benchmarkable messages_forwards($π0, $π, $log_likelihoods)),
            ("messages_forwards", "StaticArray", K, T, @benchmarkable messages_forwards($π0_s, $π_s, $log_likelihoods)),
            ("messages_backwards", "Array", K, T, @benchmarkable messages_backwards($π0, $π, $log_likelihoods)),
    #         ("messages_backwards", "StaticArray", K, T, @benchmarkable messages_backwards($π0_s, $π_s, $log_likelihoods)),
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

version = string(Pkg.installed()["HMMBase"])
@info "HMMBase v$(version)"

results = run_benchmark(2:2:20, [10, 100, 1000])
out_fp = "HMMBase_v$(version)_$(Int64(round(time()))).csv"

@info "Writing results to $(out_fp)..."
CSV.write(out_fp, results)