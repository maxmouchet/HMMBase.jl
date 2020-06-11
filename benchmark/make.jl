using PkgBenchmark
using HMMBase

# path = dirname(dirname(pathof(HMMBase)))
path = pwd()
println("Benchmarking $path...")

timestamp = Int64(time() * 1e9)
results = benchmarkpkg(path, resultfile = "benchmark_$timestamp.json")
export_markdown("benchmark_$timestamp.md", results)
