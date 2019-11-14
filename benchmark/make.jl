using PkgBenchmark
using HMMBase

# TODO: Add timestamp to result files
# results = benchmarkpkg(parent(pathof(HMMBase)))
results = benchmarkpkg(pwd(), resultfile = "benchmark.json")
export_markdown("benchmark.md", results)