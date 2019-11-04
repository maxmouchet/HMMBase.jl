Base.eval(:(have_color = true))

using PkgBenchmark
using HMMBase

# results = benchmarkpkg(parent(pathof(HMMBase)))
results = benchmarkpkg(pwd())
export_markdown("benchmark.md", results)