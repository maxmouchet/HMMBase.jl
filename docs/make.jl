using Documenter
using Glob

using Distributions
using HMMBase
using Random

Random.seed!(2019)

# https://discourse.julialang.org/t/deactivate-plot-display-to-avoid-need-for-x-server/19359
ENV["GKSwstype"] = "nul"

# Documenter wants relative paths
examples = map(glob("*.md", joinpath(@__DIR__, "src", "examples"))) do x
    joinpath("examples", basename(x))
end

makedocs(
    sitename = "HMMBase",
    modules = [HMMBase],
    format = Documenter.HTML(assets = ["assets/goatcounter.js"]),
    pages = [
        "index.md",
        "Manual" => ["basics.md", "models.md", "algorithms.md", "utilities.md"],
        "Examples" => examples,
        "internals.md",
        "migration.md",
        "_index.md",
    ],
)

deploydocs(repo = "github.com/maxmouchet/HMMBase.jl.git")
