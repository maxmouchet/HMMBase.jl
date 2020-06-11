using Documenter
using Glob
using Literate

using Distributions
using HMMBase
using Random

Random.seed!(2019)

# https://discourse.julialang.org/t/deactivate-plot-display-to-avoid-need-for-x-server/19359
ENV["GKSwstype"] = "nul"

examples = []

if !("SKIP_EXAMPLES" in keys(ENV))
    examples = map(glob("*.jl", "examples/")) do x
        joinpath("examples/", split(basename(x), ".")[1])
    end
    for example in examples
        Literate.markdown("$(example).jl", "docs/src/examples", documenter = true)
    end
end

makedocs(
    sitename = "HMMBase",
    modules = [HMMBase],
    format = Documenter.HTML(
        assets = ["assets/goatcounter.js"]
    ),
    pages = [
        "index.md",
        "Manual" => ["basics.md", "models.md", "algorithms.md", "utilities.md"],
        "Examples" => map(example -> "$(example).md", examples),
        "internals.md",
        "migration.md",
        "_index.md",
    ],
)

deploydocs(repo = "github.com/maxmouchet/HMMBase.jl.git")
