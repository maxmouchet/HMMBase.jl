using Documenter
using Glob
using Literate

using Distributions
using HMMBase
using Random

# https://discourse.julialang.org/t/deactivate-plot-display-to-avoid-need-for-x-server/19359
ENV["GKSwstype"] = "nul"

Random.seed!(2019)

if "SKIP_EXAMPLES" in keys(ENV)
    find_examples() = []
else
    find_examples() = map(x -> joinpath("examples/", split(basename(x), ".")[1]), glob("*.jl", "examples/"))
end

for example in find_examples()
    Literate.markdown("$(example).jl", "docs/src/examples", documenter=true)
end

makedocs(
    sitename="HMMBase",
    modules=[HMMBase],
    pages = [
        "index.md",
        "Manual" => ["basics.md", "models.md", "algorithms.md", "utilities.md"],
        "Examples" => map(example -> "$(example).md", find_examples()),
        "internals.md",
        "migration.md",
        "_index.md"
    ]
)

deploydocs(
    repo = "github.com/maxmouchet/HMMBase.jl.git",
)
