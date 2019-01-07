Base.eval(:(have_color = true))
push!(LOAD_PATH, "src/")

using Documenter
using Distributions
using HMMBase

makedocs(
    sitename="HMMBase.jl",
    pages = [
        "index.md",
        # "Manual" => ["distributions.md", "samplers.md", "hmm.md"],
        "_index.md"
    ]
)

deploydocs(
    repo = "github.com/maxmouchet/HMMBase.jl.git",
)
