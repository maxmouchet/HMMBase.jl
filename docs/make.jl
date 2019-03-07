Base.eval(:(have_color = true))

using Documenter
using Glob
using Literate

using Distributions
using HMMBase
using Random

Random.seed!(2019)

find_examples() = map(x -> joinpath("examples/", split(basename(x), ".")[1]), glob("*.jl", "examples/"))

if !("SKIP_EXAMPLES" in keys(ENV))
    for example in find_examples()
        Literate.markdown("$(example).jl", "docs/src/examples", documenter=true)
    end
end

makedocs(
    sitename="HMMBase.jl",
    pages = [
        "index.md",
        "Manual" => ["hmm.md", "inference.md", "sampling.md"],
        "Examples" => map(example -> "$(example).md", find_examples()),
        "_index.md"
    ]
)

deploydocs(
    repo = "github.com/maxmouchet/HMMBase.jl.git",
)
