using Glob
using Literate
using Random

config = Dict(
    :binder_root_url => "https://mybinder.org/v2/gh/maxmouchet/HMMBase.jl/master?filepath=",
    :repo_root_url => "https://github.com/maxmouchet/HMMBase.jl/blob/master"
)

function literate_documenter(inputdir, outputdir)
    Random.seed!(2019)
    map(glob("*.jl", inputdir)) do file
        Literate.markdown(file, outputdir, documenter = true; config...)
    end
end

function literate_notebooks(inputdir, outputdir)
    Random.seed!(2019)
    map(glob("*.jl", inputdir)) do file
        Literate.notebook(file, outputdir; config...)
    end
end

literate_documenter("examples/", "docs/src/examples/")
literate_notebooks("examples/", "examples/")
