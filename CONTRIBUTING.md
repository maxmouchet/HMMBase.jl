
## Developing HMMBase

[Fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo).

```julia
pkg> dev git@github.com:...
```

Now when you do `using HMMBase` it will use the fork in `~/.julia/dev/HMMBase/...`.  
To reinstall the main version, type:
```julia
pkg> add HMMBase
```

```julia
pkg> test HMMBase
```

## Submitting changes


- Individual, working commits, that can be merged separately.

- Run tests
- Run benchmarks (compare)

```bash
julia --project=@. --check-bounds=yes -e "using Pkg; Pkg.test(\"HMMBase\");"
```

```julia
pkg> dev https://github.com/maxmouchet/HMMBase.jl.git
```

```julia
pkg> test HMMBase
# or
pkg> activate .
(HMMBase) pkg> test
```

```bash
julia --project=benchmark/ benchmark/make.jl
julia --project=docs/ docs/make.j
```