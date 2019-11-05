
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