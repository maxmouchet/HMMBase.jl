## Contributing to HMMBase

Contributions to HMMBase are welcome!  
Please try to submit short pull requests with a well-defined scope.

### Development

1. To make changes, start by [forking](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the repository.
2. Then, clone the package for development:

```julia
pkg> dev git@github.com:USERNAME/HMMBase.jl.git
```

3. From now on, `using HMMBase` will use your fork in `~/.julia/dev/HMMBase/`.  
(To re-install the main version, use `pkg> add HMMBase` in the Julia REPL.

4. Make changes as desired, and ensure that the tests are passing:
```bash
pkg> test HMMBase
```

5. Commit your changes and submit the PR!

### Benchmarks and documentation

Compiler calls: JULIA_DEBUG=loading

```bash
julia --project=@. benchmark/make.jl
julia --project=@. docs/make.jl
```

### Internals

See [Internals](https://maxmouchet.github.io/HMMBase.jl/dev/internals/) in the documentation.