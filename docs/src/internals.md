# Internals

Overview of the repository structure:

```
.
├── benchmark
│   ├── benchmarks.jl   # Benchmark suite definition
│   └── make.jl         # Benchmark runner
├── docs
│   ├── src             # Documentation source
│   └── make.jl         # Documentation builder
├── examples            # Examples (included in the documentation)
├── src
│   ├── HMMBase.jl      # Main module file
│   ├── hmm.jl          # HMM type, rand, size, ...
│   ├── *_api.jl        # Public interfaces
│   ├── *.jl            # Internal in-place implementations
└── test
    ├── integration.jl  # Integration tests
    ├── pyhsmm.jl       # Python tests
    ├── runtests.jl     # Integration+Unit tests runner
    └── unit.jl         # Unit tests

```

## In-place versions

Internally HMMBase uses in-place implementations for most of the algorithms.

Public interfaces are defined in `_api.jl` files, and are responsible for copying
user provided data.

**TODO:** Add table with in-place / generated correspondence.
