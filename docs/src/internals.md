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
│   ├── *.jl            # Public interface and internal in-place implementations
└── test
    ├── deprecated.jl   # Deprecation tests
    ├── integration.jl  # Integration tests
    ├── pyhsmm.jl       # Python tests
    ├── runtests.jl     # Integration+Unit tests runner
    └── unit.jl         # Unit tests

```

## In-place versions

Internally HMMBase uses in-place implementations for most of the algorithms.

In-place                          | Public interface
:---------------------------------|:----------------
`loglikelihoods!`                 | `loglikelihoods`
`forwardlog!`                     | `forward`
`backwardlog!`                    | `backward`
`posteriors!`                     | `posteriors`
`viterbilog!`                     | `viterbi`
`fit_mle!`                        | `fit_mle`
