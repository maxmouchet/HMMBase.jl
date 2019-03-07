# HMMBase

*Hidden Markov Models for Julia.*

| **Documentation**                       | **Build Status**              |
|:---------------------------------------:|:-----------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url]| [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] |

## Introduction

HMMBase builds upon [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) to provide a lightweight and efficient abstraction for hidden Markov models in Julia.  
It support arbitrary univariate and multivariate distributions and implements the forward and backward recursions, the Viterbi algorithm, and the MLE estimator.  
More advanced models, such as Bayesian HMMs can be built upon HMMBase.

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> registry add https://github.com/maxmouchet/JuliaRegistry.git
pkg> add HMMBase
```

## Documentation

- [**STABLE**][docs-stable-url] &mdash; **documentation of the most recently tagged version.**
- [**DEVEL**][docs-dev-url] &mdash; *documentation of the in-development version.*

## Project Status

The package is tested against Julia 1.0 and the nightly builds of the Julia `master` branch on Linux and macOS.

## Questions and Contributions

Contributions are very welcome, as are feature requests and suggestions. Please open an [issue][issues-url] if you encounter any problems.

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg?style=flat
[docs-stable-url]: https://maxmouchet.github.io/HMMBase.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg?style=flat
[docs-dev-url]: https://maxmouchet.github.io/HMMBase.jl/dev

[travis-img]: https://travis-ci.org/maxmouchet/HMMBase.jl.svg?branch=master
[travis-url]: https://travis-ci.org/maxmouchet/HMMBase.jl

[codecov-img]: https://codecov.io/github/maxmouchet/HMMBase.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/github/maxmouchet/HMMBase.jl?branch=master

[issues-url]: https://github.com/maxmouchet/HMMBase.jl/issues
