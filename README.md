<p align="center">
  <img src="/docs/src/assets/logo.png" height="150"><br/>
  <i>Hidden Markov Models for Julia.</i><br/><br/>
  <a href="[https://github.com/maxmouchet/HMMBase.jl/actions](https://github.com/maxmouchet/HMMBase.jl/actions/workflows/ci.yml)">
    <img src="https://img.shields.io/github/actions/workflow/status/maxmouchet/HMMBase.jl/ci.yml?logo=github">
  </a>
  <a href="https://codecov.io/github/maxmouchet/HMMBase.jl?branch=master">
    <img src="https://img.shields.io/codecov/c/github/maxmouchet/HMMBase.jl?logo=codecov&logoColor=white">
  </a>
  <a href="https://maxmouchet.github.io/HMMBase.jl/stable">
    <img src="https://img.shields.io/badge/documentation-online-blue.svg?logo=Julia&logoColor=white">
  </a>
</p>

## Status

HMMBase is not maintained anymore. It will keep being available as a Julia package but we encourage existing and new users to migrate to [HiddenMarkovModels.jl](https://github.com/gdalle/HiddenMarkovModels.jl) which offers a similar interface. For more information see [HiddenMarkovModels.jl: when did HMMs get so fast?](https://discourse.julialang.org/t/ann-hiddenmarkovmodels-jl-when-did-hmms-get-so-fast/100191).

## Introduction

HMMBase provides a lightweight and efficient abstraction for hidden Markov models in Julia. Most HMMs libraries only support discrete (e.g. categorical) or Normal distributions. In contrast HMMBase builds upon [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) to support arbitrary univariate and multivariate distributions.  
See [HMMBase.jl - A lightweight and efficient Hidden Markov Model abstraction](https://discourse.julialang.org/t/ann-hmmbase-jl-a-lightweight-and-efficient-hidden-markov-model-abstraction/21604) for more details on the motivation behind this package.

<p align="center">
  <img src="/benchmark/benchmark_summary.png" width="640"><br/>
  <a href="/benchmark">Benchmark</a> of HMMBase against <a href="https://github.com/hmmlearn/hmmlearn">hmmlearn</a> and <a href="https://github.com/mattjj/pyhsmm">pyhsmm</a>.
</p>

**Features:**
- Supports any observation distributions conforming to the [Distribution](https://juliastats.org/Distributions.jl/latest/types/) interface.
- Fast and stable implementations of the forward/backward, EM (Baum-Welch) and Viterbi algorithms.

**Non-features:**
- Multi-sequences HMMs, see [MS_HMMBase](https://github.com/mmattocks/MS_HMMBase.jl)
- Bayesian models, probabilistic programming, see [Turing](https://github.com/TuringLang/Turing.jl)
- Nonparametric models (HDP-H(S)MM, ...)

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add HMMBase
```

## Documentation

- [**STABLE**][docs-stable-url] &mdash; **documentation of the most recently tagged version.**
- [**DEVEL**][docs-dev-url] &mdash; *documentation of the in-development version.*

## Project Status

The package is tested against Julia 1.0 and the latest Julia 1.x.  

Starting with v1.0, we follow [semantic versioning]():

> Given a version number MAJOR.MINOR.PATCH, increment the:
> 1. MAJOR version when you make incompatible API changes,
> 2. MINOR version when you add functionality in a backwards compatible manner, and
> 3. PATCH version when you make backwards compatible bug fixes.

## Questions and Contributions

Contributions are very welcome, as are feature requests and suggestions. Please read the [CONTRIBUTING.md](/CONTRIBUTING.md) file for informations on how to contribute. Please open an [issue][issues-url] if you encounter any problems.

*Logo: lego by jon trillana from the Noun Project.*

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg?style=flat
[docs-stable-url]: https://maxmouchet.github.io/HMMBase.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg?style=flat
[docs-dev-url]: https://maxmouchet.github.io/HMMBase.jl/dev

[issues-url]: https://github.com/maxmouchet/HMMBase.jl/issues
