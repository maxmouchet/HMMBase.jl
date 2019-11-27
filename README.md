<p align="center">
  <img src="/docs/src/assets/logo.png" height="150"><br/>
  <i>Hidden Markov Models for Julia.</i><br/><br/>
  <a href="https://maxmouchet.github.io/HMMBase.jl/stable">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg?style=flat">
  </a>
  <a href="https://github.com/maxmouchet/HMMBase.jl/actions">
    <img src="https://github.com/maxmouchet/HMMBase.jl/workflows/CI/badge.svg">
  </a>
  <a href="https://codecov.io/github/maxmouchet/HMMBase.jl?branch=master">
    <img src="https://codecov.io/github/maxmouchet/HMMBase.jl/coverage.svg?branch=master">
  </a>
</p>

## News

- _v1.1 (dev) :_ add integration with [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl).
- _**v1.0 (stable) :**_ HMMBase v1.0 comes with many new features and performance improvements (see the [release notes](https://github.com/maxmouchet/HMMBase.jl/releases/tag/v1.0.0)), thanks to [@nantonel PR#6](https://github.com/maxmouchet/HMMBase.jl/pull/6).
It also introduces breaking API changes (method and fields renaming), see [Migration to v1.0](https://maxmouchet.github.io/HMMBase.jl/dev/migration/) for details on migrating your code to the new version.
- _v0.0.14 :_ latest pre-release version.

Are you using HMMBase in a particular domain (Biology, NLP, ...) ? Feel free to open an issue to discuss you workflow/needs and see how we can improve HMMBase.

## Introduction

HMMBase provides a lightweight and efficient abstraction for hidden Markov models in Julia. Most HMMs libraries only support discrete (e.g. categorical) or Normal distributions. In contrast HMMBase builds upon [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) to support arbitrary univariate and multivariate distributions.  
See [HMMBase.jl - A lightweight and efficient Hidden Markov Model abstraction](https://discourse.julialang.org/t/ann-hmmbase-jl-a-lightweight-and-efficient-hidden-markov-model-abstraction/21604) for more details on the motivation behind this package.

<p align="center">
  <img src="/benchmark/benchmark_summary.png" width="640"><br/>
  <a href="/benchmark">Benchmark</a> of HMMBase against <a href="https://github.com/hmmlearn/hmmlearn">hmmlearn</a> and <a href="https://github.com/mattjj/pyhsmm">pyhsmm</a>.<br/>(log) stands for "using log-likelihoods".
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

The package is tested against Julia 1.0 and Julia 1.2.  

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
