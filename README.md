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

## Introduction

HMMBase provides a lightweight and efficient abstraction for hidden Markov models in Julia. Most HMMs libraries only support discrete (e.g. categorical) or normal distributions. In contrast HMMBase builds upon [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) to support arbitrary univariate and multivariate distributions.  

The goal is to provide well-tested and fast implementations of the basic HMMs algorithms such as the forward-backward algorithm, the Viterbi algorithm, and the MLE estimator. More advanced models, such as Bayesian HMMs, can be built upon HMMBase.

See [HMMBase.jl - A lightweight and efficient Hidden Markov Model abstraction](https://discourse.julialang.org/t/ann-hmmbase-jl-a-lightweight-and-efficient-hidden-markov-model-abstraction/21604) for more details on the motivation behind this package.

<img src="https://github.com/maxmouchet/HMMBase.jl/blob/master/benchmark/benchmark_summary.png" width="480">

## Migrating to v1.0

HMMBase v1.0 will be released before the end of the year, and contains breaking API changes.  
Many methods have been renamed, and most importantly, the fields of the `HMM` structure have
been renamed from `π0, π, D` to `a, A, B`.

See the [Migrating to v1.0](https://maxmouchet.github.io/HMMBase.jl/dev/migration/) section of the documentation for more informations.
<!-- The [release notes]() contains a detailed list of the changes. -->

You can try the new version before it is released.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add HMMBase#master
```

If after the release you'd like to temporarily keep the old version, add the following in your `Project.toml`:

```toml
[compat]
HMMBase = "0.0.14"
```

Note that HMMBase v0.0.x will not be maintained anymore.  
Starting from v1.0, we will follow [semantic versioning]():

> Given a version number MAJOR.MINOR.PATCH, increment the:
> 1. MAJOR version when you make incompatible API changes,
> 2. MINOR version when you add functionality in a backwards compatible manner, and
> 3. PATCH version when you make backwards compatible bug fixes.

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

## Questions and Contributions

Contributions are very welcome, as are feature requests and suggestions. Please open an [issue][issues-url] if you encounter any problems.

*Logo: lego by jon trillana from the Noun Project.*

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg?style=flat
[docs-stable-url]: https://maxmouchet.github.io/HMMBase.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg?style=flat
[docs-dev-url]: https://maxmouchet.github.io/HMMBase.jl/dev

[issues-url]: https://github.com/maxmouchet/HMMBase.jl/issues
