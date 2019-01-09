# HMMBase.jl

[![Build Status][travis-img]][travis-url] [![Documentation][docs-latest-img]][docs-latest-url]

HMMBase builds upon [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) to provide a lightweight and efficient abstraction for hidden Markov models in Julia.
It implements the forward and backward recursions, the Viterbi algorithm, and the MLE estimator.
More advanced models, such as Bayesian HMMs can be built upon HMMBase.

## Usage

You can run the following example by executing `julia examples/README.jl`.

```julia
using Distributions
using HMMBase

hmm = HMM([0.99 0.005 0.005; 0.005 0.99 0.005; 0.05 0.05 0.9], [Normal(5,1), Normal(10,3), Normal(15,1)])
z, y = sample_hmm(hmm, 2500)

α = messages_forward(hmm, y)
β = messages_backward(hmm, y)
γ = forward_backward(hmm, y)
```

TODO: MLE

## Development

### Build docs

```bash
julia docs/make.jl
```

```julia
using Literate
Literate.markdown("examples/README.jl", "."; documenter=false)
```

## Maintainers

[@maxmouchet](https://github.com/maxmouchet)

## Contribute

PRs accepted.

## License

MIT © 2018 Maxime Mouchet

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg?style=flat
[docs-stable-url]: https://maxmouchet.github.io/HMMBase.jl/stable

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg?style=flat
[docs-latest-url]: https://maxmouchet.github.io/HMMBase.jl/latest

[travis-img]: https://travis-ci.org/maxmouchet/HMMBase.jl.svg?branch=master
[travis-url]: https://travis-ci.org/maxmouchet/HMMBase.jl

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

