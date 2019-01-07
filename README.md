# HMMBase.jl

[![Build Status][travis-img]][travis-url] [![Documentation][docs-stable-img]][docs-stable-url]

## Usage

```julia
using Distributions
using HMMBase

hmm = HMM([0.99 0.005 0.005; 0.005 0.99 0.005; 0.05 0.05 0.9], [Normal(5,1), Normal(10,3), Normal(15,1)])
z, y = sample_hmm(hmm, 2500)
```

## Development

### Build docs

```bash
julia docs/make.jl
```

```julia
using Literate
Literate.markdown("examples/README.jl", "."; documenter=false)
```

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg?style=flat
[docs-stable-url]: https://maxmouchet.github.io/HMMBase.jl/stable

[travis-img]: https://travis-ci.org/maxmouchet/HMMBase.jl.svg?branch=master
[travis-url]: https://travis-ci.org/maxmouchet/HMMBase.jl

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

