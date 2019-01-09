# HMM Type

```@docs
HMM(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution{F}}) where F
assert_hmm(π::Matrix{Float64}, π0::Vector{Float64}, D::Vector{<:Distribution})
sample_hmm(hmm::HMM{Univariate}, timesteps::Int)
```
