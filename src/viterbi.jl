# Original implementations by @nantonel
# https://github.com/maxmouchet/HMMBase.jl/pull/6

function viterbilog!(
    T1::AbstractMatrix,
    T2::AbstractMatrix,
    z::AbstractVector,
    a::AbstractVector,
    A::AbstractMatrix,
    LL::AbstractMatrix,
)
    T, K = size(LL)
    (T == 0) && return

    fill!(T1, 0.0)
    fill!(T2, 0)

    al = log.(a)
    Al = log.(A)

    for i in OneTo(K)
        T1[1, i] = al[i] + LL[1, i]
    end

    @inbounds for t = 2:T
        for j in OneTo(K)
            amax = 0
            vmax = -Inf

            for i in OneTo(K)
                v = T1[t-1, i] + Al[i, j]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end

            T1[t, j] = vmax + LL[t, j]
            T2[t, j] = amax
        end
    end

    z[T] = argmax(T1[T, :])
    for t = T-1:-1:1
        z[t] = T2[t+1, z[t+1]]
    end
end

"""
    viterbi(a, A, L) -> Vector

Find the most likely hidden state sequence, see [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm).
"""
function viterbi(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix, logl = nothing)
    (logl !== nothing) && deprecate_kwargs("logl")
    T1 = Matrix{Float64}(undef, size(L))
    T2 = Matrix{Int}(undef, size(L))
    z = Vector{Int}(undef, size(L, 1))
    viterbilog!(T1, T2, z, a, A, L)
    z
end

"""
    viterbi(hmm, observations; robust) -> Vector

**Example**
```julia
using Distributions, HMMBase
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, 1000)
zv = viterbi(hmm, y)
```
"""
function viterbi(hmm::AbstractHMM, observations; logl = nothing, robust = false)
    (logl !== nothing) && deprecate_kwargs("logl")
    LL = loglikelihoods(hmm, observations; robust = robust)
    viterbi(hmm.a, hmm.A, LL)
end
