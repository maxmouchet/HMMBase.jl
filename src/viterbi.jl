# Original implementations by @nantonel
# https://github.com/maxmouchet/HMMBase.jl/pull/6
function viterbilog!(
    T1::AbstractArray,
    T2::AbstractArray,
    z::AbstractMatrix,
    a::AbstractVector,
    A::AbstractMatrix,
    LL::AbstractArray,
)
    T, K, N = size(LL)
    ((T == 0)||(N == 0)) && return

    fill!(T1, 0.0)
    fill!(T2, 0)

    al = log.(a)
    Al = log.(A)
    for n in OneTo(N)
        T = length(filter(!isnothing, LL[:, 1, n]))
        for i in OneTo(K)
            T1[1, i, n] = al[i] + LL[1, i, n]
        end
        @inbounds for t = 2:T
            for j in OneTo(K)
                amax = 0
                vmax = -Inf

                for i in OneTo(K)
                    v = T1[t-1, i, n] + Al[i, j]
                    if v > vmax
                        amax = i
                        vmax = v
                    end
                end

                T1[t, j, n] = vmax + LL[t, j, n]
                T2[t, j, n] = amax
            end
        end

        z[T, n] = argmax(T1[T, :, n])
        println(T)
        for t = T-1:-1:1
            z[t, n] = T2[t+1, z[t+1, n], n]
        end
    end
end

"""
    viterbi(a, A, LL) -> Vector

Find the most likely hidden state sequence, see [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm).
"""
function viterbi(a::AbstractVector, A::AbstractMatrix, LL::AbstractArray; logl = nothing)
    ## < v1.1 compatibility
    (logl !== nothing) && deprecate_kwargs("logl")
    (logl == false) && (LL = log.(LL))
    ## --------------------
    T1 = Array{Float64}(undef, size(LL))
    T2 = Array{Int}(undef, size(LL))
    z = Matrix{Union{Int,Nothing}}(nothing, size(LL, 1), size(LL, 3))
    viterbilog!(T1, T2, z, a, A, LL)
    z, LL
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