# Same methods as in `viterbi.jl` but using the
# samples log-likelihood instead of the likelihood.

function viterbilog!(T1::AbstractMatrix, T2::AbstractMatrix, z::AbstractVector, a::AbstractVector, A::AbstractMatrix, LL::AbstractMatrix)
    T, K = size(LL)
    (T == 0) && return

    fill!(T1, 0.0)
    fill!(T2, 0)

    m = vec_maximum(view(LL, 1, :))
    c = 0.0

    for i in OneTo(K)
        T1[1,i] = a[i] * exp(LL[1,i] - m)
        c += T1[1,i]
    end

    for i in OneTo(K)
        T1[1,i] /= c
    end

    @inbounds for t in 2:T
        m = vec_maximum(view(LL, t, :))
        c = 0.0

        for j in OneTo(K)
            # TODO: See comment in viterbi.jl
            amax = 0
            vmax = -Inf

            for i in OneTo(K)
                v = T1[t-1,i] * A[i,j]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end

            T1[t,j] = vmax * exp(LL[t,j] - m)
            T2[t,j] = amax
            c += T1[t,j]
        end

        for i in OneTo(K)
            T1[t,i] /= c
        end
    end

    z[T] = argmax(T1[T,:])
    @inbounds for t in T-1:-1:1
        z[t] = T2[t+1,z[t+1]]
    end
end
