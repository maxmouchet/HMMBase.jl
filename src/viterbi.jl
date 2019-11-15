# Original implementations by @nantonel
# https://github.com/maxmouchet/HMMBase.jl/pull/6

# *log! methods use the samples log-likelihood instead of the likelihood.

function viterbi!(T1::AbstractMatrix, T2::AbstractMatrix, z::AbstractVector, a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix)
    @argcheck size(T1, 1) == size(T2, 1) == size(L, 1) == size(z, 1)
    @argcheck size(T1, 2) == size(T2, 2) == size(L, 2) == size(a, 1) == size(A, 1) == size(A, 2)

    T, K = size(L)
    (T == 0) && return

    fill!(T1, 0.0)
    fill!(T2, 0)

    c = 0.0

    for i in OneTo(K)
        T1[1,i] = a[i] * L[1,i]
        c += T1[1,i]
    end

    for i in OneTo(K)
        T1[1,i] /= c
    end

    @inbounds for t in 2:T
        c = 0.0

        for j in OneTo(K)
            # TODO: If there is NaNs in T1 this may
            # stay to 0 (NaN > -Inf == false).
            # Hence it will crash when computing z[t].
            # Maybe we should check for NaNs beforehand ?
            amax = 0
            vmax = -Inf

            for i in OneTo(K)
                v = T1[t-1,i] * A[i,j]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end

            T1[t,j] = vmax * L[t,j]
            T2[t,j] = amax
            c += T1[t,j]
        end

        for i in OneTo(K)
            T1[t,i] /= c
        end
    end

    z[T] = argmax(T1[T,:])
    for t in T-1:-1:1
        z[t] = T2[t+1,z[t+1]]
    end
end

function viterbilog!(T1::AbstractMatrix, T2::AbstractMatrix, z::AbstractVector, a::AbstractVector, A::AbstractMatrix, LL::AbstractMatrix)
    T, K = size(LL)
    (T == 0) && return

    fill!(T1, 0.0)
    fill!(T2, 0)

    al = log.(a)
    Al = log.(A)

    for i in OneTo(K)
        T1[1,i] = al[i] + LL[1,i]
    end

    @inbounds for t in 2:T
        for j in OneTo(K)
            amax = 0
            vmax = -Inf

            for i in OneTo(K)
                v = T1[t-1,i] + Al[i,j]
                if v > vmax
                    amax = i
                    vmax = v
                end
            end

            T1[t,j] = vmax + LL[t,j]
            T2[t,j] = amax
        end
    end

    z[T] = argmax(T1[T,:])
    for t in T-1:-1:1
        z[t] = T2[t+1,z[t+1]]
    end
end
