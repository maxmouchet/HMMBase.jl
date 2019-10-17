# Original implementations by @nantonel
# https://github.com/maxmouchet/HMMBase.jl/pull/6

# In-place forward pass, where α and c are allocated beforehand.
function forward!(α::AbstractMatrix, c::AbstractVector, a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix)
    T, K = size(L)

    fill!(α, 0.0)
    fill!(c, 0.0)

    for j in Base.OneTo(K)
        α[1,j] = a[j]*L[1,j]
        c[1] += α[1,j]
    end

    for j in Base.OneTo(K)
        α[1,j] /= c[1]
    end

    for t in 2:T
        for j in Base.OneTo(K)
            for i in Base.OneTo(K)
                α[t,j] += α[t-1,i] * A[i,j] * L[t,j]
            end
            c[t] += α[t,j]
        end

        for j in Base.OneTo(K)
            α[t,j] /= c[t]
        end
    end
end

# In-place backward pass, where β and c are allocated beforehand.
function backward!(β::AbstractMatrix, c::AbstractVector, a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix)
    T, K = size(L)

    fill!(β, 0.0)
    fill!(c, 0.0)

    for j in Base.OneTo(K)
        β[end,j] = 1.0
    end

    for t in T-1:-1:1
        for j in Base.OneTo(K)
            for i in Base.OneTo(K)
                β[t,j] += β[t+1,i] * A[j,i] * L[t+1,i]
            end
            c[t+1] += β[t,j]
        end

        for j in Base.OneTo(K)
            β[t,j] /= c[t+1]
        end
    end

    for j in Base.OneTo(K)
        c[1] += a[j] * L[1,j] * β[1,j]
    end
end

# In-place posterior computation, where γ is allocated beforehand.
function posteriors!(γ::AbstractMatrix, α::AbstractMatrix, β::AbstractMatrix)
    T, K = size(α)
    for t in Base.OneTo(T)
        c = 0.0
        for i = Base.OneTo(K)
            γ[t,i] = α[t,i] * β[t,i]
            c += γ[t,i]
        end

        for i in Base.OneTo(K)
            γ[t,i] /= c
        end
    end
end
