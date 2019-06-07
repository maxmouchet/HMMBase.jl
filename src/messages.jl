function baum_forward!(alpha, hmm, L)
  fill!(alpha, 0.0)

  T, S = size(L,1), size(hmm.A,1)
  for j = 1:S
    alpha[1,j] = hmm.a[j]*L[1,j]
  end

  for t = 2:T
    for j = 1:S
      for i = 1:S
        alpha[t,j] += alpha[t-1,i]*hmm.A[i,j]*L[t,j] 
      end
    end
  end
  return alpha

end

function baum_backward!(beta, hmm, L)
  fill!(beta, 0.0)

  T, S = size(L,1), size(hmm.A,1)
  for j = 1:S
    beta[end,j] = 1.0
  end

  for t = T-1:-1:1
    for j = 1:S
      for k = 1:S
        beta[t,j] += beta[t+1,k]*hmm.A[j,k]*L[t+1,k]
      end
    end
  end
  return beta

end

"""
    forward(hmm, L)

Compute forward probabilities `alpha` (`Nt`x`Ns` matrix) and the scaling vector `c` (`Nt` long vector).

`L` must be a `Nt`x`Ns` matrix containing the likelihoods (see [`likelihood`](@ref)) 

"""
function forward(hmm,L)
  T, S = size(L,1), size(hmm.A,1)
  alpha = zeros(T,S)
  c = zeros(T)
  forward!(alpha,c,hmm,L)
end

"""
    forward!(alpha, c, hmm, L)

In-place version of [`forward`](@ref).

"""
function forward!(alpha, c, hmm, L)

  fill!(alpha, 0.0)
  fill!(c, 0.0)

  T, S = size(L,1), size(hmm.A,1)

  for j = 1:S
    alpha[1,j] = hmm.a[j]*L[1,j]
    c[1] += alpha[1,j]
  end

  # normalize
  for j = 1:S
    alpha[1,j] /= c[1]
  end

  for t = 2:T
    for j = 1:S
      for i = 1:S
        alpha[t,j] += alpha[t-1,i]*hmm.A[i,j]*L[t,j] 
      end
      c[t] += alpha[t,j]
    end

    # normalize
    for j = 1:S
      alpha[t,j] /= c[t]
    end

  end
  return alpha, c

end

"""
    backward(hmm, L, c)

Compute backward probabilities `beta` (`Nt`x`Ns` matrix) using the scaling vector `c` (`Nt` long vector).

`L` must be a `Nt`x`Ns` matrix containing the likelihoods (see [`likelihood`](@ref)) 

"""
backward

"""
    backward!(beta, hmm, L, c)

In-place version of [`backward`](@ref).

"""
function backward!(beta, hmm, L, c)

  fill!(beta, 0.0)

  T, S = size(L,1), size(hmm.A,1)
  for j = 1:S
    beta[end,j] = 1.0
  end

  for t = T-1:-1:1
    for j = 1:S
      for k = 1:S
        beta[t,j] += beta[t+1,k]*hmm.A[j,k]*L[t+1,k]
      end
    end

    # normalize
    for j = 1:S
      beta[t,j] /= c[t+1]
    end
  end
  return beta

end

# only for testing
function backward!(beta, hmm, L)

  fill!(beta, 0.0)

  T, S = size(L,1), size(hmm.A,1)
  for j = 1:S
    beta[end,j] = 1.0
  end

  for t = T-1:-1:1
    c = 0.0
    for j = 1:S
      for k = 1:S
        beta[t,j] += beta[t+1,k]*hmm.A[j,k]*L[t+1,k]
      end
      c += beta[t,j]
    end

    # normalize
    for j = 1:S
      beta[t,j] /= c
    end
  end
  return beta

end

"""
    posteriors(alpha,beta)

Compute posterior probabilities `gamma` (`Nt`x`Ns` matrix) using `alpha` and `beta`.

"""
function posteriors(alpha,beta)
  gamma = zeros(eltype(alpha), size(alpha))
  posteriors!(gamma, alpha, beta)
  return gamma
end

"""
    posteriors!(gamma, alpha, beta)

In-place version of [`posteriors`](@ref).

"""
function posteriors!(gamma, alpha, beta)
  Nt, Ns = size(alpha)
  for t = 1:Nt
    c = 0.0
    for i = 1:Ns
      gamma[t,i] = alpha[t,i] * beta[t,i]
      c += gamma[t,i]
    end
    for i = 1:Ns
      gamma[t,i] /= c
    end
  end
  return gamma
end

for f in [
          :baum_forward => :baum_forward!,
          :baum_backward => :baum_backward!,
          :backward => :backward!,
         ]

  @eval begin

    function $(f[1])(hmm, L, args...)

      T, S = size(L,1), size(hmm.A,1)
      message = zeros(T,S)
      $(f[2])(message, hmm, L, args...)

    end
  end

end

"""
    viterbi(hmm, L; normalize=true)

Compute the best sequence of states using the observation likelihood `L` using the Viterbi algorithm.

`L` must be a `Nt`x`Ns` matrix containing the likelihoods (see [`likelihood`](@ref)) 

`normalize` specifies if normalization is applied.

"""
function viterbi(hmm,L; normalize=true)
  T, S = size(L,1), size(hmm.A,1)
  v = zeros(T,S)
  argmax_v = zeros(Int,T,S)
  if normalize
    bp = nviterbi!(v,argmax_v, hmm, L)
  else
    bp = viterbi!(v, argmax_v, hmm, L)
  end
  return bp
end

"""
    viterbi!(v, argmax_v, beta)

In-place version of [`viterbi`](@ref) without normalization.

`v` and `argmax_v` are `Nt`x`Ns` matrices of the score and backtracking indicies respectively.

"""
function viterbi!(v, argmax_v, hmm, L)

  fill!(v,0.0)
  fill!(argmax_v,0)

  T, S = size(L,1), size(hmm.A,1)
  v = zeros(T,S)
  argmax_v = zeros(Int,T,S)
  for s = 1:S
    v[1,s] = hmm.a[s]*L[1,s]
  end

  for t = 2:T   
    for s = 1:S

      argmax_vi = -1 
      vmax      = -Inf

      for ss = 1:S
        v_t = v[t-1,ss] * hmm.A[ss,s]  
        if v_t > vmax
          argmax_vi = ss 
          vmax = v_t
        end
      end

      v[t,s] = vmax * L[t,s] 
      argmax_v[t,s] = argmax_vi

    end

  end

  bp = zeros(Int,T)
  bp[T] = argmax(v[end,:])
  for t in T-1:-1:1
    bp[t] = argmax_v[t+1,bp[t+1]]
  end

  return bp
end

"""
    nviterbi!(v, argmax_v, beta)

In-place version of [`viterbi`](@ref) using normalization.

`v` and `argmax_v` are `Nt`x`Ns` matrices of the score and backtracking indicies respectively.

"""
function nviterbi!(v, argmax_v, hmm, L)

  fill!(v,0.0)
  fill!(argmax_v,0)

  T, S = size(L,1), size(hmm.A,1)
  v = zeros(T,S)
  argmax_v = zeros(Int,T,S)
  c = 0.0
  for s = 1:S
    v[1,s] = hmm.a[s]*L[1,s]
    c += v[1,s]
  end
  # normalize
  for s = 1:S
    v[1,s] /= c
  end

  for t = 2:T   
    c = 0
    for s = 1:S

      argmax_vi = -1 
      vmax      = -Inf

      for ss = 1:S
        v_t = v[t-1,ss] * hmm.A[ss,s]  
        if v_t > vmax
          argmax_vi = ss 
          vmax = v_t
        end
      end

      v[t,s] = vmax * L[t,s] 
      c += v[t,s]
      argmax_v[t,s] = argmax_vi

    end

    # normalize
    for s = 1:S
      v[t,s] /= c
    end

  end

  bp = zeros(Int,T)
  bp[T] = argmax(v[end,:])
  for t in T-1:-1:1
    bp[t] = argmax_v[t+1,bp[t+1]]
  end

  return bp
end
