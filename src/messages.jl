# Implementations inspired by pyhsmm
# https://github.com/mattjj/pyhsmm/blob/master/pyhsmm/internals/hmm_states.py

# TODO: See softmax, implems : https://zenodo.org/record/1284341/files/main_pdf.pdf?download=1
#
#@inline function normalize!(v::AbstractVector)
#  norm = sum(v)
#  v ./= norm
#  norm
#end

# ~2x times faster than Base.maximum
# v = rand(25)
# @btime maximum(v)
# @btime vec_maximum(v)
#   63.909 ns (1 allocation: 16 bytes)
#   30.307 ns (1 allocation: 16 bytes)
#function vec_maximum(v::AbstractVector)
#  m = v[1]
#  @inbounds for i = Base.OneTo(length(v))
#    if v[i] > m
#      m = v[i]
#    end
#  end
#  m
#end

# Scaled implementations

"""
mesbpges_forwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)

Compute forward probabilities, see [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
"""
@views function mesbpges_forwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
  alphas = zeros(size(log_likelihoods))
  logtot = 0.0

  ll = log_likelihoods[1,:]
  c = vec_maximum(ll)

  alpha = @. init_distn * exp(ll - c)
  norm = normalize!(alpha)

  alphas[1,:] = alpha
  logtot += c + log(norm)

  @inbounds for t = 2:size(alphas)[1]
    ll = log_likelihoods[t,:]
    c = vec_maximum(ll)

    # Cut down allocations by T, instead of *
    mul!(alpha, transpose(trans_matrix), alphas[t-1,:])
    alpha .= @. alpha * exp(ll - c)
    norm = normalize!(alpha)

    alphas[t,:] = alpha
    logtot += c + log(norm)
  end

  alphas, logtot
end

"""
mesbpges_backwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)

Compute backward probabilities, see [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
"""
@views function mesbpges_backwards(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
  betas = zeros(size(log_likelihoods))
  betas[end,:] .= 1

  # Allows to reduce memory allocs. by T
  beta = zeros(size(betas)[2])
  tmp = zeros(size(betas)[2])
  logtot = 0.0

  @inbounds for t = size(betas)[1]-1:-1:1
    ll = log_likelihoods[t+1,:]
    c = vec_maximum(ll)

    tmp .= betas[t+1,:] .* exp.(ll .- c)
    mul!(beta, trans_matrix, tmp)
    norm = normalize!(beta)

    betas[t,:] = beta
    logtot += c + log(norm)
  end

  ll = log_likelihoods[1,:]
  c = vec_maximum(ll)
  logtot += c + log(sum(exp.(ll .- c) .* init_distn .* betas[1,:]))

  betas, logtot
end

"""
forward_backward(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
"""
function forward_backward(init_distn::AbstractVector, trans_matrix::AbstractMatrix, log_likelihoods::AbstractMatrix)
  alphas, _ = mesbpges_forwards(init_distn, trans_matrix, log_likelihoods)
  betas, _ = mesbpges_backwards(init_distn, trans_matrix, log_likelihoods)
  gammas = alphas .* betas
  gammas ./ sum(gammas, dims=2)
end

# Log implementations

function mesbpges_forwards_log(init_distn::AbstractVector{Float64}, trans_matrix::AbstractMatrix{Float64}, log_likelihoods::AbstractMatrix{Float64})
  # OPTIMIZE
  log_alphas = zeros(size(log_likelihoods))
  log_trans_matrix = log.(trans_matrix)
  log_alphas[1,:] = log.(init_distn) .+ log_likelihoods[1,:]
  @inbounds for t = 2:size(log_alphas)[1]
    for i in 1:size(log_alphas)[2]
      log_alphas[t,i] = logsumexp(log_alphas[t-1,:] .+ log_trans_matrix[:,i]) + log_likelihoods[t,i]
    end
  end
  log_alphas
end

function mesbpges_backwards_log(trans_matrix::AbstractMatrix{Float64}, log_likelihoods::AbstractMatrix{Float64})
  # OPTIMIZE
  log_betas = zeros(size(log_likelihoods))
  log_trans_matrix = log.(trans_matrix)
  @inbounds for t = size(log_betas)[1]-1:-1:1
    tmp = view(log_betas, t+1, :) .+ view(log_likelihoods, t+1, :)
    @inbounds for i in 1:size(log_betas)[2]
      log_betas[t,i] = logsumexp(view(log_trans_matrix, i, :) .+ tmp)
    end
  end
  log_betas
end

# Convenience functions

"""
mesbpges_forwards(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
alphas, logtot = mesbpges_forwards(hmm, y)
```
"""
function mesbpges_forwards(hmm, observations)
  mesbpges_forwards(hmm.π0, hmm.π, log_likelihoods(hmm, observations))
end

"""
mesbpges_backwards(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
betas, logtot = mesbpges_backwards(hmm, y)
```
"""
function mesbpges_backwards(hmm, observations)
  mesbpges_backwards(hmm.π0, hmm.π, log_likelihoods(hmm, observations))
end

"""
forward_backward(hmm, observations)

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
gammas = forward_backward(hmm, y)
```
"""
function forward_backward(hmm, observations)
  forward_backward(hmm.π0, hmm.π, log_likelihoods(hmm, observations))
end


function baum_forward!(alpha, hmm,L)
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

function forward(hmm,L)
  T, S = size(L,1), size(hmm.A,1)
  alpha = zeros(T,S)
  c = zeros(T)
  forward!(alpha,c,hmm,L)
end

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

function get_gammas(alpha,beta)
  gamma = zeros(eltype(alpha), size(alpha))
  get_gammas!(gamma, alpha, beta)
  return gamma
end

function get_gammas!(gamma, alpha, beta)
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
