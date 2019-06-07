function update_A!(hmm, epsilon, alpha, beta, gamma, L)

  Nt = size(beta,1)
  Ns = size(hmm.A,1)

  for t = 1:Nt-1
    c = 0.0
    for i = 1:Ns, j = 1:Ns
      epsilon[t, i, j] = alpha[t,i]*hmm.A[i,j]*L[t+1,j]*beta[t+1,j]
      c += epsilon[t, i, j]
    end
    for i = 1:Ns, j = 1:Ns
      epsilon[t, i, j] /= c 
    end
  end

  fill!(hmm.A,0.0)
  c = sum(gamma[1:Nt-1,:], dims=1)

  for i = 1:Ns
    for j = 1:Ns
      for t = 1:Nt-1
        hmm.A[i,j] += epsilon[t, i, j] 
      end
      hmm.A[i,j] /= c[i]
    end
  end

end

function update_a!(hmm, gamma)
  Ns = size(hmm.A,1)
  for i = 1:Ns
    hmm.a[i] = gamma[1,i]
  end
end

function update_B!(hmm::HMM{S,O,T,V,M,D}, gamma, y) where {S,O,T,V,M,
                                                           D <: Array{<:Categorical} }
  Nt = length(y)
  Ns = length(hmm.B)

  for j = 1:Ns
    fill!(hmm.B[j].p,0.0)
  end

  for j = 1:Ns, k = 1:Ns
    for t = 1:Nt
      if y[t] == k 
        hmm.B[j].p[k] += gamma[t,j]
      end
    end
  end

  for j = 1:Ns
    c = 0.0
    for t = 1:Nt
      c += gamma[t,j]
    end
    for k = 1:Ns
      hmm.B[j].p[k] /= c
    end
  end

end

function update_B!(hmm::HMM{S,O,T,V,M,D}, gamma, y) where {S,O,T,V,M,
                                                           D <: Array{<:Normal} }
  Nt = length(y)
  Ns = length(hmm.B)

  mu = zeros(Ns)
  si = zeros(Ns)

  for j = 1:Ns
    # get mean
    c = 0.0
    for t = 1:Nt
      mu[j] += gamma[t,j] * y[t]
      c += gamma[t,j]
    end
    mu[j] = mu[j]/c

    # get var
    for t = 1:Nt
      si[j] += gamma[t,j] * abs2(y[t] - mu[j])
    end
    si[j] = sqrt(si[j]/c) 
    hmm.B[j] = Normal(mu[j],si[j])
  end

end



"""
  baum_welch!(hmm0, y; kwargs...)

Performs an unsupervised training of `hmm0` using the Baum Welch algorithm.

`y` is a `Nt`-long vector containing the observations.

Currently supports only HMM with observation distributions `Categorical` and `Normal`. 

Keyword arguments:

  * `maxit=50` maximum number of iterations
  * `tol=1e-3` tolerance used in the stopping criterion
  * `normalize=true` apply scaling in the forward-backward algorithm
  * `verbose=true` print iterations

"""
function baum_welch!(hmm::HMM{S,O,T,V,M,D}, y::AbstractArray{O};
                     maxit = 50, 
                     tol = 1e-3, 
                     normalize = true,
                     verbose = true,
                    ) where {S,O,T,V,M,D}

  L = likelihoods(hmm, y, log=false)
  Nt, Ns  = size(L)
  if normalize
    alphas, c = forward( hmm,L)
    betas  = backward(hmm, L, c)
    gammas = alphas .* betas
  else
    alphas = baum_forward( hmm,L)
    betas  = baum_backward(hmm,L)
    gammas = posteriors(alphas, betas)
  end
  nlogL = zeros(maxit)
  epsilon = zeros(Nt-1,Ns,Ns)

  if verbose
    @printf("|-----Baum-Welch----|\n")
    @printf("|  it   |  -log(L)  |\n")
    @printf("|-------------------|\n")
  end

  p, p_prev = -Inf, -Inf
  its = 0
  for k = 1:maxit

    p = 0.0
    if normalize
      for t = 1:Nt
        p += log(c[t])
      end
      p = -p
    else
      for j = 1:Ns
        p += alphas[end,j]
      end
      p = -log(p)
    end
    nlogL[k] = p
    if verbose
      @printf("| %5d | % .2e |\n", k, p)
    end

    # update model parameters
    update_A!(hmm, epsilon, alphas, betas, gammas, L)
    update_a!(hmm, gammas)
    update_B!(hmm, gammas, y)

    likelihoods!(L, hmm, y)

    if normalize
      forward!(alphas, c, hmm, L)
      backward!(betas, hmm, L, c)
      gammas .= alphas .* betas
    else
      baum_forward!( alphas, hmm,L)
      baum_backward!(betas , hmm,L)
      posteriors!(gammas, alphas, betas)
    end

    # stoppin criteria 
    if abs(p_prev - p) < tol 
      break
    else
      its += 1
      p_prev = p
    end

  end
  return nlogL[1:its]

end
