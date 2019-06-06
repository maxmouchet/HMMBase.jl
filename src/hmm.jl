"""
    AbstractHMM

An HMM type must at-least implement the following interface:
```julia
struct CustomHMM{S,O} <: AbstractHMM{S,O}
    a              # Initial state distribution
    A              # Transition matrix
    D              # Observations distributions
    # Custom fields ....
end
```
where `S` is the state type and `O` is the observation type.
"""
abstract type AbstractHMM{S,O} end

"""
HMM([a::AbstractVector{T}], A::AbstractMatrix{T}, B::AbstractVector{<:Distribution}) where {T}

Build an HMM with transition matrix `A` and observations distributions `B`.  
If the initial state distribution `a` is not specified, a uniform distribution is assumed. 

Observations distributions can be of different types (for example `Normal` and `Exponential`).  
However they must be of the same dimension (all scalars or all multivariates).

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
```
"""
struct HMM{S, O,
           T <: AbstractFloat, 
           V <: AbstractVector{T},
           M <: AbstractMatrix{T},
           D <: AbstractVector{<:Distribution}
          } <: AbstractHMM{S,O}
    a::V
    A::M
    B::D
    function HMM(a::V, A::M, B::D) where {T,V<:AbstractArray{T},M<:AbstractMatrix{T},D}  
      O = typeof(rand(B[1]))
      assert_hmm(a, A, B) && new{Int,O,T,V,M,D}(a, A, B) 
    end
end

HMM(A::AbstractMatrix{T}, B) where {T} = HMM(ones(T,size(A)[1])./size(A,1), A, B)

"""
    assert_hmm(π0::AbstractVector{Float64}, π::AbstractMatrix{Float64}, D::AbstractVector{<:Distribution})

Throw an `AssertionError` if the initial state distribution and the transition matrix rows does not sum to 1,
and if the observations distributions does not have the same dimensions.
"""
function assert_hmm(a::AbstractVector{T}, 
                    A::AbstractMatrix{T}, 
                    B::AbstractVector{<:Distribution}) where {T}
    
    if !isprobvec(a)
      error("Initial state distribution must sum to 1")
    end
    if !(isTMatrix(A))
      error("Trasition matrix rows must sum to 1")
    end
    if any(length.(B) .!= length(B[1]))
      error("All distributions must have the same dimensions")
    end
    if size(A,1) != size(A,2)
      error("Transition matrix must be squared")
    end
    if !(length(a) == size(A,1) == length(B))
      error("length(a), length(B), size(A,1) = $(length(a)),$(length(B)),$(size(A,1)) should be matching")
    end
    return true

end


isTMatrix(A::AbstractMatrix) = all([isprobvec(A[i,:]) for i in 1:size(A,1)])

"""
    rand(hmm::AbstractHMM, Nt::Int; [s0])

Generate a random trajectory of `hmm` for `Nt` timesteps. 
Returns two `Nt`-long arrays:  
* `y` observation sequence.
* `s` state sequence.

If the initial state `s0` is not specified it will be randomly drawn from the initial state distribution `hmm.a`.   

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
z, y = rand(hmm, 1000)
```
"""
function rand(hmm::AbstractHMM{S,O}, Nt::Int; s0::S=rand(Categorical(hmm.a))) where {S,O}

  # init
  z = Array{S,1}(undef,Nt)
  y = Array{O,1}(undef,Nt)

  z[1] = s0 
  y[1] = rand(hmm.B[z[1]])

  for t = 2:Nt
    z[t] = rand(Categorical(hmm.A[z[t-1],:])) # update state
    y[t] = rand(hmm.B[z[t]])               # get observation
  end

  return y, z
end

"""
    rand(hmm::AbstractHMM, s::AbstractVector)

Generate observations `y` from a `hmm` according to trajectory `s`.

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
y = rand(hmm, [1, 1, 2, 2, 1])
```
"""
rand(hmm::AbstractHMM{S,O}, s::AbstractVector{S}) where {S,O} =
[ rand(hmm.B[si]) for si in s ]



"""
    size(hmm::AbstractHMM)

Returns a tuple containing the number of states `S` and the dimension of the observations `N`.

# Example
```julia
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
size(hmm) # (2,1)
```
"""
size(hmm::AbstractHMM) = (length(hmm.B), length(hmm.B[1]))
size(hmm::AbstractHMM, dims::Int) = size(hmm)[dims]

function likelihood(hmm::AbstractHMM{S,O}, y::AbstractArray{O}; 
                    log::Bool = false) where {S,O}
  Nt = length(y)     # time steps
  N = size(hmm,1)   # number HMM's states
  L = zeros(Nt,N)
  log ? loglikelihood!(L, hmm, y) : likelihood!(L, hmm, y)
  return L
end

for f in [:loglikelihood! => :logpdf, :likelihood! => :pdf]

  @eval begin

    function $(f[1])(L::AbstractMatrix, 
                     hmm::AbstractHMM{S,O}, y::AbstractArray{O} ) where {S,O}
      for t = 1:size(L,1), j = 1:size(L,2) 
        L[t,j] = $(f[2])( hmm.B[j], y[t] )
      end
      return L
    end

  end

end
#
#"""
#    n_parameters(hmm::AbstractHMM)
#
#Returns the number of parameters in `hmm`.  
#
## Example
#```julia
#hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
#n_parameters(hmm) # 6
#```
#"""
#function n_parameters(hmm::AbstractHMM)
#    length(hmm.π) - size(hmm.π)[1] + sum(d -> length(params(d)), hmm.D)
#end
