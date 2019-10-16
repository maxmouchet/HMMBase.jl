# Convenience functions
# Auto-magically generated for forward/backward

# Generate the following methods from the in-place versions
# forward(a, A, L) -> \alpha, logtot
# forward(hmm, observation) -> \alpha, logtot
# backward(...)

for f in (:forward, :backward)
    f!  = Symbol("$(f)!")   # forward!
    fl  = Symbol("$(f)log") # forwardlog
    fl! = Symbol("$(fl)!")  # forwardlog!
    @eval begin
        """
            $($f)(a, A, L)

        Compute $($f) probabilities using samples likelihoods.
        See [Forward-backward algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm).
        """
        function $(f)(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix)
            m = Matrix{Float64}(undef, size(L))
            c = Vector{Float64}(undef, size(L)[1])
            $(f!)(m, c, a, A, L)
            m, sum(log.(c))
        end

        """
            $($fl)(a, A, LL)

        Compute $($f) probabilities using samples log-likelihoods.
        See [`$($f)`](@ref).
        """
        function $(fl)(a::AbstractVector, A::AbstractMatrix, L::AbstractMatrix)
            m = Matrix{Float64}(undef, size(L))
            c = Vector{Float64}(undef, size(L)[1])
            $(fl!)(m, c, a, A, L)
            m, sum(log.(c))
        end
        
        """
            $($f)(hmm, observations)

        # Example
        ```julia
        hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
        z, y = rand(hmm, 1000)
        probs, tot = $($f)(hmm, y)
        ```
        """
        function $(f)(hmm::AbstractHMM, observations)
            $(f)(hmm.π0, hmm.π, likelihoods(hmm, observations))
        end

        """
            $($fl)(hmm, observations)

        # Example
        ```julia
        hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
        z, y = rand(hmm, 1000)
        probs, tot = $($fl)(hmm, y)
        ```
        """
        function $(fl)(hmm::AbstractHMM, observations)
            $(fl)(hmm.π0, hmm.π, loglikelihoods(hmm, observations))
        end
    end
end