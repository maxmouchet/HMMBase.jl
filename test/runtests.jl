using Test
using Distributions
using HMMBase
using Random
using JSON
using LinearAlgebra

using HMMBase: from_dict, issquare

Random.seed!(2019)
@testset "All" begin
    # Low-level, isolated, functions tests.
    include("unit.jl")

    # High-level API tests.
    # Ensure that everything works well together, for different kinds of HMMs.
    include("integration.jl")

    # Test that deprecated methods are still working.
    include("deprecated.jl")
end
