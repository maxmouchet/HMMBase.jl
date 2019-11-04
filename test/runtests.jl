# Low-level, isolated, functions tests.
include("unit.jl")

# High-level API tests.
# Ensure that everything works well together, for different kinds of HMMs.
include("integration.jl")
