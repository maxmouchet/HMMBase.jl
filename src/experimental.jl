# Currently (as of Julia 1.4) it is not possible to directly
# dezerialize JSON to a Julia structure.
# E.g. `Normal(JSON.parse(JSON.json(Normal())))` is not possible.
# We provide the following methods for convenience.

function from_dict(T::Type{<:Distribution}, d::AbstractDict)
    args = ntuple(i -> d[string(fieldname(T, i))], fieldcount(T))
    T(args...)
end

function from_dict(T::Type{<:MixtureModel{VF,VS,C,CT}}, d::AbstractDict) where {VF,VS,C,CT}
    prior = Vector{CT}(d["prior"]["p"])
    components = map(x -> from_dict(C, x), d["components"])
    MixtureModel(components, prior)
end

function from_dict(::Type{HMM{F,T}}, D::AbstractVector{<:Type}, d::AbstractDict) where {F,T}
    a = Vector{T}(d["a"])
    A = Matrix{T}(hcat(d["A"]...))
    B = [from_dict(Dx, x) for (Dx, x) in zip(D, d["B"])]
    HMM{F,T}(a, A, B)
end

function from_dict(::Type{HMM{F,T}}, D::Type, d::AbstractDict) where {F,T}
    from_dict(HMM{F,T}, fill(D, length(d["B"])), d)
end

# function parsefile(::Type{HMM{F,T}}, D, filename) where {F,T}
# from_dict(HMM{F,T}, D, parsefile)
# end

# HMM{F,T}(D, d::AbstractDict) where {F,T} = from_dict(HMM{F,T}, D, d)

function MixtureModel(hmm::AbstractHMM)
    sdists = statdists(hmm)
    @check length(sdists) == 1
    MixtureModel([hmm.B...], sdists[1])
end

function HMM(m::MixtureModel)
    K = length(m.prior.p)
    a = m.prior.p
    A = repeat(permutedims(m.prior.p), K, 1)
    B = m.components
    HMM(a, A, B)
end
