# # Maximum a Posteriori

using Distributions
using HMMBase
using PyPlot
using Seaborn

rc("axes", xmargin = 0) # hide
set_style("whitegrid")  # hide

# Let's consider a simple time series with one outlier:
y = rand(1000)
y[100] = 10000
figure(figsize = (9,2)) # hide
plot(y)
gcf() # hide
#-

# An MLE approach to observations distributions parameters
# may fail with a singularity (variance = 0) if the outlier
# is the single observation associated to some state:
hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(5,1)])
try
    fit_mle(hmm, y, display=:iter);
catch e
    println(e)
end

# We can avoid this by putting a prior on the variance:
import ConjugatePriors: InverseGamma, NormalKnownMu, posterior_canon
import StatsBase: Weights

function fit_map(::Type{<:Normal}, observations, responsibilities)
    μ = mean(observations, Weights(responsibilities))

    ss = suffstats(NormalKnownMu(μ), observations, responsibilities)
    prior = InverseGamma(2,1)
    posterior = posterior_canon(prior, ss)
    σ2 = mode(posterior)

    Normal(μ, sqrt(σ2))
end

hmm, _ = fit_mle(hmm, y, estimator=fit_map, display=:iter);
hmm.B