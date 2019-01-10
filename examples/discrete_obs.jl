# # HMM with discrete observations

using Distributions
using HMMBase
using Plots

π = [0.9 0.1; 0.2 0.8]
D = [Categorical([0.9, 0.1]), Categorical([0.2, 0.8])]
hmm = HMM(π, D)

z, y = sample_hmm(hmm, 250)
plot(y)