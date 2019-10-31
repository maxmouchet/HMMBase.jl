# # Maximum Likelihood Estimation

using Distributions
using HMMBase
using Plots

y1 = rand(Normal(0,2), 1000)
y2 = rand(Normal(10,1), 500)
y = vcat(y1, y2, y1, y2)

plot(y, linetype=:steppre, size=(600,200))

# For now `HMMBase` does not handle the initialization of the parameters.
# Hence we must instantiate an initial HMM by hand.

hmm = HMM([0.5 0.5; 0.5 0.5], [Normal(-1,2), Normal(15,2)])
hmm = fit_mle(hmm, y, display = :iter)
#-

z_viterbi = viterbi(hmm, y)
plot(z_viterbi, linetype=:steppre, label="Viterbi decoded hidden state", size=(600,200))
