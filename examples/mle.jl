# # MLE Estimator

using Distributions
using HMMBase
using Plots

y1 = rand(Normal(0,2), 1000)
y2 = rand(Normal(10,1), 500)
y = vcat(y1, y2, y1, y2)

plot(y, linetype=:steppre, size=(600,200))

# For now `HMMBase` does not handle the initialization of the parameters.
# Hence we must instantiate an initial HMM by hand.

hmm = HMM([0.5 0.5; 0.5 0.5], [Normal(0,1), Normal(10,1)])
hmm, log_likelihood = fit_mle!(hmm, y, verbose=true)
#-

z_viterbi = viterbi(hmm, y)
plot(z_viterbi, linetype=:steppre, label="Viterbi decoded hidden state", size=(600,200))

# We can also perform individual EM steps.

hmm, log_likelihood = mle_step(hmm, y)