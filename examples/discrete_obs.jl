# # HMM with discrete observations

using Distributions, Random
using HMMBase
using Plots

# https://en.wikipedia.org/wiki/Viterbi_algorithm#Example

a = [0.6, 0.4]
A = [0.7 0.3; 0.4 0.6]
B = [Categorical([0.5, 0.4, 0.1]), Categorical([0.1, 0.3, 0.6])]
hmm = HMM(a, A, B)
#-

z, y = rand(hmm, 250)
z_viterbi = viterbi(hmm, y);
#-

plot(y, linetype=:steppre, label="Observations", size=(600,200))
#-

plot(z, linetype=:steppre, label="True hidden state", size=(600,200))
#-

plot(z_viterbi, linetype=:steppre, label="Viterbi decoded hidden state", size=(600,200))
