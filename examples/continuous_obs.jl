# # HMM with continuous observations

using Distributions
using HMMBase
using Plots

a = [0.6, 0.4]
A = [0.7 0.3; 0.4 0.6]
B = [MvNormal([0.0,5.0],[1.0,1.0]), MvNormal([5.0,10.0],[1.0,1.0])]
hmm = HMM(a, A, B)
#-

z, y = rand(hmm, 250)
z_viterbi = viterbi(hmm, y);
#-

plot(y, linetype=:steppre, label=["Observations (1)", "Observations (2)"], size=(600,200))
#-

plot(z, linetype=:steppre, label="True hidden state", size=(600,200))
#-

plot(z_viterbi, linetype=:steppre, label="Viterbi decoded hidden state", size=(600,200))