# # Numerical Stability

using Distributions
using HMMBase
using PyPlot
using Seaborn

rc("axes", xmargin = 0) # hide
set_style("whitegrid")  # hide

# Let's consider the case of a Normal distribution with null variance,
# such a case can appear during maximum likelihood estimation if only
# one observation is associated to a state:

A = [0.99 0.01; 0.1 0.9]
B = [Normal(0, 1), Normal(10.5, 0)]
hmm = HMM(A, B)
#-

y = rand(hmm, 500)
figure(figsize = (9,2)) # hide
plot(y)
gcf() # hide
#-

# The likelihood of a Normal distribution with null variance goes to infinity for `y = μ`,
# as there is a division by zero in the density function:
println(extrema(likelihoods(hmm, y)))
println(extrema(likelihoods(hmm, y, logl = true)))

# To avoid propagating these non-finite quantities (for example in the forward-backward algorithm),
# you can use the `robust` option:
println(extrema(likelihoods(hmm, y, robust = true)))
println(extrema(likelihoods(hmm, y, logl = true, robust = true)))

# This truncates `+Inf` to the largest Float64, and `-Inf` to the smallest Float64:
prevfloat(Inf), nextfloat(-Inf)

# In the log. case we use `log(prevfloat(Inf))` to avoid overflows when taking
# the exp. of the log-likelihood.
