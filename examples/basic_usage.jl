# # Basic Usage
# [![Binder](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/examples/basic_usage.ipynb)

using Distributions
using HMMBase
using PyPlot
using Seaborn

rc("axes", xmargin = 0) # hide
set_style("whitegrid")  # hide

# ### Model Specification

# B can contains any probability distribution from the `Distributions` package

a = [0.6, 0.4]
A = [0.9 0.1; 0.1 0.9]
B = [MvNormal([0.0, 5.0], ones(2) * 1), MvNormal([0.0, 5.0], ones(2) * 3)]
hmm = HMM(a, A, B)
size(hmm) # (number of states, observations dimension)

# ### Sampling

z, y = rand(hmm, 500, seq = true)

# Let's plot the observations and the hidden state sequence:

_, axes = subplots(nrows = 2, figsize = (9, 3))
axes[1].plot(y)
axes[2].plot(z, linestyle = :steps)
gcf() # hide

# We can also drop the time dimension and plot the data in the plane:

_, axes = subplots(ncols = 2, figsize = (9, 3))
axes[1].scatter(y[:, 1], y[:, 2], s = 3.0)
axes[2].scatter(y[:, 1], y[:, 2], s = 3.0, c = z, cmap = "tab10")
axes[1].set_title("Observations")
axes[2].set_title("Observations and hidden states")
gcf() # hide

# ### Inference

α, logtot = forward(hmm, y)
β, logtot = backward(hmm, y)

γ = posteriors(hmm, y) # or
γ = posteriors(α, β)

size(α), size(β), size(γ)
#-

figure(figsize = (9, 2)) # hide
plot([α[:, 1] β[:, 1] γ[:, 1]])
legend(["Forward", "Backward", "Posteriors"], loc = "upper right")
gcf() # hide
#-

_, axes = subplots(ncols = 3, figsize = (9, 3))
for (ax, probs, title) in zip(axes, [α, β, γ], ["Forward", "Backward", "Posteriors"])
    ax.scatter(y[:, 1], y[:, 2], s = 3.0, c = probs[:, 1], cmap = "Reds")
    ax.set_title(title)
end
gcf() # hide
#-

z_map = [z.I[2] for z in argmax(γ, dims = 2)][:]
z_viterbi = viterbi(hmm, y)

figure(figsize = (9, 2)) # hide
plot([z z_map z_viterbi])
legend(["True sequence", "MAP", "Viterbi"], loc = "upper right")
gcf() # hide
#-

_, axes = subplots(ncols = 2, figsize = (9, 3))
for (ax, seq, title) in zip(axes, [z_map, z_viterbi], ["MAP", "Viterbi"])
    ax.scatter(y[:, 1], y[:, 2], s = 3.0, c = seq, cmap = "Reds_r")
    ax.set_title(title)
end
gcf() # hide

# ### Parameters Estimation

hmm = HMM(randtransmat(2), [MvNormal(rand(2), ones(2)), MvNormal(rand(2), ones(2))])
hmm, hist = fit_mle(hmm, y, display = :iter, init = :kmeans)
hmm
#-

figure(figsize = (4, 3)) # hide
plot(hist.logtots)
gcf() # hide
