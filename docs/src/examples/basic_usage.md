```@meta
EditURL = "https://github.com/maxmouchet/HMMBase.jl/blob/master/examples/basic_usage.jl"
```

# Basic Usage
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxmouchet/HMMBase.jl/master?filepath=/examples/basic_usage.inpynb)

```@example basic_usage
using Distributions
using HMMBase
using PyPlot
using Seaborn

rc("axes", xmargin = 0) # hide
set_style("whitegrid")  # hide
```

### Model Specification

B can contains any probability distribution from the `Distributions` package

```@example basic_usage
a = [0.6, 0.4]
A = [0.9 0.1; 0.1 0.9]
B = [MvNormal([0.0, 5.0], ones(2) * 1), MvNormal([0.0, 5.0], ones(2) * 3)]
hmm = HMM(a, A, B)
size(hmm) # (number of states, observations dimension)
```

### Sampling

```@example basic_usage
z, y = rand(hmm, 500, seq = true)
```

Let's plot the observations and the hidden state sequence:

```@example basic_usage
_, axes = subplots(nrows = 2, figsize = (9, 3))
axes[1].plot(y)
axes[2].plot(z, linestyle = :steps)
gcf() # hide
```

We can also drop the time dimension and plot the data in the plane:

```@example basic_usage
_, axes = subplots(ncols = 2, figsize = (9, 3))
axes[1].scatter(y[:, 1], y[:, 2], s = 3.0)
axes[2].scatter(y[:, 1], y[:, 2], s = 3.0, c = z, cmap = "tab10")
axes[1].set_title("Observations")
axes[2].set_title("Observations and hidden states")
gcf() # hide
```

### Inference

```@example basic_usage
α, logtot = forward(hmm, y)
β, logtot = backward(hmm, y)

γ = posteriors(hmm, y) # or
γ = posteriors(α, β)

size(α), size(β), size(γ)
```

```@example basic_usage
figure(figsize = (9, 2)) # hide
plot([α[:, 1] β[:, 1] γ[:, 1]])
legend(["Forward", "Backward", "Posteriors"], loc = "upper right")
gcf() # hide
```

```@example basic_usage
_, axes = subplots(ncols = 3, figsize = (9, 3))
for (ax, probs, title) in zip(axes, [α, β, γ], ["Forward", "Backward", "Posteriors"])
    ax.scatter(y[:, 1], y[:, 2], s = 3.0, c = probs[:, 1], cmap = "Reds")
    ax.set_title(title)
end
gcf() # hide
```

```@example basic_usage
z_map = [z.I[2] for z in argmax(γ, dims = 2)][:]
z_viterbi = viterbi(hmm, y)

figure(figsize = (9, 2)) # hide
plot([z z_map z_viterbi])
legend(["True sequence", "MAP", "Viterbi"], loc = "upper right")
gcf() # hide
```

```@example basic_usage
_, axes = subplots(ncols = 2, figsize = (9, 3))
for (ax, seq, title) in zip(axes, [z_map, z_viterbi], ["MAP", "Viterbi"])
    ax.scatter(y[:, 1], y[:, 2], s = 3.0, c = seq, cmap = "Reds_r")
    ax.set_title(title)
end
gcf() # hide
```

### Parameters Estimation

```@example basic_usage
hmm = HMM(randtransmat(2), [MvNormal(rand(2), ones(2)), MvNormal(rand(2), ones(2))])
hmm, hist = fit_mle(hmm, y, display = :iter, init = :kmeans)
hmm
```

```@example basic_usage
figure(figsize = (4, 3)) # hide
plot(hist.logtots)
gcf() # hide
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

