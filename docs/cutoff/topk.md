# Top-K

## Optimal Cutoff Timestep (OCT)

$$
g(\boldsymbol{X}) = \arg\min_{\hat{t}}\{  \forall \hat{t}_1 > \hat{t}: \mathbf{1}(f(\boldsymbol{X}[\hat{t}_1])= \boldsymbol{y})\}
$$

## Top-K Gap for Cutoff Approximation

The defination of $Top_k(\boldsymbol{Y}(t))$ as the top-$k$ output occurring in one neuron of the output layer,

$$
Y_{gap}= Top_1(\boldsymbol{Y}(t)) - Top_2(\boldsymbol{Y}(t)),
$$

which denotes the gap of top-1 and top-2 values of output $\boldsymbol{Y}(t)$. Then, we let $ D\{\cdot\}$ denote the inputs in subset of $D$ that satisfy a certain condition. Now, we can define the confidence rate as follows:

$$
\textit{Confidence rate: } C(\hat{t}, D\{Y_{gap}>\beta\}) = \frac{1}{|D\{Y_{gap}>\beta\}|}\sum_{\boldsymbol{X}\in D\{Y_{gap}>\beta\}} (g(\boldsymbol{X}) \leq \hat{t}),
$$

The algorithm searches for a minimum $\beta \in \mathbb{R^+}$ at a specific $\hat t$, as expressed in the following optimization objective:

$$
\arg\min_{\beta} C(\hat t, D\{Y_{gap} > \beta\}) \geq 1-\epsilon,
$$

where $\epsilon$ is a pre-specified constant such that $1-\epsilon$ represents an acceptable level of confidence for activating cutoff, and a set of $\beta$ is extracted under different $\hat t$ using training samples.
