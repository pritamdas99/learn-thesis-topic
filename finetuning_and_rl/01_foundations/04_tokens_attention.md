# Tokens, embeddings, and causal attention

Companion code: [04_tokens_attention.py](04_tokens_attention.py). Read this explanation before running the code.

## Why attention is useful

To predict the next token, a model needs context. In “the cat sat,” a prediction
may depend on several earlier tokens. Attention lets each position choose a
weighted mixture of available contextual representations.

Read [softmax](01_python_and_math.md) first. The script is a mechanics
experiment, not a trained language model.

## Text becomes discrete IDs, then continuous vectors

For `cat sat`, the sorted character vocabulary is `[' ', 'a', 'c', 's', 't']`.
The encoded IDs are `[2,1,4,0,3,1,4]`. IDs are lookup addresses, not numerical
measurements: ID 4 is not “twice as meaningful” as ID 2.

An embedding table $E\in\mathbb R^{V\times d}$ contains one learned vector per
token. Looking up token $i$ returns row $E_i$. Here $V=5$ and $d=8$.
During training, gradients update the rows used by examples; the coordinates
are learned representations, not predefined linguistic labels.

For next-token prediction, inputs are IDs except the last, while labels are
IDs except the first. Position $t$ uses context through $t$ to predict $t+1$.

For example, use the character tokens of `cat sat`, including its space:

```python
ids = [2, 1, 4, 0, 3, 1, 4]  # c a t SPACE s a t
inputs = ids[:-1]             # [2, 1, 4, 0, 3, 1] = "cat sa"
labels = ids[1:]              # [1, 4, 0, 3, 1, 4] = "at sat"
```

Positions start at zero. Each output predicts the token immediately after
its input position, using all allowed context:

| Position $t$ | Input token | Allowed context | Correct next token |
| --- | --- | --- | --- |
| 0 | `c` | `c` | `a` |
| 1 | `a` | `ca` | `t` |
| 2 | `t` | `cat` | space |
| 3 | space | `cat ` | `s` |
| 4 | `s` | `cat s` | `a` |
| 5 | `a` | `cat sa` | `t` |

At position 4, the model predicts `a` from `cat s`. The causal mask prevents
it from reading the `a` at position 5, even though that token is in the input.
The model can compute all six predictions in one forward pass with this mask.

For original IDs $z_0,\ldots,z_{N-1}$, the aligned input and target are
$x_t=z_t$ and $y_t=z_{t+1}$ for $t=0,\ldots,N-2$. The training objective is:

$$
\mathcal L=-\frac{1}{N-1}\sum_{t=0}^{N-2}
\log p_\theta(z_{t+1}\mid z_0,\ldots,z_t).
$$

The last token is excluded from inputs because this sequence provides no
following token as its target. The first token is excluded from labels because
there is no preceding input position to predict it. Seven tokens therefore
provide six input-target pairs. The companion script demonstrates the attention
mechanics; it does not train this predictor.

## Derive scaled dot-product attention

Let $X\in\mathbb R^{T\times d}$ contain one representation per position.
A full attention head learns projections:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V.
$$

A query asks what a position is looking for, a key describes what a position
can match, and a value is the information it contributes. Similarity scores are:

$$
S_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_k}}.
$$

When query and key components are roughly independent, have mean zero,
and have unit variance, a dot product of $d_k$ terms has variance roughly
$d_k$. Dividing by $\sqrt{d_k}$ keeps score scale more stable across dimensions
and helps avoid overly saturated softmax.

To see why, let $q$ and $k$ be one query and one key, each with $d_k$
coordinates. Their dot product is a sum of coordinate products:

$$
D=q\cdot k=\sum_{r=1}^{d_k}q_r k_r.
$$

For the following idealized calculation, assume all coordinates are mutually
independent, with $\mathbb E[q_r]=\mathbb E[k_r]=0$ and
$\operatorname{Var}(q_r)=\operatorname{Var}(k_r)=1$. Unit variance means
expected squared deviation from the mean is 1; it does not mean each value
is 1. Since the means are zero, $\mathbb E[q_r^2]=\mathbb E[k_r^2]=1$.
Each product then has mean zero and variance one:

$$
\mathbb E[q_rk_r]=\mathbb E[q_r]\mathbb E[k_r]=0,
$$

$$
\operatorname{Var}(q_rk_r)
=\mathbb E[q_r^2k_r^2]-(\mathbb E[q_rk_r])^2
=\mathbb E[q_r^2]\mathbb E[k_r^2]=1.
$$

Independent terms have zero covariance, so their variances add:

$$
\operatorname{Var}(D)=\sum_{r=1}^{d_k}\operatorname{Var}(q_rk_r)=d_k.
$$

For a concrete random example, imagine every coordinate is independently
chosen to be $-1$ or $+1$, each with probability $1/2$. Each coordinate and
each product has mean zero and variance one. With $d_k=4$, one draw could be:

$$
q=[1,-1,1,1],\qquad k=[1,1,-1,1],
$$

$$
D=1-1-1+1=0.
$$

Other draws give dot products of $-4,-2,0,2,$ or $4$, with probabilities
$1/16,4/16,6/16,4/16,1/16$, respectively. Across these draws the mean is zero
and the variance is:

$$
\mathbb E[D^2]=\frac{16+4(4)+6(0)+4(4)+16}{16}=4=d_k.
$$

Variance describes the spread across possible draws, not the value of one
particular dot product. Its square root, the standard deviation, is
$\sqrt{d_k}$. Scaling a random quantity by $c$ scales its variance by $c^2$:

$$
\operatorname{Var}\left(\frac{D}{\sqrt{d_k}}\right)
=\frac{1}{d_k}\operatorname{Var}(D)=1.
$$

| Key dimension $d_k$ | Raw dot-product standard deviation | Divisor | Scaled standard deviation |
| --- | --- | --- | --- |
| 4 | 2 | 2 | 1 |
| 64 | 8 | 8 | 1 |
| 256 | 16 | 16 | 1 |

Larger score differences make softmax more extreme. For example, scores
`[0, 8]` give probabilities approximately `[0.0003, 0.9997]`. Dividing these
scores by 8 gives `[0, 1]`, whose probabilities are approximately
`[0.269, 0.731]`. Scaling helps prevent dimension alone from driving softmax
toward near-zero and near-one probabilities, where its gradients can be small.

This is a motivation, not a guarantee about learned queries and keys. Their
coordinates may be correlated or have different means and variances. In
particular, the companion script uses $Q=K=X$, so a self-dot-product does not
satisfy the independent-query-and-key assumption above.

Normalize each row over source positions, then form a weighted sum:

$$
P_{ij}=\frac{e^{S_{ij}}}{\sum_l e^{S_{il}}},\qquad
O_i=\sum_jP_{ij}V_j.
$$

## A two-position worked example

For easy arithmetic, take one-dimensional queries, keys, and values
$Q=K=V=[1,2]^T$. Then $S=[[1,2],[2,4]]$.
At position 0, causal attention allows only position 0, so weights are `[1,0]`
and output is 1. At position 1, softmax of `[2,4]` is approximately
`[0.1192,0.8808]`, giving output $0.1192(1)+0.8808(2)=1.8808$.

Each row's weights sum to one. Different query positions can produce different
mixtures of the same value vectors.

## Why the future must be masked

For causal language modeling, position $i$ may see only $j\le i$:

$$
S_{ij}\leftarrow-\infty\quad\text{when }j>i.
$$

Because $e^{-\infty}=0$, future tokens receive zero probability. Without this
mask, the model could read the token it is supposed to predict during training.
It could appear successful while failing at generation, where that token does
not yet exist.

| Code | Formula/meaning |
| --- | --- |
| `x @ x.T / sqrt(dimension)` | Scores with the simplifying choice $Q=K=V=X$ |
| `triu(diagonal=1)` | True above the main diagonal: future positions |
| `masked_fill(..., -inf)` | Eliminate forbidden positions before normalization |
| `.softmax(-1)` | Normalize over keys for each query row |
| `probabilities @ x` | Weighted value mixtures |

The script's embeddings are randomly initialized and never trained. It also
omits learned Q/K/V projections, multiple heads, and position embeddings.
Therefore its printed weights demonstrate valid shapes and masking, not
meaningful learned language relationships.

## Run and check

Run `python 01_foundations/04_tokens_attention.py`. Expect a `[6,8]` output
and a lower-triangular `[6,6]` attention matrix with row sums one. Change only
the final input vector: earlier outputs must not change. Remove the mask and
repeat: earlier outputs can now depend on the changed future token.

Next: [a complete tiny transformer](05_tiny_transformer.md).
Reference: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
