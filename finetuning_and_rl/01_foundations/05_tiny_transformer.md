# A tiny transformer language model from start to sampling

Companion code: [05_tiny_transformer.py](05_tiny_transformer.py). Read this explanation before running the code.

## The learning objective

This file trains a small character language model from random initialization.
It learns one repeated sentence, so it demonstrates pretraining mechanics and
memorization, not broad language understanding. Read [attention](04_tokens_attention.md)
and [training loops](03_neural_network_training.md) first.

A causal model factorizes a sequence probability:

$$
p_\theta(x_1,\ldots,x_T)=\prod_{t=1}^Tp_\theta(x_t\mid x_{<t}).
$$

Taking negative logs turns products into sums. The training objective is the
average next-token negative log-probability. We can predict every next token
in a window in parallel because the causal mask prevents future leakage.

## What each part of the architecture contributes

Input token $x_t$ becomes $h_t^{(0)}=E[x_t]+P[t]$. $E$ stores token embeddings;
$P$ stores learned position embeddings. Adding positions lets the model
represent order: the same token at two locations need not have the same input.

The block uses four attention heads. With width 32, each head uses width 8.
Independent projected queries/keys/values allow different heads to aggregate
different relationships. Head outputs are concatenated and projected back to
width 32.

The default encoder layer here uses residual additions and post-layer
normalization, with dropout set to zero:

$$
u=\operatorname{LayerNorm}(h+\operatorname{MHA}(h)),\qquad
h'=\operatorname{LayerNorm}(u+\operatorname{FFN}(u)).
$$

The feed-forward network is $W_2\operatorname{ReLU}(W_1u+b_1)+b_2$, with
intermediate width 64. Residual paths make it easier to propagate information
and gradients without forcing every block to relearn an identity mapping.
Layer normalization standardizes features within a token representation and
then applies learned scale and offset; it does not normalize a whole batch.

Although the PyTorch class is named `TransformerEncoder`, supplying a causal
mask makes this use causal self-attention. There is no cross-attention or
separate encoder-decoder architecture in this example.

## Shapes and loss

| Stage | Shape |
| --- | --- |
| Input IDs | `[B,T]` |
| Embeddings and block output | `[B,T,32]` |
| Vocabulary logits | `[B,T,V]` |
| Flattened logits for cross-entropy | `[B*T,V]` |
| Flattened next-token labels | `[B*T]` |

A sampled batch contains 25 tokens per example. The first 24 are inputs; the
last 24, offset by one, are targets. The output projection produces a score
for every vocabulary token at every input position.

$$
L=-\frac1{BT}\sum_{b,t}\log p_\theta(x_{b,t+1}\mid x_{b,\le t}).
$$

Each of 120 updates uses 16 randomly selected windows. This code has no held-out
split, so reported loss measures fitting the repeated training sentence only.
If evaluating a larger corpus, split source documents before making overlapping
windows; otherwise train and validation can contain nearly identical contexts.

## Why generation is sequential

Training knows all target tokens, but generation must create them. Start with
`a`, obtain the last position's logits, sample one token, append it, and repeat.
Temperature $\tau>0$ changes the distribution:

$$
p_i(\tau)=\frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}.
$$

For two logits differing by 2, their probability odds are $e^{2/\tau}$.
At $\tau=1$ the odds are about 7.39; at $\tau=0.5$ they are about 54.6.
Lower temperature sharpens preferences, but cannot make an inaccurate model
correct. `multinomial` samples rather than always taking the largest score.

Only the last 24 tokens are passed back to the model. Their learned positions
restart at zero for each window; this is a simple sliding-context demonstration,
not a KV-cached long-context implementation. `eval()` and `no_grad()` prepare
inference behavior and prevent unnecessary backward graphs.

## Run and experiment

Run `python 01_foundations/05_tiny_transformer.py`. Loss should fall and sampled
text should resemble the repeated sentence. Try a new starting character and
a different temperature. Add a held-out corpus and compare loss: memorizing
one sentence does not establish generalization. Check causality by changing a
future input ID and confirming earlier logits stay unchanged in evaluation mode.

Reference: [original transformer architecture](https://arxiv.org/abs/1706.03762).
Next: [transfer learning](../02_finetuning/01_transfer_learning.md).
