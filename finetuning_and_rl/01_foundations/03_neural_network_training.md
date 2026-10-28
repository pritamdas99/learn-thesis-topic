# Neural networks and the complete training loop

Companion code: [03_neural_network_training.py](03_neural_network_training.py). Read this explanation before running the code.

## Why use a neural network?

A linear classifier makes a straight decision boundary. The script labels an
example positive when $x_1x_2>0$: both coordinates have the same sign. Positive
examples occupy two opposite quadrants. One straight line cannot separate
these from the other two quadrants, so we need a nonlinear model.

Prerequisite: [tensors and gradients](02_tensors_autograd.md). Our network is:

$$
h=\operatorname{ReLU}(W_1x+b_1),\qquad z=W_2h+b_2.
$$

$h$ has 32 hidden features; $z$ has two class scores. ReLU is
$\max(0,u)$ elementwise. It creates different linear behavior in different
regions. Without ReLU, composing the layers gives
$z=W_2W_1x+W_2b_1+b_2$, still just one affine map.

## From scores to a trainable objective

For class $k$, softmax gives $p_k=e^{z_k}/\sum_j e^{z_j}$. If the correct class
is $y$, the per-example loss is:

$$
\ell=-\log p_y=-z_y+\log\sum_j e^{z_j}.
$$

Differentiating gives a useful identity:

$$
\frac{\partial\ell}{\partial z_k}=p_k-\mathbf1[k=y].
$$

For probabilities `[0.7,0.3]` and correct class 1, gradients on the logits are
`[0.7,-0.7]`. Gradient descent pushes the wrong score down and correct score up.
The chain rule then propagates through the output layer, ReLU, and first layer.
This is why decreasing cross-entropy encourages correct classification.

`cross_entropy(model(inputs), labels)` accepts raw logits and integer class
indices. It combines log-softmax and negative log-likelihood internally.
Passing softmax probabilities instead applies the wrong transformation.

## Minibatches, epochs, and AdamW

The empirical objective averages losses across training examples:

$$
L(\theta)=\frac1N\sum_i\ell_i(\theta).
$$

A shuffled minibatch estimates its gradient using a smaller average. An epoch
visits all training examples once. In this file, 400 examples with batch size 32
produce 13 batches, the last one containing 16 examples.

Adam keeps moving averages of gradients and their squares:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

Bias-corrected versions divide by $1-\beta_1^t$ and $1-\beta_2^t$.
The adaptive update scales each coordinate roughly by
$\hat m_t/(\sqrt{\hat v_t}+\varepsilon)$. AdamW additionally shrinks parameters
by a factor approximately $1-\eta\lambda$ through decoupled weight decay.
This discourages unnecessarily large weights; it does not guarantee generalization.

Global gradient clipping replaces $g$ by
$g\min(1,c/\|g\|_2)$, with threshold $c=1$ here. It limits an unusually large
gradient before the optimizer uses it.

## Map the loop to its purpose

| Code | Meaning |
| --- | --- |
| `optimizer.zero_grad(...)` | Remove gradients left by the previous minibatch |
| `loss.backward()` | Compute this minibatch's parameter derivatives |
| `clip_grad_norm_` | Limit the global gradient norm |
| `optimizer.step()` | Apply AdamW's update |
| `model.train()` | Enable training behavior in layers such as dropout |
| `model.eval()` and `no_grad()` | Select evaluation behavior and disable graph recording |
| `copy.deepcopy(model.state_dict())` | Preserve the actual best weights, not references to changing tensors |

## Why separate training, validation, and test data?

The first 400 examples update weights. The next 100 select the stopping point.
The final 100 are evaluated once after restoring the best validation checkpoint.
This separation reduces optimistic estimates caused by choosing settings on
the same examples used to report final quality.

If validation loss fails to improve for eight epochs, training stops. This is
patience-based early stopping. Because validation influenced model selection,
its score is not an untouched final test result.

Accuracy is the fraction whose largest logit selects the correct class. A
model can improve confidence and lower cross-entropy without changing any
argmax predictions, so accuracy and loss measure different things.

## Run and investigate

Run `python 01_foundations/03_neural_network_training.py`. Expect falling
validation loss and high test accuracy on this easy synthetic task; exact
numbers are not universal guarantees. Remove ReLU and explain the remaining
classification errors. Reduce training examples and increase width: compare
training and validation behavior. Do not repeatedly use the final test split
to decide which modification to keep.

Next: [tokens and attention](04_tokens_attention.md).
