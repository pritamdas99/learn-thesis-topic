# Transfer learning: reuse features, then adapt them

Companion code: [01_transfer_learning.py](01_transfer_learning.py). Read this explanation before running the code.

## What changes when we fine-tune?

Training from scratch initializes parameters without task-specific knowledge.
Transfer learning begins with parameters learned on a source task, then adapts
them to a target task. The hope is that useful source features reduce target
data needs. Transfer can also hurt if the source representation is unsuitable.

The code provides a small, fully local experiment. A backbone maps two input
features to 16 hidden features; a head maps those features to two logits.

$$
h=f_\phi(x),\qquad z=g_\psi(h),\qquad
L(\phi,\psi)=\frac1N\sum_i-\log p_{\phi,\psi}(y_i\mid x_i).
$$

$\phi$ names backbone parameters, and $\psi$ names head parameters.
Read [neural-network training](../01_foundations/03_neural_network_training.md)
for cross-entropy and backpropagation.

## Source and target tasks in the script

Source labels are $y_s=\mathbf1[x_1>0]$. Target labels are
$y_t=\mathbf1[x_1+0.5x_2>0]$. Both depend on similar inputs, but the target
boundary is tilted. For input $[-0.2,1]$, the source label is 0 while the
target label is 1 because $-0.2+0.5(1)=0.3>0$.

The script uses the first 300 inputs for training in both phases and the final
100 only for final target evaluation. Source and target training tasks share
inputs intentionally; they do not share the held-out evaluation inputs.

## The three training phases

1. **Source training:** optimize both $\phi$ and $\psi_s$ for 100 steps.
2. **Head adaptation:** replace the head with randomly initialized $\psi_t$,
   freeze $\phi$, and train only $\psi_t$ for 60 steps.
3. **Full adaptation:** unfreeze $\phi$ and train both for 60 steps at a smaller
   learning rate.

While frozen, the backbone is still evaluated in the forward pass. “Frozen”
means no learned parameter update, not that its output is constant for all inputs.
The head learns a new decision rule over the existing representation:

$$
\psi_t\leftarrow\psi_t-\eta\nabla_{\psi_t}L,
\qquad \phi\text{ unchanged}.
$$

After unfreezing, both derivatives matter. A lower learning rate reduces the
size of initial changes to reused features. It is a heuristic to preserve useful
structure, not a guarantee against forgetting.

## How the code implements the idea

| Code | Purpose |
| --- | --- |
| `model[1] = nn.Linear(16, 2)` | Discard the source decision head and create a target head |
| `p.requires_grad_(False)` | Exclude backbone parameters from gradient computation |
| Filter `model.parameters()` by `requires_grad` | Give the optimizer only the intended parameters |
| Call `fit` again after unfreezing | Create a new optimizer that includes the newly trainable backbone |
| `with torch.no_grad()` on final inputs | Evaluate without modifying parameters |

Each `fit` call creates a new AdamW optimizer, so optimizer moments restart at
each stage. That is a teaching choice. Some practical workflows preserve or
carefully rebuild optimizer state when changing trainable parameter groups.

### Model weights and optimizer memory are different

The model stores learned weights and biases. AdamW separately stores a step
counter and two moving averages for each parameter it updates. For a scalar
parameter with gradient $g_t$, these averages are:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

The first moment $m_t$ tracks the recent gradient direction; the second moment
$v_t$ tracks squared gradient magnitude (it is not the centered variance).
For tensors, these calculations happen element by element. Both start at zero.
AdamW uses bias-corrected versions,
$\hat m_t=m_t/(1-\beta_1^t)$ and $\hat v_t=v_t/(1-\beta_2^t)$, in its update:

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Here $\eta$ is the learning rate, $\lambda$ is weight decay, and $\epsilon$
is a small positive number for numerical stability. The optimizer therefore
uses recent gradient history as well as the current gradient.

For example, take $\beta_1=0.9$ and $\beta_2=0.999$. If the first gradient
is $g_1=2$, then $m_1=0.2$ and $v_1=0.004$. If the next gradient is $-1$,
continuing with that same optimizer gives:

$$
m_2=0.9(0.2)+0.1(-1)=0.08,\qquad
v_2=0.999(0.004)+0.001(1)=0.004996.
$$

Recreating the optimizer before that second gradient instead gives fresh
first-step values $m_1=-0.1$ and $v_1=0.001$. Even with the same model weight
and current gradient, the gradient-based update can differ because the history
and step counter differ. These numbers are the raw moments, before bias correction.

In this script, the state transitions are:

| Stage | Model parameters | AdamW state |
| --- | --- | --- |
| Source training | Train backbone and source head | Start fresh |
| Head adaptation | Keep learned backbone; replace source head with a random target head | Start fresh for target head only |
| Full adaptation | Keep the backbone and the now-trained target head; unfreeze backbone | Start fresh for both |

Creating a new optimizer does **not** reinitialize the model. In particular,
the third `fit` call keeps the target head's learned weights but discards its
60 steps of optimizer history. Only the explicit `nn.Linear(16, 2)` replacement
creates a new head with new initial weights.

An alternative at unfreezing is to keep the head optimizer and add the newly
trainable backbone with `optimizer.add_param_group(...)`. That retains the
head's history while giving the newly added parameters fresh state; learning
rates must also be set as intended for each group. Unfreezing alone does not
add parameters to an optimizer that excluded them. Replacing a head also
requires updating optimizer parameter references: an old optimizer still
refers to the old head. Preserving state is a deliberate choice, since source
gradient history may not suit a changed target objective.

### Freezing parameters is different from freezing BatchNorm statistics

For a vision backbone containing BatchNorm, freezing weights alone does not
freeze running statistics; train/eval mode needs separate attention. This small
backbone has only a linear layer and ReLU, so that issue is absent here.

BatchNorm normally has two kinds of stored state:

| State | How it changes |
| --- | --- |
| Learned scale $\gamma$ and shift $\beta$ | Gradient-based optimizer updates |
| Running mean and running variance | Updates during forward passes in training mode |

Setting `requires_grad_(False)` stops new gradients for the learned parameters.
It does not change the layer's training mode, so BatchNorm can still update
its running statistics when the frozen backbone processes a batch. Even
`torch.no_grad()` does not stop those training-mode updates.

For example, suppose a BatchNorm feature has running mean 10. A target batch
has mean 20, and BatchNorm momentum is 0.1. A training-mode forward pass updates:

$$
\text{running mean}_{\text{new}}=0.9(10)+0.1(20)=11.
$$

That change happens even if every backbone parameter is frozen and no
optimizer step runs. The running variance updates separately as well. During
training, BatchNorm normalizes using the current batch statistics; during
evaluation, standard BatchNorm uses the stored running statistics. Thus, the
representation at evaluation can change even though the weights did not.

If the intention is to keep both the frozen backbone's parameters and its
BatchNorm statistics fixed while training the head, a typical setup is:

```python
for p in backbone.parameters():
    p.requires_grad_(False)

model.train()      # Put the overall model in training mode first.
backbone.eval()    # Then keep the backbone in evaluation mode.
# Train using an optimizer containing only the head parameters.
```

Calling `model.train()` again also puts its backbone back into training mode,
so reapply `backbone.eval()` afterward when needed. Conversely, `eval()` by
itself does not disable gradients or freeze parameters. These are separate
controls. The example assumes BatchNorm tracks running statistics, its default
behavior; evaluation mode also disables dropout in a backbone containing it.

Sometimes adapting BatchNorm statistics to target data is intentional. The
choice is whether to retain source statistics or update them using target
batches. None of this changes the supplied script: its `Linear` and `ReLU`
backbone has no running statistics or dropout.

## What the result means

Run `python 02_finetuning/01_transfer_learning.py`. Expect high target accuracy.
The file illustrates the mechanics but does not prove transfer helped: there is
no matched target-only baseline in its printed result. To test that claim,
train a fresh model on the same target data and compare multiple seeds under a
fair budget. Also evaluate the source task after adaptation to measure forgetting.

Exercise: train with only 20 target examples. Compare head-only, full adaptation,
and scratch training. Predict which has lower variance, then check rather than
assuming pretrained features always win.
