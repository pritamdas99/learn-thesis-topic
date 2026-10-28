# LoRA: adapting a model through low-rank weight updates

Companion code: [03_lora_from_scratch.py](03_lora_from_scratch.py). Read this explanation before running the code.

## Why train fewer parameters?

A pretrained model may contain millions or billions of weights. Full tuning
stores gradients and optimizer states for all of them. LoRA freezes the base
weights and trains a smaller structured update. The assumption is that a useful
task adaptation may be expressible in a lower-dimensional family than an
arbitrary full weight change. This need not hold for every task.

Prerequisites: [matrix multiplication](../01_foundations/02_tensors_autograd.md)
and [transfer learning](01_transfer_learning.md).

## Factor the update, not the original weight

For a linear layer, PyTorch stores $W\in\mathbb R^{d_{out}\times d_{in}}$ and
computes $XW^T+b$. LoRA defines:

$$
W_{eff}=W_0+sBA,\quad s=\alpha/r,
$$

where $A\in\mathbb R^{r\times d_{in}}$ and
$B\in\mathbb R^{d_{out}\times r}$ are trainable. Since
$\operatorname{rank}(BA)\le r$, the added update is low-rank. The original
$W_0$ itself can remain full-rank.

For row-oriented batches:

$$
XW_{eff}^T=XW_0^T+sXA^TB^T.
$$

This explains the exact transpose order in `x @ self.a.T @ self.b.T`.
For example, input `[128,8]` multiplied by $A^T[8,2]$ becomes `[128,2]`, then
by $B^T[2,4]$ becomes `[128,4]`, matching the base output.

## Parameter and rank arithmetic

A full update requires $d_{out}d_{in}$ parameters. LoRA uses
$r(d_{in}+d_{out})$. In this lesson, full weights contain $4\times8=32$
entries and LoRA uses $2(8+4)=24$. The toy saving is modest. A 4096-by-4096
layer at rank 8 would use 65,536 adapter parameters instead of 16,777,216 full
weight parameters, excluding any biases and other trainable modules.

Low-rank tuning reduces trainable state. It does not automatically eliminate
base weights, forward computation, or activation memory.

## Why initialize one factor to zero?

The script initializes $A$ randomly and $B=0$. Then $BA=0$, so attaching the
adapter initially preserves the base model's predictions.

Let $G=\partial L/\partial W_{eff}$. Matrix derivatives give:

$$
\frac{\partial L}{\partial B}=sGA^T,\qquad
\frac{\partial L}{\partial A}=sB^TG.
$$

Initially, $B$ can receive a nonzero gradient because $A$ is nonzero. $A$'s
first gradient is zero because $B=0$; after $B$ changes, $A$ can learn too.
If both factors started at zero, both gradients would be zero and this adapter
would be stuck. This is a concrete reason for the asymmetric initialization.

## What the experiment actually learns

The code constructs targets using a frozen base weight plus a synthetic rank-2
update. Thus a rank-2 adapter has enough representational capacity to reproduce
the target. It fits predictions using mean squared error on 128 random inputs.
This is an intentionally favorable reconstruction problem, not proof that rank
2 is enough for language-model adaptation.

| Code | Meaning |
| --- | --- |
| `self.base.requires_grad_(False)` | Freeze the original layer |
| `self.scale = alpha / rank` | Scale the learned update |
| Optimizer filters `requires_grad` | Optimize only adapter factors |
| `merged_weight()` | Construct $W_0+sBA$ |
| Frozen-weight assertion | Check the original model was preserved during adaptation |
| Merged-output assertion | Check the factorized and merged computations agree |

Merging preserves the output of this simple adapter in exact arithmetic. The
base bias must still be included if present. Real adapters may have dropout,
quantized bases, or other variants that require additional handling.

## Run and reason

Run `python 02_finetuning/03_lora_from_scratch.py`. Expect 24 trainable
parameters and very small final reconstruction error. Try rank 1: it cannot
represent a general rank-2 target update exactly. Increase rank and compare
error, parameter count, and convergence rather than assuming bigger always wins.

An adapter checkpoint needs its compatible original base weights to work.
Saving $A$ and $B$ alone does not save a standalone pretrained model.
Reference: [LoRA paper](https://arxiv.org/abs/2106.09685).
