# Activation checkpointing: recompute values to save training memory

Companion code: [06_activation_checkpointing.py](06_activation_checkpointing.py).
Prerequisites: [tensors and autograd](../01_foundations/02_tensors_autograd.md),
[neural network training](../01_foundations/03_neural_network_training.md), and
[training efficiency](01_training_efficiency.md).
Read this explanation before running the code.

## 1. What problem are we solving now?

Training memory has several parts:

| Memory item | Depends on batch size? | Changed by activation checkpointing? |
| --- | --- | --- |
| Weights | No | No |
| Gradients | No | No |
| Optimizer state (e.g. Adam moments) | No | No |
| **Saved activations** for backward | Yes, and on sequence length | **Yes** |

A forward pass computes intermediate tensors called **activations**. Backward
needs many of them to apply the chain rule, so autograd keeps them alive until
backward reaches the layer that needs them. For deep networks with long
sequences and large batches, these saved activations can use more memory than
the weights.

**Activation checkpointing** (also called gradient checkpointing) keeps only
selected tensors, usually the inputs at block boundaries. During backward it
re-runs the forward pass of each block to rebuild the missing intermediates.
It trades **extra computation** for **less saved activation memory**.

This is different from saving a model checkpoint to disk. Nothing here writes
files; the word "checkpoint" means "a saved point in the computation".

## 2. Why does backward need forward values?

Start with the smallest possible chain. Let

$$u=wx,\qquad y=u^2.$$

The chain rule gives

$$
\frac{\partial y}{\partial w}=\frac{\partial y}{\partial u}\,
\frac{\partial u}{\partial w}=2u\,x.
$$

The derivative contains $u$, which was produced during the forward pass. There
are two ways to have $u$ available when backward arrives:

| Strategy | What is kept after forward | What backward does |
| --- | --- | --- |
| Ordinary | $x$ and $u$ | Uses the stored $u$ |
| Checkpointed | Only $x$ (and $w$, which is a parameter anyway) | Recomputes $u=wx$, then uses it |

With $w=3$ and $x=2$:

1. Forward: $u=3\cdot2=6$, $y=6^2=36$.
2. Ordinary backward: read stored $u=6$; derivative $=2(6)(2)=24$.
3. Checkpointed backward: recompute $u=3\cdot2=6$; derivative $=2(6)(2)=24$.

Same answer, one extra multiplication, one fewer stored value. A large block is
the same idea at scale: its internal tensors can be reconstructed from its input
and parameters. Checkpointing does not stop those tensors from existing
**temporarily** during computation; it stops them from being **retained** from
the forward pass until backward reaches the block.

## 3. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $B$ | Batch size (examples) | 8 in the script: `torch.randn(8, 16)` |
| $S$ | Sequence length (tokens per example) | 1024 in the worked example |
| $H$ | Hidden size (features per token) | 16 input, 64 hidden in the script |
| $L$ | Number of layers / blocks | 32 in the worked example |
| $b$ | Bytes per number | 2 for fp16/bf16, 4 for fp32 |
| $c$ | Retained internal tensors per layer, in units of one $B\times S\times H$ tensor | Assumed 16 in the worked example |
| $k$ | Layers per checkpointed segment | 8 in the $\sqrt{L}$ example |
| $M_{act}$ | Activation memory retained for backward | — |

One "hidden-sized" activation tensor costs

$$\boxed{\text{bytes}=B\times S\times H\times b}.$$

## 4. The script's tiny model: what is actually saved?

The script builds a linear–GELU–linear network and an identical deep copy:

```python
ordinary = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 16))
recomputed = copy.deepcopy(ordinary)
x = torch.randn(8, 16)
```

Name the intermediate tensors:

```text
x [8,16] --Linear--> h [8,64] --GELU--> a [8,64] --Linear--> out [8,16] --square,mean--> loss
```

- The second `Linear` needs `a` to compute its weight gradient.
- `GELU` needs `h` to compute its local derivative.
- The first `Linear` needs `x` for its weight gradient.

`h` and `a` each have $8\times64=512$ numbers. In the script's default fp32
($b=4$), each costs $512\times4=2048$ bytes, so the two internal activations
together cost **4096 bytes**.

As an extra check (not part of the script), wrapping each forward in
`torch.autograd.graph.saved_tensors_hooks` and recording what autograd saves
gave, for the same seed and shapes:

| Run | Tensors saved for backward (shapes) | Hidden activations `h`, `a` retained? |
| --- | --- | --- |
| Ordinary | `[8,16]`, `[8,64]`, `[8,64]`, `[64,16]`, `[8,16]` | Yes, 1024 numbers |
| Checkpointed | `[8,16]`, `[8,16]` | No |

The `[64,16]` entry is a view of a weight that exists anyway, `x` exists anyway,
and the final `[8,16]` is the block output saved by `.square()`. The two
`[8,64]` tensors are the activations that checkpointing avoided keeping.
A forward hook on the checkpointed copy also showed its first `Linear` running
**once during forward and once more during backward**: that second call is the
recomputation.

These bytes are trivial; the script is about **correctness**, not savings.

## 5. Why should the gradients agree?

If recomputation uses the same function, parameters, inputs, and relevant random
state, the rebuilt activations equal the originals, so the chain rule produces
the same gradients up to floating-point effects. Checkpointing changes the
**execution schedule**, not the mathematical objective.

The script runs one model normally and the copy through:

```python
checkpoint(recomputed, x, use_reentrant=False)
```

Each computes the mean square of its output and backpropagates. The file then
compares every pair of corresponding parameter gradients with
`torch.allclose(..., atol=1e-6)`. In the run used to write this lesson, both
losses were `0.04553...` and the largest absolute gradient difference was 0.

`use_reentrant=False` selects the non-reentrant implementation explicitly. It
also handles this case, where the input `x` does not itself require gradients
but the module parameters do.

## 6. Worked example: how much memory could a real model save?

The script does not measure this. The following is a **back-of-envelope model**
with an assumed constant, intended only to show the arithmetic.

Take a transformer-like stack with $B=8$, $S=1024$, $H=4096$, $L=32$, in fp16
($b=2$). One hidden-sized tensor:

$$8\times1024\times4096\times2=67{,}108{,}864\text{ bytes}=64\text{ MiB}.$$

**Assume** each layer retains internal tensors worth $c=16$ such tensors
(attention and MLP intermediates; the true value depends on the architecture,
the attention kernel, and what each operator saves).

**Ordinary training** keeps everything for all layers:

$$M_{act}=L\times c\times64\text{ MiB}=32\times16\times64\text{ MiB}=32{,}768\text{ MiB}=32\text{ GiB}.$$

**Checkpoint every layer** keeps only each layer's input, then rebuilds one
layer's internals at a time during backward:

| Part | Arithmetic | Memory |
| --- | --- | --- |
| Stored boundaries | $32\times64$ MiB | 2 GiB |
| One layer rebuilt during backward | $16\times64$ MiB | 1 GiB |
| Peak activation total | $2+1$ | **about 3 GiB** |

Under these assumptions activation memory drops by about $32/3\approx10.7\times$.
Weights, gradients and optimizer state are unchanged, so the **total** memory
reduction is smaller than that ratio.

## 7. What does it cost in compute?

A common rule of thumb: backward costs about twice a forward pass. Then one
training step costs about $1+2=3$ forward-equivalents. Checkpointing every block
re-runs each block's forward once more:

$$\frac{1+1+2}{1+2}=\frac43\quad\Rightarrow\quad\text{about }33\%\text{ more compute per step}.$$

This is an estimate, not a measurement. Real overhead varies with which layers
are checkpointed, kernel efficiency, and whether the run was limited by compute
or memory. Sometimes the freed memory allows a larger batch, which can recover
some throughput.

## 8. The $\sqrt{L}$ idea: checkpoint segments, not every layer

Use a simpler unit model: a chain of $L$ layers where each layer's output is one
unit of memory. Ordinary backprop stores all $L$ units.

Split the chain into segments of $k$ layers. Keep only the $L/k$ segment
boundaries. During backward, recompute one segment at a time, which temporarily
needs its $k$ layer outputs:

$$\boxed{M(k)\approx\frac{L}{k}+k}.$$

This is smallest near $k=\sqrt L$, giving $M\approx2\sqrt L$. For $L=64$:

| Segment size $k$ | Boundaries $L/k$ | Rebuilt $k$ | Total units |
| --- | --- | --- | --- |
| 1 | 64 | 1 | 65 |
| 2 | 32 | 2 | 34 |
| 4 | 16 | 4 | 20 |
| **8** | **8** | **8** | **16** |
| 16 | 4 | 16 | 20 |
| 64 | 1 | 64 | 65 |

So 16 units instead of 64, while still recomputing each layer's forward about
once (roughly the same one-extra-forward cost as in Section 7). In practice,
libraries often checkpoint each transformer block because a block contains many
internal tensors relative to its single input, as in Section 6.

## 9. When can recomputation surprise you?

- **Randomness.** Dropout consumes random numbers; the recomputed mask must match
  the original. PyTorch's checkpoint utilities handle supported RNG
  preservation, but arbitrary state, external randomness, or device movement
  inside a block can require care.
- **Side effects.** Incrementing counters, logging, or updating running
  statistics (such as batch-norm buffers) may execute again during
  recomputation. Do not put side effects or state-changing operations inside
  recomputed blocks.
- **Library switches.** Many transformer libraries expose
  `gradient_checkpointing_enable()`. Training usually disables generation KV
  caching (`use_cache`); caching and activation recomputation serve different
  purposes. Inspect the model's implementation rather than assuming one switch
  covers every block.

## 10. What the script shows and what it does not

| The script shows | The script does not show |
| --- | --- |
| Checkpointed and ordinary gradients match for one small network | Any peak-memory reduction |
| How to call `checkpoint(..., use_reentrant=False)` on a module | Runtime overhead |
| Gradient equality as a test of correct recomputation | Behavior with dropout, batch norm, or GPUs |

Do not infer a particular memory-reduction ratio from this script. To measure
savings, use a larger workload, reset peak-memory counters, synchronize GPU
operations when timing, and compare identical model/data configurations.

## 11. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `torch.manual_seed(7)` | Fix initial weights and the input for reproducibility |
| `nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 16))` | The block whose activations are saved or recomputed |
| `recomputed = copy.deepcopy(ordinary)` | Identical weights, separate gradient storage |
| `x = torch.randn(8, 16)` | Batch of 8 inputs with 16 features; no `requires_grad` |
| `ordinary(x).square().mean()` | Ordinary forward: autograd retains `h` and `a` |
| `checkpoint(recomputed, x, use_reentrant=False)` | Retain only the input; rerun the block in backward |
| `.square().mean()` | Same mean-square loss on both outputs |
| `regular_loss.backward()` / `checkpoint_loss.backward()` | Compute gradients; the second triggers recomputation |
| `torch.allclose(p.grad, q.grad, atol=1e-6)` | Check each parameter's gradients agree |

## 12. Run and check

Run from the course folder:

```bash
python3 05_advanced/06_activation_checkpointing.py
```

Expected output:

```text
recomputed and ordinary gradients match
```

The assertion checks that all four gradient tensors (two weights, two biases)
agree within `atol=1e-6`. It does not check memory or speed.

Experiment: change only the copied model's weights, for example add 0.01 to the
first layer's weight inside `torch.no_grad()`. The assertion should then fail,
because the two compared functions differ. Restore equality afterward. Then
profile a larger model on a GPU, as the script's docstring suggests, to study the
actual memory-versus-runtime tradeoff.

## 13. Practice calculations

1. An fp16 activation has $B=4$, $S=2048$, $H=4096$. How many MiB is one
   hidden-sized tensor? **$4\times2048\times4096\times2=67{,}108{,}864$ bytes $=64$ MiB.**
2. In the script, how many bytes do `h` and `a` take together in fp32?
   **$2\times8\times64\times4=4096$ bytes.**
3. Using the Section 6 assumptions but $L=48$, what is ordinary activation
   memory, and the per-layer-checkpoint peak?
   **$48\times16\times64$ MiB $=48$ GiB; $48\times64$ MiB $+16\times64$ MiB $=3+1=4$ GiB.**
4. For a unit chain of $L=100$, what segment size minimizes $L/k+k$, and what
   is the total? **$k=10$, total $10+10=20$ units instead of 100.**
5. If backward costs twice a forward, what is the extra compute of recomputing
   every forward once? **$4/3-1\approx33\%$.**
6. Does checkpointing reduce optimizer-state memory? **No. It only changes how
   many activations are retained for backward.**

Next: [distributed data parallel training](07_distributed_training.md).
