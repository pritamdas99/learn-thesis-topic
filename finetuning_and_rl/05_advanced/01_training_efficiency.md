# Efficient training: accumulation, precision, clipping, schedules, and checkpoints

Companion code: [01_training_efficiency.py](01_training_efficiency.py).
Prerequisites: [neural-network training loops](../01_foundations/03_neural_network_training.md)
and [tensors and autograd](../01_foundations/02_tensors_autograd.md).
Read this explanation before running the code. No GPU is required; on a CPU
the script quietly turns the mixed-precision parts off (section 11 explains how).

## 1. What problem are we solving now?

A real fine-tuning run is usually limited by memory and time, not by ideas.
You may want 16 examples per update, but only 4 fit on the device at once.
You may want faster arithmetic, but lower-precision numbers can round small
gradients to zero. You may need to stop a run and resume it tomorrow.

The script packs five standard tools into one tiny regression loop:

| Tool | Question it answers |
| --- | --- |
| Gradient accumulation | How can 4-example forward passes produce a 16-example update? |
| Mixed precision + loss scaling | How can FP16 arithmetic avoid losing small gradients? |
| Gradient clipping | How do we stop one unusually large gradient from causing a huge step? |
| Warmup + cosine schedule | How should the learning rate change across updates? |
| Disk checkpoint | What must be saved so training can be resumed later? |

The toy setup: 36 random inputs with 8 features, 36 random targets, a two-layer
network `Linear(8, 32) -> ReLU -> Linear(32, 1)`, mean-squared-error loss, and
AdamW with base learning rate 0.003. The targets are random noise, so nothing
meaningful is learned. The point is the **mechanics** of each update.

## 2. First identify which memory you need to reduce

Training stores four kinds of numbers:

| Stored item | Grows with | Main technique that reduces it |
| --- | --- | --- |
| Parameters | Model size | Smaller model, quantization ([lesson 04](04_distillation_and_quantization.md)) |
| Gradients | Number of trainable parameters | Freezing, LoRA ([LoRA lesson](../02_finetuning/03_lora_from_scratch.md)) |
| Optimizer state (AdamW keeps two moments per parameter) | Number of trainable parameters | LoRA, sharded optimizers |
| Intermediate activations | Batch size x sequence length x depth | Gradient accumulation, [activation checkpointing](06_activation_checkpointing.md) |

Gradient accumulation reduces the per-forward batch of activations; mixed
precision changes numeric storage and computation; LoRA reduces trainable
parameter state; activation checkpointing recomputes intermediate values.
None is a universal memory fix. In particular, accumulation does **not** shrink
parameters, gradients, or optimizer state at all.

## 3. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $N$ | Examples in one effective (optimizer-step) batch | `samples` (16, 16, then 4) |
| $n_k$ | Examples in microbatch $k$ | `len(inputs)` = 4 |
| $L_k$ | Mean loss over microbatch $k$ | `(model(inputs)-labels).square().mean()` |
| $L$ | Mean loss over the whole effective batch | sum of weighted microbatch losses |
| $K$ | Accumulation count (microbatches per update) | `accumulation = 4` |
| $W$ | Number of data-parallel workers | 1 here ([distributed lesson](07_distributed_training.md)) |
| $S$ | Loss scale for FP16 | `scaler` (65536 by default on CUDA; disabled on CPU) |
| $c$ | Gradient-norm clipping threshold | `1.0` in `clip_grad_norm_` |
| $u$ | Index of optimizer update, starting at 0 | `updates` |
| $m(u)$ | Learning-rate multiplier | `multiplier` |
| $\eta$ | Actual learning rate | `0.003*multiplier` |

## 4. How many updates does the script make?

Count step by step:

1. 36 examples split into microbatches of 4 gives $36/4=9$ microbatches.
2. Groups of $K=4$ microbatches: microbatches 1-4, 5-8, and 9 alone.
3. So `total_updates = math.ceil(9/4) = 3` optimizer updates.

| Update $u$ | Microbatches in group | Examples $N$ | Weight per microbatch $n_k/N$ |
| --- | --- | --- | --- |
| 0 | 4 | 16 | $4/16=0.25$ |
| 1 | 4 | 16 | $4/16=0.25$ |
| 2 | 1 | 4 | $4/4=1$ |

The script docstring states the general rule:

$$\boxed{\text{effective batch}=\text{microbatch}\times K\times W}$$

Here $4\times4\times1=16$ for full groups. With 8 data-parallel workers each
doing the same, one update would see $4\times4\times8=128$ examples. The last
group is short (4 examples), which is why the weighting below uses the
**actual** count rather than the nominal 16.

## 5. Why gradient accumulation can equal a larger batch

Suppose an effective batch has $N$ examples split into microbatches of sizes
$n_k$, with microbatch mean loss $L_k$. Then:

$$
\boxed{L=\sum_k\frac{n_k}{N}L_k,\qquad
\nabla L=\sum_k\frac{n_k}{N}\nabla L_k.}
$$

The second equality follows from linearity of differentiation. Backward calls
**add** gradients into parameter `.grad` buffers, so: weight each microbatch
loss, call backward for each, and step the optimizer only after the group is
complete. Clear gradients before the group, not between its microbatches.

### Worked example: six examples, one weight

Take a one-parameter model $\hat y=wx$ with $w=0.5$ and six examples, split into
a microbatch of 4 and a microbatch of 2:

| $x$ | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| $y$ | 2 | 1 | 4 | 3 | 6 | 9 |
| $wx-y$ | -1.5 | 0 | -2.5 | -1 | -3.5 | -6 |
| $(wx-y)^2$ | 2.25 | 0 | 6.25 | 1 | 12.25 | 36 |

Microbatch losses: $L_1=(2.25+0+6.25+1)/4=2.375$ and $L_2=(12.25+36)/2=24.125$.
Full-batch loss: $57.75/6=9.625$, which equals $\tfrac46(2.375)+\tfrac26(24.125)$.

The gradient of a mean squared error is $\frac{1}{n}\sum 2x(wx-y)$:

- Microbatch 1: $\tfrac{2}{4}[1(-1.5)+2(0)+3(-2.5)+4(-1)]=\tfrac{2}{4}(-13)=-6.5$.
- Microbatch 2: $\tfrac{2}{2}[5(-3.5)+6(-6)]=-53.5$.

| How microbatch gradients are combined | Accumulated gradient |
| --- | --- |
| Weight by size: $\tfrac46(-6.5)+\tfrac26(-53.5)=-4.333-17.833$ | **-22.167** (equals full batch) |
| Naive average: $(-6.5-53.5)/2$ | -30.0 (wrong: over-weights the 2 examples) |

The correct accumulated value matches the gradient computed on all six at once
(both -22.1667 in PyTorch). The script's line `loss = loss*len(inputs)/samples`
is exactly the $n_k/N$ weight.

### The short last group

In the script's third update there is only one microbatch. Its weight is
$4/4=1$. Blindly dividing every loss by $K=4$ would give it weight $0.25$,
shrinking that update's gradient by a factor of 4. (AdamW partly normalizes
gradient scale, but its moment estimates average over earlier steps, so the
shrink is not cleanly cancelled.) For token losses, weight by
supervised-token count rather than example count if that is the intended objective.

### When the equivalence breaks

Exact large-batch equivalence also requires compatible randomness, reductions,
and batch-dependent operations. BatchNorm, for example, sees different statistics
on a microbatch than on a combined batch. Dropout masks differ unless random
streams match. Optimizer steps between microbatches also break the equivalence
because parameters and optimizer state change. Floating-point addition order
differs, so expect agreement up to small rounding differences, not bit-identity.

## 6. Mixed precision: why small gradients disappear

FP16 uses less memory and can run faster on supported hardware, but has limited
range. Its key limits (from `numpy.finfo(numpy.float16)`):

| FP16 property | Value |
| --- | --- |
| Largest finite value | 65504 |
| Smallest normal value | $\approx 6.10\times10^{-5}$ |
| Smallest positive (subnormal) value | $\approx 5.96\times10^{-8}$ |

A gradient smaller than about $3\times10^{-8}$ rounds to zero: it
**underflows**. Values just above that keep very few significant bits.

Fix: scale the loss by $S$ before backward. By linearity,

$$
\boxed{\nabla_\theta(SL)=S\nabla_\theta L.}
$$

Every backward value is $S$ times larger, so small values move into FP16's
representable range. Before the optimizer uses them, the gradients are divided
by $S$ again in FP32.

### Worked example with $S=65536=2^{16}$ (PyTorch's default initial scale)

| True gradient | Stored directly in FP16 | Scaled $\times65536$ in FP16 | Unscaled back (FP32) |
| --- | --- | --- | --- |
| $1\times10^{-8}$ | 0 (lost) | 0.000655 | $\approx1.0\times10^{-8}$ |
| $3\times10^{-8}$ | $6\times10^{-8}$ (100% error) | 0.001966 | $\approx3.0\times10^{-8}$ |
| $1\times10^{-4}$ | 0.0001 | 6.555 | $\approx1.0\times10^{-4}$ |
| $2.0$ | 2.0 | **inf** (131072 > 65504) | step must be skipped |

The last row shows the opposite danger: **overflow**. Dynamic loss scaling
handles it. With PyTorch's `GradScaler` defaults, if any gradient is inf or NaN
the optimizer step is skipped and $S$ is multiplied by 0.5 (65536 -> 32768); after
2000 consecutive successful steps $S$ is multiplied by 2. These are library
defaults (`backoff_factor=0.5`, `growth_factor=2.0`, `growth_interval=2000`),
not values chosen in this script.

In PyTorch autocast, parameters and their `.grad` buffers usually stay FP32;
the FP16 values are activations and intermediate results of eligible operations.
Underflow therefore happens to gradients flowing backward through FP16 operations,
which is exactly what scaling protects.

BF16 has a wider exponent range (about the same as FP32) and often does not need
loss scaling, but supported kernels and hardware still matter. The script uses
FP16 on CUDA and ordinary FP32 on CPU; it does not demonstrate BF16 execution.

## 7. The order of operations matters

For each update the script does:

1. Clear gradients once: `optimizer.zero_grad(set_to_none=True)`.
2. For every microbatch: autocast the forward pass (CUDA only), weight the loss,
   then `scaler.scale(loss).backward()` so gradients accumulate.
3. Unscale gradients **once** after accumulation: `scaler.unscale_(optimizer)`.
4. Clip the actual gradient norm: `clip_grad_norm_(..., 1.0)`.
5. Attempt the optimizer step (`scaler.step`) and update the scale (`scaler.update`).

Why must unscaling come before clipping? Gradient clipping applies

$$
\boxed{g\leftarrow g\cdot\min\!\left(1,\frac{c}{\|g\|}\right)}
$$

with $c=1$ here. Take a true gradient $g=(3,4)$, so $\|g\|=5$:

- **Correct order** (unscale, then clip): $g\cdot\min(1,1/5)=(0.6,0.8)$, norm 1.
- **Wrong order** with $S=1024$: the scaled gradient is $(3072,4096)$, norm 5120.
  Clipping gives $(0.6,0.8)$; unscaling then divides by 1024, leaving
  $(0.000586,0.000781)$, norm $1/1024\approx0.000977$.

Clipping before unscaling enforced the threshold on artificially enlarged
gradients, making the real step about 1000 times too small.

`clip_grad_norm_` uses one **global** norm over all parameters together, not a
separate norm per tensor. Clipping controls gradient magnitude, not the exact
AdamW parameter step magnitude, because AdamW divides by its own running
second-moment estimate.

## 8. Learning-rate schedule: warmup, then cosine decay

Warmup increases the multiplier over initial updates. Afterwards a cosine
multiplier takes the form:

$$
\boxed{m(u)=\tfrac12\left[1+\cos(\pi p(u))\right],\qquad
p(u)=\frac{u-\text{warmup}}{\max(1,\ \text{total}-\text{warmup})}}
$$

where $p$ measures progress after warmup. During warmup the code uses
$m(u)=(u+1)/\text{warmup}$. Schedules should count effective
**optimizer updates**, not microbatches, which is why the script computes the
multiplier once per group.

With `warmup = 1` and `total_updates = 3`:

| $u$ | Branch | Computation | $m(u)$ | Learning rate |
| --- | --- | --- | --- | --- |
| 0 | warmup | $(0+1)/1$ | 1.0 | 0.003 |
| 1 | cosine, $p=0/2=0$ | $\tfrac12(1+\cos0)$ | 1.0 | 0.003 |
| 2 | cosine, $p=1/2$ | $\tfrac12(1+\cos(\pi/2))$ | 0.5 | 0.0015 |

Two honest observations about this tiny setting:

- With `warmup = 1` there is no visible ramp: the single warmup step already
  has multiplier 1. A real run would use many warmup updates (e.g. hundreds),
  starting from a small fraction of the base rate.
- The last multiplier is 0.5 rather than reaching zero. That follows from the
  code's indexing ($p$ never reaches 1 in three updates), not a convergence
  promise. The saved optimizer state indeed records `lr = 0.0015`.

CUDA overflow-skipped steps are not specially accounted for in this teaching
counter (`updates` increases even if `scaler.step` skipped the step); a production
scheduler should define its skipped-step behavior.

## 9. What a resumable checkpoint needs

Model weights define predictions, but future training also depends on optimizer
moments, scaler state, RNG state, scheduler progress, and data order. The file
saves:

| Key in the saved dictionary | Why it is needed |
| --- | --- |
| `'model'` | Network weights |
| `'optimizer'` | AdamW moments, step counts, and current learning rate |
| `'scaler'` | Current loss scale and growth tracker (an empty dict `{}` on CPU, where the scaler is disabled) |
| `'updates'` | Update counter (3 at the end), which drives the schedule |
| `'torch_rng'` | CPU PyTorch random-number-generator state |

It then reloads them with `weights_only=True` as a demonstration.

It does not capture Python/NumPy/CUDA RNG, sampler cursor, all configuration,
or a general scheduler state. Thus it is not an exact full-run-resume system.
Also note the script saves **after** training finishes, so the reload only
proves the round trip works; it does not continue training afterwards.
Activation checkpointing in [lesson 06](06_activation_checkpointing.md) is a
separate memory technique, not this disk-saving operation, despite the shared word.

## 10. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `device = 'cuda' if torch.cuda.is_available() else 'cpu'` | Choose GPU if present; everything else adapts |
| `scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')` | Dynamic loss scaling, only active on CUDA |
| `microbatch, accumulation = 4, 4` | $n_k=4$, $K=4$ |
| `total_updates = math.ceil(len(microbatches)/accumulation)` | $\lceil 9/4\rceil=3$ optimizer updates |
| `samples = sum(len(a) for a, _ in group)` | $N$ for this group: 16, 16, then 4 |
| `optimizer.zero_grad(set_to_none=True)` | Clear gradients once per group, before accumulation |
| `multiplier = (updates+1)/warmup if updates < warmup else 0.5*(1+math.cos(...))` | Warmup then cosine $m(u)$ |
| `param_group['lr'] = 0.003*multiplier` | Set actual learning rate $\eta$ |
| `torch.autocast('cuda', dtype=torch.float16) if device == 'cuda' else nullcontext()` | FP16 forward on CUDA; plain FP32 on CPU |
| `loss = loss*len(inputs)/samples` | Weight $n_k/N$, correct for the short last group |
| `scaler.scale(loss).backward()` | Backward on $S\cdot$ loss; gradients add up |
| `scaler.unscale_(optimizer)` | Divide accumulated gradients by $S$ once |
| `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` | Global-norm clipping with $c=1$ |
| `scaler.step(optimizer)` / `scaler.update()` | Step (skipped on inf/NaN) and adjust $S$ |
| `torch.save({...}, output)` | Write checkpoint to `outputs/training_checkpoint.pt` |
| `torch.load(output, map_location=device, weights_only=True)` | Reload safely (tensors and plain containers only) |

## 11. Run it

Run from the course folder:

```bash
python3 05_advanced/01_training_efficiency.py
```

The only output is one line, for example:

```text
saved and reloaded checkpoint: <course folder>/outputs/training_checkpoint.pt
```

There is no assertion and no printed loss. Success means: three updates ran,
the checkpoint was written to `outputs/`, and all pieces loaded back without error.

On a CPU-only machine (the setting used to verify this lesson) the script still
runs, but `GradScaler(..., enabled=False)` makes `scale` return the loss
unchanged, `unscale_` and `update` do nothing, and `step` simply calls
`optimizer.step()`; autocast is replaced by `nullcontext()`. So on CPU you are
exercising accumulation, clipping, the schedule, and checkpointing in FP32, not
FP16 loss scaling. The CUDA mixed-precision path was not tested here.

## 12. Practice calculations

1. 100 examples, microbatch 4, accumulation 8, one worker. How many optimizer
   updates, and how big is the last group? **25 microbatches; $\lceil25/8\rceil=4$
   updates; the last group has 1 microbatch = 4 examples.**
2. In that last group, what weight should its single microbatch loss get?
   **$4/4=1$, not $1/8$.**
3. Microbatch 8, accumulation 4, 4 GPUs. Effective batch? **$8\times4\times4=128$.**
4. Gradient $(6,8)$ with $c=1$. Clipped gradient? **Norm 10, so $(0.6,0.8)$.**
   With $c=20$? **Unchanged, $(6,8)$, because $\min(1,20/10)=1$.**
5. A true gradient of $5\times10^{-9}$ with $S=65536$: what is the scaled value,
   and does it survive FP16? **About $3.3\times10^{-4}$; yes, it is representable.**
6. Warmup 1, total 5 updates. Multipliers for $u=0,\dots,4$?
   **$p=0,\tfrac14,\tfrac12,\tfrac34$ for $u=1..4$, giving 1, 1, 0.854, 0.5, 0.146.**
7. Exercise from the script: compare one FP32 accumulated gradient with the
   same full-batch gradient before stepping, using a model without dropout or
   BatchNorm. **They should agree up to floating-point summation differences.**

Next: [Soft Actor-Critic: continuous actions and entropy](02_sac_update.md).
