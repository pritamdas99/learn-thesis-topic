# Distributed data parallel training: split data and average gradients

Companion code: [07_distributed_training.py](07_distributed_training.py).
Prerequisites: [training loops](../01_foundations/03_neural_network_training.md),
[gradient accumulation](01_training_efficiency.md), and
[activation checkpointing](06_activation_checkpointing.md).
Read this explanation before running the code.

## 1. What problem are we solving now?

One process can only work through so many examples per second. **Data parallel
training** runs several workers at once:

1. Every worker holds a **full copy** (replica) of the model.
2. Each worker receives a **different slice** of the data.
3. Each worker computes gradients on its own slice.
4. Workers **average** their gradients, so every replica sees the same gradient.
5. Every replica applies the same optimizer step and stays identical.

This increases how much data is processed per distributed step. It does **not**
divide one model's weight memory among workers: each one still stores all
weights, gradients and optimizer state.

The example uses two CPU processes with the Gloo communication backend. No GPU
is required.

## 2. The toy task in the script

The synthetic target is

$$y=x_1+x_2+x_3+x_4.$$

The script draws 128 random inputs with 4 features each and trains
`nn.Linear(4, 1)`. A linear model can represent this target exactly with weight
vector `[1,1,1,1]` and bias 0, so we know what a correct run should learn.

| Setting | Value in the script |
| --- | --- |
| Dataset size $N$ | 128 |
| Workers $P$ (`--nproc_per_node`) | 2 |
| Local batch size $B_{local}$ | 16 |
| Optimizer | SGD, learning rate 0.1 |
| Epochs | 10 |
| Loss | Mean squared error on each local batch |

## 3. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $P$ | World size: number of worker processes | 2; `--nproc_per_node=2` |
| $p$ or rank | This worker's index, $0,\dots,P-1$ | `dist.get_rank()` |
| $B_{local}$ | Examples per worker per step | `batch_size=16` |
| $K$ | Gradient-accumulation steps | 1 (none in the script) |
| $B_{eff}$ | Examples contributing to one optimizer update | 32 |
| $L_p$ | Mean loss on rank $p$'s local batch | `loss` on each rank |
| $g_p$ | Gradient of $L_p$ | `param.grad` before syncing |
| $\bar g$ | Averaged gradient after all-reduce | `param.grad` after `backward()` |
| $\eta$ | Learning rate | `lr=0.1` |

**All-reduce** is a collective operation: every rank contributes a tensor and
every rank receives the combined result. DDP all-reduces gradients (sum, then
divide by $P$) during `loss.backward()`.

## 4. Why is averaging gradients mathematically sensible?

Suppose $P$ workers each receive $n$ examples, with local mean loss $L_p$. The
mean loss over all $Pn$ examples is

$$
L=\frac1P\sum_{p=1}^P L_p,
\qquad
\boxed{\nabla L=\frac1P\sum_{p=1}^P\nabla L_p=\bar g}.
$$

So the averaged gradient equals the gradient of one big batch containing every
worker's examples, **when local batch sizes are equal**.

### Worked 2-rank example

Use a one-weight model $\hat y=wx$, squared error, $w=0$, $\eta=0.1$. For mean
squared error, $\partial L/\partial w=\frac2n\sum_i x_i(wx_i-y_i)$.

| Rank | Local data $(x,y)$ | Per-example $x(wx-y)$ at $w=0$ | Local gradient $g_p$ |
| --- | --- | --- | --- |
| 0 | $(1,1),(2,2)$ | $-1,\ -4$ | $\frac22(-1-4)=-5$ |
| 1 | $(1,3),(1,3)$ | $-3,\ -3$ | $\frac22(-3-3)=-6$ |

Step by step:

1. All-reduce sum: $-5+(-6)=-11$.
2. Divide by $P=2$: $\bar g=-5.5$ on **both** ranks.
3. Check against one big batch of all four examples:
   $\frac24(-1-4-3-3)=\frac24(-11)=-5.5$. Same.
4. SGD on each rank: $w\leftarrow0-0.1(-5.5)=0.55$. Both replicas now hold
   $w=0.55$.

If both replicas start from the same weights and optimizer state, identical
steps keep them synchronized. (DDP also broadcasts rank 0's parameters to other
ranks when the wrapper is constructed; in this script both ranks additionally
use the same seed.) Without synchronization they would drift into separate
models fitting separate data streams.

The original two-number example works the same way: local gradients 2 and 6
average to 4 on each worker.

### Caveat: unequal batch sizes

Equal weighting of ranks equals equal weighting of examples only when local
batch sizes match. If rank 0 has 8 examples with mean gradient 2 and rank 1 has
24 examples with mean gradient 6:

| Method | Arithmetic | Result |
| --- | --- | --- |
| Default DDP average | $\frac12(2)+\frac12(6)$ | 4 |
| True mean over 32 examples | $\frac{8}{32}(2)+\frac{24}{32}(6)=0.5+4.5$ | 5 |

The correct weights are $1/4$ and $3/4$, not $1/2$ and $1/2$. Unequal token
counts create the analogous issue for masked language-model objectives, where
each rank's "mean over tokens" covers a different number of tokens.

## 5. How is the dataset split across ranks?

Every rank in the script builds the **same** 128-example dataset (same seed).
`DistributedSampler` then decides which indices each rank reads. Roughly:
shuffle all indices with a seed that depends on the epoch, pad if needed, and
give rank $p$ every $P$-th index starting at $p$.

### Script case: 128 examples, 2 ranks

- Per rank: $128/2=64$ examples.
- Batches per rank per epoch: $64/16=4$.
- Synchronized optimizer updates: 4 per epoch, $4\times10=40$ in total.
- The two ranks' index sets are disjoint and together cover all 128 indices
  (checked by building both samplers in one process; 0 overlap, 128 union).

For the shuffle used in epoch 0 (seed 0 is the sampler default), the first
eight indices were rank 0: `[44, 121, 31, 71, 105, 15, 48, 30]` and rank 1:
`[94, 13, 116, 84, 14, 22, 11, 77]`. After `sampler.set_epoch(1)` they change,
e.g. rank 0 starts `[37, 20, 103, 33, ...]`. Without `set_epoch`, every epoch
would reuse the same order.

### Small uneven case: 10 examples, 3 ranks, no shuffle

$10$ is not divisible by $3$. Verified with `DistributedSampler(...,
num_replicas=3, rank=r, shuffle=False)`:

| Setting | Samples per rank | Rank 0 | Rank 1 | Rank 2 |
| --- | --- | --- | --- | --- |
| `drop_last=False` (default) | $\lceil10/3\rceil=4$ | `[0,3,6,9]` | `[1,4,7,0]` | `[2,5,8,1]` |
| `drop_last=True` | $\lfloor10/3\rfloor=3$ | `[0,3,6]` | `[1,4,7]` | `[2,5,8]` |

With padding, indices 0 and 1 are seen twice (12 slots for 10 examples). With
`drop_last=True`, example 9 is skipped. Account for duplicates when computing
evaluation metrics across ranks.

## 6. Effective batch size and learning rate

The number of examples behind one optimizer update is

$$
\boxed{B_{eff}=B_{local}\times P\times K},
$$

assuming equal-sized batches and correctly weighted accumulation. In the script:
$B_{eff}=16\times2\times1=32$.

| Run | $B_{local}$ | $P$ | $K$ | $B_{eff}$ | Updates per epoch ($128/B_{eff}$) |
| --- | --- | --- | --- | --- | --- |
| One worker | 16 | 1 | 1 | 16 | 8 |
| Script | 16 | 2 | 1 | 32 | 4 |
| One worker, matched | 32 | 1 | 1 | 32 | 4 |
| One worker, accumulation | 16 | 1 | 2 | 32 | 4 |

Two consequences:

- Increasing worker count with fixed local batch **changes the global batch** and
  **reduces updates per epoch**. Ten epochs on one worker with batch 16 is 80
  updates; ten epochs in the script is 40. Matched epoch count does not imply
  matched optimizer-update count.
- A larger batch gives a less noisy gradient but fewer steps. A common heuristic,
  the "linear scaling rule", multiplies the learning rate by the batch-size
  factor (here $0.1\times2=0.2$), often with warmup. This is a heuristic, not a
  guarantee: it can fail at large batch sizes and interacts with the optimizer.
  The script keeps `lr=0.1` and does not test any scaling rule.

As an extra check (not part of the script), a single-process replay that used
the same per-rank index order and averaged the two ranks' batch losses for 40
updates produced the same weights as the two-process run, to all printed digits.
That is the equivalence in Section 4 holding in practice.

## 7. Read the process lifecycle

| Code | Role |
| --- | --- |
| `torchrun` | Launch workers and set rank/world-size environment variables |
| `'RANK' not in os.environ` check | Refuse to run as a plain `python` script |
| `init_process_group('gloo')` | Establish the distributed communication group |
| `DistributedSampler` | Partition data consistently across ranks |
| `DistributedDataParallel(model)` | Synchronize replicated model training |
| `loss.backward()` | Trigger gradient synchronization through DDP hooks |
| Rank-zero condition | Print once instead of duplicating output from all workers |
| `destroy_process_group()` in `finally` | Clean up communication resources |

Every participating worker must execute compatible collective operations. A
worker taking a different branch and skipping backward can leave the others
waiting forever. This equal-sized teaching dataset avoids uneven-input
complications. Save checkpoints only on rank zero, and aggregate evaluation
metrics across ranks rather than reporting one rank's numbers.

## 8. DDP versus sharding

DDP normally keeps full weights, gradients, and optimizer state on each worker.
FSDP and ZeRO variants **shard** some of these states to reduce per-worker
memory. Those methods add gathering/scattering and have different
memory/communication tradeoffs. They are not implemented in this file.

For multiple NVIDIA GPUs, a typical adaptation uses one process per GPU, selects
the local device, moves inputs and model there, and uses NCCL. Do not assume
this CPU script can use GPUs merely by changing the backend string.

## 9. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `if 'RANK' not in os.environ:` | Detect a missing `torchrun` launch and exit with instructions |
| `dist.init_process_group('gloo')` | Join the CPU communication group |
| `torch.manual_seed(7)` | Same dataset and initial weights on every rank |
| `x = torch.randn(128, 4)` | 128 examples, 4 features |
| `TensorDataset(x, x.sum(-1, keepdim=True))` | Target $y=x_1+x_2+x_3+x_4$ |
| `DistributedSampler(dataset, shuffle=True)` | Give each rank its own 64 indices |
| `DataLoader(dataset, batch_size=16, sampler=sampler)` | Local batch $B_{local}=16$ |
| `DistributedDataParallel(nn.Linear(4, 1))` | Wrap the replica so gradients are averaged |
| `torch.optim.SGD(model.parameters(), lr=0.1)` | Identical optimizer on every rank |
| `sampler.set_epoch(epoch)` | Different deterministic shuffling each epoch |
| `(model(inputs)-targets).square().mean()` | Local mean squared error $L_p$ |
| `loss.backward()` | Compute $g_p$ and all-reduce to $\bar g$ |
| `optimizer.step()` | Same update on every rank |
| `if dist.get_rank() == 0:` | Print from one rank only |
| `model.module.weight` | The wrapped `nn.Linear` inside DDP |
| `dist.destroy_process_group()` | Clean up in `finally` |

## 10. Run and investigate

Run from the course directory:

```bash
torchrun --standalone --nproc_per_node=2 05_advanced/07_distributed_training.py
```

`--standalone` runs a single-node rendezvous on this machine;
`--nproc_per_node=2` starts two worker processes (ranks 0 and 1). Local socket
access is needed for the rendezvous even though both workers are on the same
machine. Running the file with plain `python` exits with a message telling you
to use `torchrun`.

`torchrun` may first print a warning that it set `OMP_NUM_THREADS=1`. Rank zero
then prints the learned weight matrix as a nested list, e.g.

```text
DDP learned weights: [[0.99894118309021, 0.9991607069969177, 1.0007399320602417, 0.9972922205924988]]
```

These are approximately `[0.999, 0.999, 1.001, 0.997]`, close to the target of
all ones, matching the record in `VERIFICATION.md`. The bias is not printed. The
script has no assertion; you judge success by these numbers. Exact digits can
vary with PyTorch version and platform.

Investigations:

1. Print sample indices per rank (the script's exercise) and verify that the two
   ranks' sets are disjoint and cover all 128 examples.
2. Compare one and two workers at matched **global** batch size and update count.
   Explain why matched epoch count alone may not imply matched update count.
3. Try `--nproc_per_node=4` with the same local batch size and predict the new
   $B_{eff}$ and updates per epoch before running.

## 11. Practice calculations

1. Two ranks have local gradients $-2$ and $8$. What gradient does each rank use
   after DDP? **$(-2+8)/2=3$.**
2. With $\eta=0.1$ and starting weight $1.0$, what is the new weight on each rank?
   **$1.0-0.1(3)=0.7$ on both.**
3. Four ranks, local batch 16, accumulation 2. What is $B_{eff}$?
   **$16\times4\times2=128$.**
4. With 4 ranks and local batch 16 on the 128-example dataset, how many examples
   per rank and updates per epoch? **$128/4=32$ examples; $32/16=2$ updates.**
5. Ranks hold 10 and 30 examples with mean gradients 1 and 5. What does default
   DDP average give, and what is the true per-example mean?
   **DDP: 3. True: $\frac{10}{40}(1)+\frac{30}{40}(5)=0.25+3.75=4$.**
6. 7 examples on 2 ranks with `drop_last=False` and no shuffle: what does each
   rank get? **$\lceil7/2\rceil=4$ each: rank 0 `[0,2,4,6]`, rank 1 `[1,3,5,0]`
   (index 0 is repeated as padding).**
7. Does DDP let a model too large for one worker's memory fit across two
   workers? **No. Each rank holds a full replica; that needs sharding (FSDP/ZeRO)
   or model parallelism.**

Next: [projects](../06_projects/PROJECTS.md).
