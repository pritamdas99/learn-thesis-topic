# Distillation and quantization: two different ways to make models smaller

Companion code: [04_distillation_and_quantization.py](04_distillation_and_quantization.py).
Prerequisites: [softmax](../01_foundations/01_python_and_math.md),
[cross-entropy training](../01_foundations/03_neural_network_training.md),
[KL divergence in RLHF](../04_alignment/02_rlhf_kl_penalty.md), and the QLoRA
section of [Hugging Face SFT](../02_finetuning/04_huggingface_sft.md).
Read this explanation before running the code.

## 1. What problem are we solving now?

A large trained model can be too slow, too memory-hungry, or too expensive to
serve. There are two very different ways to shrink it:

| Idea | What changes | Requires training? |
| --- | --- | --- |
| **Distillation** | A *student* model learns to imitate a *teacher's* output distribution | Yes, the student is trained |
| **Quantization** | The same numbers are stored or computed with fewer representable values | Not in the simple post-training form used here |

Distillation is about **learning**; quantization is about **numeric compression**.
They solve different problems and can be combined: distill a smaller model,
then quantize its weights. The script demonstrates the two ideas separately,
on purpose, so you can see each one's arithmetic.

Two honest warnings before we start:

- The "student" in the script is a trainable **2×3 matrix of logits**, not a
  deployable small network. There is no input, no hidden layer, and no
  generalization to new examples. It only shows what the loss pushes toward.
- The quantization part is symmetric per-tensor INT8 **fake quantization** of a
  random 8×8 matrix. It is not NF4, not full QLoRA, and does not run integer
  matrix multiplications.

## 2. The tiny distillation setup in the script

The teacher is fixed. It has two examples ("rows"), each with three class logits:

```python
teacher = torch.tensor([[4., 2., -1.], [-1., 2., 4.]])
```

Row 0 prefers class 0; row 1 is the mirror image and prefers class 2. The
hard labels are `torch.tensor([0, 2])`, which agree with the teacher's top choice.
The student starts at all-zero logits, so initially it says "1/3, 1/3, 1/3"
for every class.

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $z^{t}$ | Teacher logits for one example | `[4, 2, -1]`, `teacher_logits` |
| $z^{s}$ | Student logits for that example | starts at `[0, 0, 0]`, `student_logits` |
| $T$ | Temperature, $T>0$ | `temperature=2.0` |
| $q_T$ | Teacher probabilities at temperature $T$ | `(teacher_logits.detach()/temperature).softmax(-1)` |
| $p_T$ | Student probabilities at temperature $T$ | `exp` of `(student_logits/temperature).log_softmax(-1)` |
| $y$ | Hard label (correct class index) | `labels`, here `[0, 2]` |
| $\alpha$ | Weight on the soft (teacher) loss | `alpha=0.7` |
| $B$ | Batch size (number of rows) | 2 |

## 3. Why can soft labels teach more than a single class?

A hard label says only which class is correct. A teacher distribution might say
class A has probability 0.7, B has 0.25, and C has 0.05. The relative probability
of incorrect classes can communicate similarity or uncertainty unavailable in
a one-hot label: "B is a plausible confusion, C is not." Hinton et al. called
this information "dark knowledge". A mistaken teacher can also transfer its
errors, so soft labels are only as good as the teacher.

## 4. What does temperature do?

At temperature $T>0$, teacher and student probabilities are

$$
\boxed{q_i=\operatorname{softmax}(z^{t}/T)_i,\qquad
p_i=\operatorname{softmax}(z^{s}/T)_i.}
$$

Dividing logits by $T>1$ shrinks the gaps between them, so the distribution
becomes flatter ("softer"). Worked example for the teacher row `[4, 2, -1]`:

| $T$ | Scaled logits | $e^{\text{scaled}}$ | Sum | Probabilities $q_T$ |
| --- | --- | --- | --- | --- |
| 1 | `[4, 2, -1]` | `[54.598, 7.389, 0.368]` | 62.355 | `[0.8756, 0.1185, 0.0059]` |
| 2 | `[2, 1, -0.5]` | `[7.389, 2.718, 0.607]` | 10.714 | `[0.6897, 0.2537, 0.0566]` |
| 4 | `[1, 0.5, -0.25]` | `[2.718, 1.649, 0.779]` | 5.146 | `[0.5283, 0.3204, 0.1513]` |

For example, at $T=2$: $7.389/10.714=0.6897$. At $T=1$ class 2 gets only 0.6%
of the mass; at $T=2$ it gets 5.7%. Softer targets expose differences among
lower-ranked classes instead of almost all mass going to the winner. Relative
**order** never changes, because dividing by a positive $T$ preserves order.

## 5. The soft-target loss: KL divergence from teacher to student

The soft loss is the Kullback–Leibler divergence

$$
\boxed{D_{KL}(q\|p)=\sum_i q_i(\log q_i-\log p_i).}
$$

It is zero exactly when $p=q$ and positive otherwise. Because teacher
probabilities are fixed, the $\sum_i q_i\log q_i$ part (minus the teacher's
entropy) is a constant with respect to the student. Minimizing this KL is
therefore equivalent to minimizing the cross-entropy $-\sum_i q_i\log p_i$:
it encourages the student to match the teacher distribution.

`teacher_logits.detach()` enforces that the teacher is not optimized by the
student loss. In this script the teacher is a plain tensor that never required
gradients, so `detach()` is a safety habit here; it matters when the teacher
is itself a network that is also being trained or shares parameters.

**Worked example at initialization ($T=2$).** The student is uniform,
$p=[1/3,1/3,1/3]$, and $q=[0.6897,0.2537,0.0566]$:

| Class $i$ | $q_i$ | $q_i/p_i=3q_i$ | $\log(3q_i)$ | $q_i\log(3q_i)$ |
| --- | --- | --- | --- | --- |
| 0 | 0.6897 | 2.0690 | 0.7271 | 0.5014 |
| 1 | 0.2537 | 0.7611 | −0.2729 | −0.0692 |
| 2 | 0.0566 | 0.1698 | −1.7731 | −0.1004 |
| Sum | 1 | | | **0.3318** |

Individual terms can be negative; the total KL cannot.

## 6. The combined objective and the $T^2$ factor

The script mixes soft and hard targets:

$$
\boxed{L=\alpha\,T^2\,D_{KL}(q_T\|p_T)
+(1-\alpha)\operatorname{CE}(z^{s},y),\qquad \alpha=0.7,\ T=2.}
$$

The hard cross-entropy uses the student's **untempered** logits ($T=1$), as in
`nn.functional.cross_entropy(student_logits, labels)`.

**Worked initial loss.** Both rows give the same numbers by symmetry, so the
batch average equals one row's value:

$$
L=0.7\cdot4\cdot0.3318+0.3\cdot\ln 3=0.9291+0.3296=1.2587.
$$

Running the script's `distillation_loss` on the zero student indeed returns
`1.2587`.

**Why $T^2$?** The derivative of the unscaled soft loss with respect to a
student logit is

$$
\frac{\partial D_{KL}(q_T\|p_T)}{\partial z^s_i}=\frac{p_i-q_i}{T}.
$$

One factor $1/T$ comes from the chain rule through $z/T$. At high temperatures,
probability differences $p_i-q_i$ also tend to shrink roughly with $1/T$, so the
soft gradient shrinks roughly like $1/T^2$, motivating the usual $T^2$
compensation. It is a heuristic for keeping soft and hard terms on a comparable
scale, not an exact statement that all gradient magnitudes remain invariant at
every temperature.

**Worked initial gradient (row 0).** With $T^2$ scaling, the soft gradient is
$\alpha T(p_i-q_i)$; the hard gradient is $(1-\alpha)(\operatorname{softmax}(z^s)_i-[i=y])$;
both are divided by $B=2$ because of the batch average:

| Class | Soft: $0.7\cdot2\,(p-q)/2$ | Hard: $0.3\,(p-\text{onehot})/2$ | Total gradient |
| --- | --- | --- | --- |
| 0 | $0.7(0.3333-0.6897)=-0.2494$ | $0.15(0.3333-1)=-0.1000$ | **−0.3494** |
| 1 | $0.7(0.3333-0.2537)=+0.0557$ | $0.15(0.3333)=+0.0500$ | **+0.1057** |
| 2 | $0.7(0.3333-0.0566)=+0.1937$ | $0.15(0.3333)=+0.0500$ | **+0.2437** |

These match `student.grad` from PyTorch for the first row. Gradient descent
subtracts the gradient, so class 0's logit goes up and classes 1 and 2 go down,
class 2 most strongly. (Adam rescales each coordinate, so its actual step sizes
differ, but the signs are the same on the first step.)

## 7. PyTorch details that are easy to get wrong

`nn.functional.kl_div(input, target)` expects **log** student probabilities as
`input` and ordinary teacher probabilities as `target`. It computes
$\sum \text{target}\cdot(\log\text{target}-\text{input})=D_{KL}(q\|p)$. Passing
the arguments backwards would compute a different KL direction (or nonsense,
if probabilities are passed where log-probabilities are expected).

`reduction='batchmean'` sums class contributions and averages over examples,
giving the mathematical KL per example. The default `'mean'` would also divide
by the number of classes.

## 8. What does training produce?

The script runs 100 Adam steps with learning rate 0.05 and prints
`student.softmax(-1)`, the student's probabilities at **$T=1$**, not at $T=2$.
Observed output for row 0, compared with the teacher:

| Quantity | Class 0 | Class 1 | Class 2 |
| --- | --- | --- | --- |
| Teacher at $T=1$ | 0.8756 | 0.1185 | 0.0059 |
| Student at $T=1$ (printed) | 0.8960 | 0.0957 | 0.0083 |
| Teacher at $T=2$ (soft target $q$) | 0.6897 | 0.2537 | 0.0566 |
| Student at $T=2$ | 0.7027 | 0.2297 | 0.0676 |

Row 1 prints the mirror image `[0.0083, 0.0957, 0.8960]`. What to notice:

- The student's ranking matches the teacher and the labels.
- The student does **not** copy the teacher exactly. The hard-label term pulls
  extra probability onto the correct class (0.896 vs. 0.876), so the optimum is
  a compromise between the two terms; the final loss is about 0.0396, not 0.
- 100 steps are a short run, not a convergence test. In a separate check (not
  part of the script), 2000 steps gave approximately `[0.898, 0.096, 0.0055]`.

## 9. Symmetric INT8 quantization: the formula

Now the second, unrelated idea. For a floating weight tensor $W$, choose one
scale for the whole tensor:

$$
\boxed{s=\frac{\max|W|}{127},\qquad
q=\operatorname{clip}(\operatorname{round}(W/s),-127,127),\qquad
\hat W=s\,q.}
$$

| Symbol | Meaning | Code |
| --- | --- | --- |
| $s$ | Scale: float value of one integer step | `scale` |
| $q$ | Stored integer code in $[-127,127]$ | `integers` (dtype `torch.int8`) |
| $\hat W$ | Dequantized (reconstructed) float weights | `reconstructed` |
| $W-\hat W$ | Quantization error | `weights-reconstructed` |

"Symmetric" means zero maps exactly to integer 0 and there is no separate zero
point (offset). INT8 can represent −128, but this scheme uses only
$[-127,127]$ so positive and negative ranges are equal. The script also guards
against an all-zero tensor with `clamp_min(1e-8)` so it never divides by zero.

## 10. Worked example: quantize five weights by hand

Take $W=[0.30,\,-1.00,\,0.05,\,0.62,\,-0.004]$. Then $\max|W|=1$ and
$s=1/127\approx0.0078740$.

| $W$ | $W/s$ | $q=\operatorname{round}$ | $\hat W=q/127$ | Error $W-\hat W$ |
| --- | --- | --- | --- | --- |
| 0.30 | 38.10 | 38 | 0.299213 | +0.000787 |
| −1.00 | −127.00 | −127 | −1.000000 | 0 |
| 0.05 | 6.35 | 6 | 0.047244 | +0.002756 |
| 0.62 | 78.74 | 79 | 0.622047 | −0.002047 |
| −0.004 | −0.508 | −1 | −0.007874 | +0.003874 |

The largest-magnitude weight is reconstructed exactly. Mean squared error:

$$
\text{MSE}=\tfrac15(0.000787^2+0+0.002756^2+0.002047^2+0.003874^2)
\approx5.48\times10^{-6}.
$$

Without clipping, nearest rounding gives absolute error at most $s/2\approx0.003937$
per weight, ignoring floating-point roundoff; the worst entry above (0.003874)
is just under that bound. Notice that −0.004 became −0.0079: tiny weights have
large **relative** error even when their absolute error is small.

(Detail: `torch.round` rounds exact halves to the nearest even integer, e.g.
2.5 → 2. Exact halves are rare with real weights.)

## 11. Why can one outlier ruin a per-tensor scale?

One global scale can be dominated by an outlier. Consider a 2×3 matrix whose
first row has small weights and whose second row contains 100:

$$
W=\begin{bmatrix}0.012&-0.02&0.015\\100&-40&3\end{bmatrix}.
$$

**Per-tensor:** $s=100/127\approx0.7874$. Row 0 becomes
$[0.0152,-0.0254,0.0190]/1\to$ codes `[0, 0, 0]`. Every small weight is erased.
Row 1 becomes codes `[127, -51, 4]`, reconstructed as `[100, -40.157, 3.150]`.

**Per-row (per-channel):** row 0 gets its own scale $0.02/127\approx0.00015748$,
codes `[76, -127, 95]`, reconstructed as `[0.011969, -0.02, 0.014961]`; row 1
keeps scale 0.7874.

| Scheme | Row-0 reconstruction | Row-0 relative error | Whole-matrix MSE |
| --- | --- | --- | --- |
| Per-tensor | `[0, 0, 0]` | 100% | 0.007992 |
| Per-row | `[0.011969, -0.02, 0.014961]` | below 0.3% | 0.007864 |

The overall MSE barely changes because row 1's errors dominate it, yet row 0
went from destroyed to nearly exact. A single average error number can hide
which channels lost their information. Per-channel scales are standard
practice for weights for exactly this reason. (These per-row numbers are a
hand calculation for the exercise; the script itself only does per-tensor.)

## 12. What the script's quantization demonstrates, and what it does not

The script draws `weights = torch.randn(8, 8)` (deterministic because of
`torch.manual_seed(7)` and because Adam consumes no random numbers). In the
verification environment (PyTorch 2.14.1, CPU), $\max|W|\approx2.9662$, so
$s\approx0.023356$. The printed MSE was about $4.98\times10^{-5}$. A rough
sanity check: if rounding errors were uniform on $[-s/2,s/2]$, the expected
MSE would be $s^2/12\approx4.55\times10^{-5}$, the same order of magnitude.
The largest error was 0.01160, below $s/2\approx0.01168$.

Storage arithmetic (not measured by the script): 64 float32 values need
256 bytes; 64 int8 codes need 64 bytes plus one 4-byte scale, about a 3.8×
reduction for this tiny tensor.

What the script does **not** do: it stores an INT8 tensor and then
reconstructs float values before any use. It does not use accelerated integer
matrix multiplication kernels, pack a complete model, quantize activations, or
implement NF4/double quantization as used in QLoRA. Storage savings and runtime
speed are separate claims that require separate measurement on real hardware.

## 13. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `(student_logits/temperature).log_softmax(-1)` | $\log p_T$, the student's tempered log-probabilities |
| `(teacher_logits.detach()/temperature).softmax(-1)` | $q_T$, fixed teacher soft targets, no gradient to the teacher |
| `nn.functional.kl_div(..., reduction='batchmean')` | $D_{KL}(q_T\|p_T)$ averaged over examples |
| `nn.functional.cross_entropy(student_logits, labels)` | Hard-label loss at $T=1$ |
| `alpha*temperature**2*soft+(1-alpha)*hard` | Combined objective $L$ with $\alpha=0.7$, $T=2$ |
| `teacher = torch.tensor([[4., 2., -1.], [-1., 2., 4.]])` | Two fixed teacher rows |
| `student = nn.Parameter(torch.zeros_like(teacher))` | Student logits, starting uniform |
| `torch.optim.Adam([student], lr=0.05)` | Optimizer, run `for _ in range(100):` |
| `distillation_loss(student, teacher, torch.tensor([0, 2]))` | Labels agree with the teacher's top classes |
| `scale = weights.abs().max().clamp_min(1e-8)/127` | Per-tensor scale $s$ |
| `integers = (weights/scale).round().clamp(-127, 127).to(torch.int8)` | Integer codes $q$ |
| `reconstructed = integers.float()*scale` | Dequantized $\hat W=sq$ |
| `student.softmax(-1)` | Printed student probabilities at $T=1$ |
| `(weights-reconstructed).square().mean()` | Printed quantization MSE |

## 14. Run and investigate

This script needs PyTorch (see the course [README](../README.md) for setup).
Run from the course folder:

```bash
python3 05_advanced/04_distillation_and_quantization.py
```

The script has no command-line flags. Observed output:

```text
student probabilities: [[0.8960..., 0.0956..., 0.0082...], [0.0082..., 0.0956..., 0.8960...]]
quantization MSE: 4.98...e-05
```

Expect the student's preferred classes to match teacher and labels, and a
small reconstruction error on random weights. The script contains no
assertion; you check these properties by reading the output. Exact trailing
digits may differ across PyTorch versions or hardware.

Experiments:

1. Increase one weight to create an outlier (e.g. `weights[0, 0] = 100.`) and
   compare per-tensor versus per-row quantization, using
   `weights.abs().amax(dim=1, keepdim=True)/127` as per-row scales.
2. Change `temperature` to 1 and 4 and compare the student's final probabilities.
3. Set `alpha=1.0` (pure distillation) and check that the student moves closer to
   the teacher's $T=1$ probabilities.

Distillation reference:
[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).

## 15. Practice calculations

1. Teacher logits `[3, 1]` at $T=2$: what is $q$? **Scaled logits `[1.5, 0.5]`,
   $q_0=1/(1+e^{-1})=0.7311$, $q_1=0.2689$.**
2. With a uniform student $p=[0.5,0.5]$, what is $D_{KL}(q\|p)$?
   **$0.7311\ln(1.4621)+0.2689\ln(0.5379)=0.2777-0.1668=0.1109$.**
3. With $\alpha=0.7$ and $T=2$, what is the scaled soft term $\alpha T^2 D_{KL}$?
   **$0.7\cdot4\cdot0.1109=0.3106$.**
4. Weights `[0.5, -0.25, 0.1]` with symmetric INT8: scale, codes, and
   reconstruction? **$s=0.5/127=0.003937$; codes `[127, -64, 25]`
   (from 127, −63.5, 25.4; the exact half −63.5 rounds to the even −64); $\hat W=[0.5, -0.25197, 0.098425]$.**
5. What is the maximum per-weight rounding error in question 4?
   **$s/2\approx0.00197$; the actual largest is $|0.1-0.098425|=0.001575$.**
6. Does quantizing to INT8 by itself make inference faster? **Not necessarily.
   It reduces storage; speed requires kernels and hardware that compute with
   the quantized format. This script dequantizes before use.**

Next: [Evaluation and uncertainty](05_evaluation_and_uncertainty.md).
