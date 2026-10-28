# DPO: derive preference optimization from a regularized reward objective

Companion code: [03_dpo.py](03_dpo.py).
Prerequisites: [pairwise reward likelihood](01_reward_model.md),
[the KL-regularized optimum](02_rlhf_kl_penalty.md), and
[instruction data and loss masks](../02_finetuning/02_instruction_data_and_masks.md).
Read this explanation before running the code.

## 1. What problem does DPO simplify?

The previous two lessons describe a two-stage pipeline:

```text
preferences --> fit reward model r_phi --> sample responses, optimize policy with RL + KL penalty
```

That means training and storing an extra model, generating fresh responses
during training, and tuning an RL algorithm such as PPO. **Direct Preference
Optimization (DPO)** trains a policy directly from fixed
prompt/chosen/rejected triples and a frozen reference. No online reward model
or rollout collection is required for this objective.

DPO is offline preference optimization; it is not identical to on-policy
reinforcement learning. It reuses the same preference data a reward model would
use, but turns the policy's own log-probabilities into an *implicit* reward.
The formulation comes from the [DPO paper](https://arxiv.org/abs/2305.18290).

## 2. The tiny example in the script

The script's “responses” are three categorical actions for one implicit prompt:

| Index | Role in the preference pair | Reference prob | Starting policy prob |
| --- | --- | --- | --- |
| 0 | rejected $y_l$ | 1/3 | 1/3 |
| 1 | neutral (never compared) | 1/3 | 1/3 |
| 2 | chosen $y_w$ | 1/3 | 1/3 |

There is exactly one preference pair, “2 is better than 0,” repeated for 100
training steps. The reference is uniform and frozen; the policy starts uniform
(all logits zero). Watch what happens to the neutral action, which no label
mentions.

## 3. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $x$ | Prompt | implicit in the script |
| $y_w$, $y_l$ | Chosen and rejected responses | indices 2 and 0 |
| $\pi_\theta(y\mid x)$ | Trainable policy probability | `logits.softmax(-1)` |
| $\pi_{ref}(y\mid x)$ | Frozen reference probability | `reference_logs.exp()` = 1/3 each |
| $\log\frac{\pi_\theta}{\pi_{ref}}$ | Policy log-ratio for one response | `policy_chosen-ref_chosen` |
| $m$ | Unscaled margin: chosen log-ratio minus rejected log-ratio | `margin` |
| $\beta$ | Strength of the implicit KL constraint | `beta=0.1` |
| $z$ | Scaled margin $\beta m$ | `beta*margin` |
| $\hat r(x,y)$ | Implicit reward $\beta\log\frac{\pi_\theta}{\pi_{ref}}$ | not printed by the script |
| $Z(x)$ | Normaliser of the KL-regularized optimum | cancels; never computed |

## 4. Derivation: replace rewards with policy log-ratios

**Step 1: recall the optimum.** In the [previous lesson](02_rlhf_kl_penalty.md)
we found that maximizing $E[r]-\beta D_{KL}(\pi\|\pi_{ref})$ gives, for each
prompt $x$:

$$
\pi^*(y\mid x)=\frac{\pi_{ref}(y\mid x)e^{r(x,y)/\beta}}{Z(x)}.
$$

**Step 2: solve for the reward.** Take logs and rearrange:

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)}+\beta\log Z(x).
$$

So a reward and its optimal policy carry the same information, up to
$\beta\log Z(x)$.

**Step 3: plug into Bradley–Terry.** The [reward-model lesson](01_reward_model.md)
used $P(y_w\succ y_l)=\sigma(r_w-r_l)$. For two responses to the **same**
prompt, the unknown $\beta\log Z(x)$ terms cancel in $r_w-r_l$. This is exactly
the “only differences matter” property from section 8 of that lesson.

**Step 4: replace $\pi^*$ by the trainable model $\pi_\theta$** and maximize the
likelihood of the observed preferences:

$$\boxed{z=\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right],\qquad
L_{DPO}=-\mathbb E\big[\log\sigma(z)\big]}$$

This is the objective implemented in `dpo_loss`. In code, `margin` is the
bracket (which we call $m$), and the loss is `-logsigmoid(beta*margin).mean()`.

The preference model and regularized-optimum assumptions motivate the
transformation; they do not imply that any finite noisy dataset recovers human
utility perfectly.

## 5. A numerical example and the loss value

Suppose the reference assigns chosen and rejected probabilities 0.2 and 0.4,
while the current model assigns 0.4 and 0.2.

| Response | $\pi_\theta$ | $\pi_{ref}$ | Log-ratio |
| --- | --- | --- | --- |
| chosen | 0.4 | 0.2 | $\log 2=0.6931$ |
| rejected | 0.2 | 0.4 | $\log 0.5=-0.6931$ |

Then the unscaled margin is:

$$
m=\log(0.4/0.2)-\log(0.2/0.4)=\log4\approx1.3863.
$$

With beta 0.1, $z=0.1386$, preference probability is $\sigma(0.1386)\approx0.5346$,
and loss is $-\log0.5346\approx0.6262$. If the policy equals the reference,
both log-ratios are zero, $z=0$, and the loss is $\log2=0.6931$.

**What is being compared?** Not the raw probabilities, but how much the policy
has moved each response **relative to the reference**. If the reference already
preferred the chosen answer, the policy gets no credit for merely keeping that
preference.

**Sequence-scale example.** Real responses have summed log-probabilities like
$-12$. Suppose $\log\pi_\theta(y_w)=-12$, $\log\pi_\theta(y_l)=-15$,
$\log\pi_{ref}(y_w)=-13$, $\log\pi_{ref}(y_l)=-14$. The chosen log-ratio is
$+1$, the rejected is $-1$, so $m=2$.

| $\beta$ | $z=\beta m$ | $\sigma(z)$ | Loss | $\partial L/\partial m$ |
| --- | --- | --- | --- | --- |
| 0.1 | 0.2 | 0.5498 | 0.5981 | $-0.0450$ |
| 0.5 | 1.0 | 0.7311 | 0.3133 | $-0.1345$ |
| 1.0 | 2.0 | 0.8808 | 0.1269 | $-0.1192$ |

## 6. Which way does the gradient push?

Let $m$ be the unscaled margin. Its derivative is:

$$\boxed{\frac{\partial L}{\partial m}=\beta[\sigma(\beta m)-1]}$$

This is negative, so descent encourages increasing the chosen log-probability
relative to the rejected one and relative to the reference preference odds.
For the first example ($m=1.3863$, $\beta=0.1$): $0.1(0.5346-1)=-0.0465$.

Through the chain rule, the parameter gradient is

$$
\nabla_\theta L=-\beta\,\sigma(-z)\big[\nabla_\theta\log\pi_\theta(y_w)-\nabla_\theta\log\pi_\theta(y_l)\big],
$$

using $1-\sigma(z)=\sigma(-z)$. The weight $\sigma(-z)$ is large when the implicit
reward ranks the pair wrongly and small when it already ranks it confidently,
just like the reward-model gradient in the previous lessons.

Because probabilities are normalized and model parameters are shared, this
does not guarantee the absolute probability of every chosen response always rises.
DPO only promises to widen the *gap*; in real LLM training both log-probabilities
can fall while the gap still grows.

**Beta has two roles to distinguish.** In the ideal regularized optimum it
controls the reward/KL tradeoff: small beta means a weak tether to the reference.
Inside the fixed DPO loss it also changes margin scale and gradient saturation.
The table in section 5 shows the gradient is not monotonic in beta: going from
0.5 to 1.0 *shrinks* it because the sigmoid saturates. A larger beta is not
simply “a larger learning rate.”

## 7. Watch the script's first step by hand

With all logits zero, $\log\pi_\theta=\log\pi_{ref}=\log(1/3)=-1.0986$ for every
action, so $m=0$, $z=0$, and the loss is $\log2=0.6931$.

Because log-softmax subtracts the same log-normaliser from every entry,
$\log\pi_\theta(2)-\log\pi_\theta(0)=z_2-z_0$ (here $z_i$ denotes logits).
The reference is uniform, so its difference is zero. Therefore

$$
L=-\log\sigma\big(0.1(z_2-z_0)\big),\qquad
\frac{\partial L}{\partial z_2}=0.1(\sigma(0)-1)=-0.05,\quad
\frac{\partial L}{\partial z_0}=+0.05,\quad
\frac{\partial L}{\partial z_1}=0.
$$

The logit gradient is `[0.05, 0.0, -0.05]`. **The neutral logit receives exactly
zero gradient** for the whole run. Adam's first step moves each nonzero-gradient
coordinate by about the learning rate 0.05, giving logits `[-0.05, 0, 0.05]`
and probabilities `[0.3168, 0.3331, 0.3501]`.

## 8. What the script ends with

After 100 Adam steps (lr 0.05):

| Quantity | Rejected (0) | Neutral (1) | Chosen (2) |
| --- | --- | --- | --- |
| Final logit | $-4.5324$ | 0 | $4.5324$ |
| Probability | 0.000114 | 0.0106 | 0.9892 |
| Implicit reward $0.1\log(\pi_\theta/\tfrac13)$ | $-0.7977$ | $-0.3445$ | $+0.1088$ |

The final margin is $m=1.0878-(-7.9770)=9.0648$, so $z=0.9065$,
$\sigma(z)=0.7123$, and the loss is still about 0.3393. The loss is nowhere
near zero, yet the policy is already extreme. Two lessons:

1. **The neutral response lost almost all of its probability** (0.333 to 0.0106)
   although no label mentioned it and its logit never moved. Its probability fell
   only because softmax renormalised around the growing chosen logit.
2. **Adam moves at roughly constant speed.** The logits travelled 4.53 of a
   possible $100\times0.05=5$, because Adam normalises gradient size. This clean
   repeated preference can drive the distribution to an extreme; real training
   needs held-out evaluation and attention to label noise and coverage.

**About the exercise in the script's docstring (“make beta smaller and compare
gradient size and final preference”).** The rows below come from edited copies
of the script run separately; they are not outputs of the provided file:

| Variant | Final probabilities `[rejected, neutral, chosen]` | Final loss |
| --- | --- | --- |
| Adam, $\beta=0.01$ | `[0.000049, 0.0070, 0.9930]` | 0.6448 |
| Adam, $\beta=0.1$ (script) | `[0.000114, 0.0106, 0.9892]` | 0.3393 |
| Adam, $\beta=1.0$ | `[0.0140, 0.1106, 0.8754]` | 0.0158 |
| Plain SGD lr 0.05, $\beta=0.1$ | `[0.2552, 0.3267, 0.4182]` | 0.6688 |

The raw gradient is ten times smaller at $\beta=0.01$, but Adam rescales it,
so the smaller-beta run actually ends **more** extreme: its loss never
saturates, so the push never fades. That matches the theory (small beta = weak
tether to the reference). With plain SGD, the raw gradient size directly sets
the step size, and the same 100 steps barely move the policy. Optimizer choice
matters when you interpret beta experiments.

## 9. Sequence likelihood and masking

For response tokens $y_t$, sum conditional log-probabilities:

$$
\log\pi_\theta(y\mid x)=\sum_t\log\pi_\theta(y_t\mid x,y_{<t}).
$$

`sequence_logprob` applies log-softmax over vocabulary, shifts to next-token
alignment, uses `gather` to select the actual target token, multiplies by the
response mask, and sums over positions. This implements a sequence log-probability,
not a token average. Replacing the sum by a mean changes the objective and its
length dependence.

**Worked example** (batch 1, length 4, vocabulary 3). Sequence
`token_ids=[[2, 1, 0, 0]]` = one prompt token, two response tokens, one pad;
`response_mask=[[0, 1, 1, 0]]` marks the response tokens. The logit row at
position $t$ predicts token $t+1$:

| Logit row | Predicts position | Target token | Prob of target | Mask `response_mask[:,1:]` | Contribution |
| --- | --- | --- | --- | --- | --- |
| 0, probs `[0.1, 0.8, 0.1]` | 1 | 1 | 0.8 | 1 | $\log0.8=-0.2231$ |
| 1, probs `[0.6, 0.2, 0.2]` | 2 | 0 | 0.6 | 1 | $\log0.6=-0.5108$ |
| 2, probs `[0.3, 0.3, 0.4]` | 3 | 0 (pad) | 0.3 | 0 | 0 |
| 3 | dropped by `logits[:, :-1]` | | | | |

The function returns $-0.7340$ (checked by calling it). A token mean would give
$-0.3670$, a different objective that changes how response length affects
the margin.

`token_ids` must contain valid vocabulary indices even at masked positions;
use the separate mask rather than placing -100 in IDs passed to `gather`.
Chosen and rejected responses must share the same prompt and consistent template.
Reference computations must be frozen (compute them under `torch.no_grad()` or
precompute them once).

## 10. What the runnable example simplifies

`main` trains a three-action categorical distribution with action 2 preferred
to action 0 and a uniform reference. The helper for sequences is available for
understanding the LLM connection, but the main loop does not run a transformer
and never calls `sequence_logprob`. There is one prompt, one pair, no batching,
no label noise, and no held-out evaluation.

General claims in this lesson about LLM training (chosen probabilities falling,
length effects of mean vs. sum, the need for frozen references) are background
from the method's design, not behaviors the script demonstrates.

## 11. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `logits[:, :-1].log_softmax(-1)` | Next-token log-probabilities; last position has no target |
| `.gather(` then `-1, token_ids[:, 1:, None]).squeeze(-1)` | Log-prob of each actual next token (one call split over two lines) |
| `(token_logs * response_mask[:, 1:]).sum(-1)` | Sum over response tokens only: $\log\pi(y\mid x)$ |
| `margin = (policy_chosen-policy_rejected)-(ref_chosen-ref_rejected)` | Unscaled margin $m$ (algebraically equal to chosen log-ratio minus rejected log-ratio) |
| `-nn.functional.logsigmoid(beta*margin).mean()` | $L_{DPO}=-\mathbb E[\log\sigma(\beta m)]$ |
| `logits = nn.Parameter(torch.zeros(3))` | Trainable policy, starts uniform |
| `reference_logs = torch.zeros(3).log_softmax(-1)` | Frozen uniform reference, $\log(1/3)$ each |
| `dpo_loss(logs[2], logs[0], reference_logs[2], reference_logs[0])` | Preference “2 beats 0”, default `beta=0.1` |
| `torch.optim.Adam([logits], lr=0.05)` | Optimizer; 100 steps |
| `logits.softmax(-1)` | Printed final probabilities |

## 12. Run and investigate

Run from the course folder:

```bash
python3 04_alignment/03_dpo.py
```

Output:

```text
probabilities [rejected, neutral, chosen]: [0.00011442094546509907, 0.010639103129506111, 0.9892464280128479]
```

The script has no command-line flags and no assertion. The chosen action's
probability increases markedly; the rejected action nearly vanishes; the neutral
action shrinks only through normalisation. The course test suite
(`tests/test_learning_invariants.py`) separately checks that `dpo_loss` gives
the winner a negative gradient and the loser a positive one, and that
`sequence_logprob` excludes prompt and padding positions.

Experiments:

1. Reverse the pair (`logs[0], logs[2]`) and observe the reversed learning direction.
2. Add conflicting preferences (also “0 beats 2”) and examine the resulting compromise.
3. Change beta, then swap Adam for SGD, and compare with section 8.

## 13. Practice calculations

1. Policy log-probs: chosen $-20$, rejected $-22$; reference both $-21$; $\beta=0.1$.
   What are $m$, $z$, and the loss? **$m=(-20+22)-(-21+21)=2$, $z=0.2$, loss $0.5981$.**
2. What is $\partial L/\partial m$ there? **$0.1(0.5498-1)=-0.0450$.**
3. The policy equals the reference. What is the loss for any $\beta$? **$\log2=0.6931$.**
4. Both policy log-probs drop by 3 from their reference values. What is $m$?
   **Zero; DPO sees no change in preference.**
5. In the script, what gradient does the neutral logit get? **Exactly 0.**
6. With token log-probs $-0.2231$ and $-0.5108$ on response tokens, what does
   `sequence_logprob` return? **Their sum, $-0.7340$.**

Next: [GRPO: compare groups of answers instead of learning a value critic](04_grpo.md).
