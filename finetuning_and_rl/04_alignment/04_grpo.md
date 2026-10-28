# GRPO: compare groups of answers instead of learning a value critic

Companion code: [04_grpo.py](04_grpo.py).
Prerequisites: [PPO and GAE](../03_reinforcement_learning/06_ppo_and_gae.md) for
clipped probability ratios, [REINFORCE and actor-critic](../03_reinforcement_learning/05_reinforce_actor_critic.md)
for baselines, and [RLHF with a KL penalty](02_rlhf_kl_penalty.md) for the frozen
reference policy. The [math primer](../00_start/MATH_PRIMER.md) reviews softmax,
standard deviation, and KL divergence. Read this explanation before running the code.

## 1. What problem are we solving now?

A language model can answer the same prompt in several ways. Some answers are
correct; others are not. With a **verifiable reward** (for example, "does the
final answer match the known solution?") we can score each answer with 1 or 0.

PPO needs an **advantage**: "how much better was this action than usual?" PPO
usually obtains it from a separately trained value network, the **critic**.
GRPO (Group Relative Policy Optimization) avoids the critic with a simple idea:

> Sample a **group** of answers to the **same** prompt, score them all, and judge
> each answer against the other members of its own group.

An answer that beats its siblings gets a positive advantage; one that does worse
gets a negative advantage. The group itself supplies the baseline.

The companion script uses the smallest possible version of this idea:

```text
prompt 0: possible one-token answers {0, 1, 2, 3}, correct answer = 1
prompt 1: possible one-token answers {0, 1, 2, 3}, correct answer = 3
```

Each "answer" is a single token, so a whole response is one action. Each
iteration samples a group of $G=8$ answers per prompt and awards reward 1 to a
correct answer and 0 otherwise. The policy starts uniform: every answer has
probability 0.25. Training should push probability toward answer 1 for prompt 0
and answer 3 for prompt 1.

## 2. Notation: read the symbols before the equations

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $x$ | A prompt (one row of the table) | row index 0 or 1 of `logits` |
| $a_i$ | The $i$-th sampled answer for that prompt | `actions[prompt, i]` |
| $G$ | Group size: answers sampled per prompt | 8, from `.sample((8,))` |
| $r_i$ | Reward of answer $i$: 1 if correct, else 0 | `rewards` |
| $\bar r$ | Mean reward in the group | `rewards.mean(-1, keepdim=True)` |
| $s$ | Population standard deviation of the group's rewards | `rewards.std(-1, ..., unbiased=False)` |
| $\varepsilon$ | Small constant that prevents division by zero | `1e-6` |
| $A_i$ | Group-relative advantage of answer $i$ | `advantages` |
| $\pi_\theta(a\mid x)$ | Current policy: softmax of trainable logits | `logits.log_softmax(-1)` |
| $\pi_{old}$ | Frozen copy of the policy that **generated** the group | `old_logs` |
| $\pi_{ref}$ | Frozen **reference** policy: the uniform starting point | `reference_logs` |
| $\rho_i$ | Ratio $\pi_\theta(a_i\mid x)/\pi_{old}(a_i\mid x)$ | `ratio` |
| $\epsilon$ | PPO clip width, here 0.2 | `ratio.clamp(0.8, 1.2)` |
| $\beta$ | Weight of the reference KL penalty, here 0.02 | `0.02*kl` |

Two different "old" distributions appear, and confusing them is a common mistake:
$\pi_{old}$ changes every iteration (it is whatever generated the latest group),
while $\pi_{ref}$ never changes.

## 3. Step one: turn a group of rewards into advantages

For rewards $r_1,\ldots,r_G$ in one group, the script computes

$$
\bar r=\frac1G\sum_i r_i,\qquad
s=\sqrt{\frac1G\sum_i(r_i-\bar r)^2},\qquad
\boxed{A_i=\frac{r_i-\bar r}{s+\varepsilon}}.
$$

In words: subtract the group average (centering), then divide by the group's
spread (scaling). The script uses the **population** standard deviation
(`unbiased=False`, divide by $G$, not $G-1$) and $\varepsilon=10^{-6}$. Other
implementations may use different normalization conventions.

### Worked example: rewards `[1, 0, 0, 1]`

1. Mean: $\bar r=(1+0+0+1)/4=0.5$.
2. Deviations: $[0.5,-0.5,-0.5,0.5]$; squared: $[0.25,0.25,0.25,0.25]$.
3. Population variance $=0.25$, so $s=0.5$.
4. Advantages: $0.5/0.500001\approx 1$ and $-0.5/0.500001\approx-1$.

Result: approximately `[1, -1, -1, 1]`. The two correct answers are pushed up
and the two wrong answers pushed down by the same amount.

### Worked example: rewards `[0, 0, 0, 1]`

1. $\bar r=0.25$.
2. Deviations: $[-0.25,-0.25,-0.25,0.75]$; squared: $[0.0625,0.0625,0.0625,0.5625]$.
3. Variance $=0.75/4=0.1875$, so $s=\sqrt{0.1875}\approx0.4330$.
4. Advantages: $-0.25/0.4330\approx-0.5774$ (three times) and $0.75/0.4330\approx1.7321$.

The rare successful answer receives a stronger positive relative signal.

### The script's group size: eight answers

For binary rewards with $k$ correct answers out of $G$, the mean is $p=k/G$ and
the population standard deviation is $\sqrt{p(1-p)}$. Some group-of-8 cases:

| Correct out of 8 | $\bar r$ | $s$ | $A$ for a correct answer | $A$ for a wrong answer |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | — (none) | 0 |
| 1 | 0.125 | 0.3307 | 2.6457 | -0.3780 |
| 3 | 0.375 | 0.4841 | 1.2910 | -0.7746 |
| 7 | 0.875 | 0.3307 | 0.3780 | -2.6457 |
| 8 | 1 | 0 | 0 | — (none) |

Check the 3-of-8 row: $s=\sqrt{0.375\times0.625}=\sqrt{0.234375}\approx0.4841$,
so $0.625/0.4841\approx1.2910$ and $-0.375/0.4841\approx-0.7746$. Notice also
that the advantages in any group **sum to zero**: $3(1.2910)+5(-0.7746)\approx0$.

### When every reward is equal

If every reward is equal (all 0 or all 1), every deviation is zero, so every
advantage is zero. Epsilon avoids division by zero ($0/(0+10^{-6})=0$), but it
cannot invent a preference signal. Such a group teaches the policy-gradient term
nothing. More samples, better exploration, or more informative rewards may be
needed when all attempts fail.

At the start of this toy problem the chance that all 8 answers to a prompt are
wrong is $0.75^8\approx0.100$, so roughly one group in ten is uninformative even
on the first iteration. Late in training the opposite happens: once the policy is
almost always right, most groups are all-correct and also give zero advantage.
In the script's actual run (seed 7), 122 of the 200 prompt-groups collected over
100 iterations had all-equal rewards.

### A caveat about the baseline

Because the baseline and scale are computed from the sampled group itself,
this is not simply the action-independent baseline proof from REINFORCE.
Group standardization changes weighting and can introduce finite-group bias;
it is a practical objective choice rather than a universal unbiased estimator.

## 4. Step two: a PPO-style clipped update, without a critic

Store the log-probabilities of the policy that generated the group. For answer
$a_i$, the ratio of new to old probability is

$$
\rho_i=\exp(\log\pi_\theta(a_i\mid x)-\log\pi_{old}(a_i\mid x)).
$$

The minimized teaching loss is

$$
\boxed{L=-\operatorname{mean}_i\min\!\big(\rho_iA_i,\;
\operatorname{clip}(\rho_i,0.8,1.2)A_i\big)
+0.02\,D_{KL}(\pi_\theta\|\pi_{ref})}.
$$

| Part | Interpretation |
| --- | --- |
| $\rho_iA_i$ | Raise probability of positive-advantage answers, lower negative ones |
| $\operatorname{clip}(\rho_i,0.8,1.2)A_i$ | Same, but the ratio's influence is capped at ±20% |
| $\min(\cdot,\cdot)$ | Take the more pessimistic of the two, as in PPO |
| $\operatorname{mean}_i$ | Average over all 16 entries (2 prompts × 8 answers) |
| Leading minus sign | Optimizers minimize; we want to maximize the surrogate |
| $0.02\,D_{KL}$ | Penalty for drifting away from the reference policy |

This is exactly PPO's clipped surrogate with **group advantages replacing
critic-based advantages**. If you need a refresher on why the minimum is used,
see sections 9–10 of the [PPO lesson](../03_reinforcement_learning/06_ppo_and_gae.md).

### Worked clipping example

Take the 3-of-8 advantages $A=1.2910$ (correct) and $A=-0.7746$ (wrong):

| $A$ | $\rho$ | $\rho A$ | $\operatorname{clip}(\rho)A$ | Selected minimum | Gradient through $\rho$? |
| --- | --- | --- | --- | --- | --- |
| 1.2910 | 1.0 | 1.2910 | 1.2910 | 1.2910 | Yes |
| 1.2910 | 1.3 | 1.6783 | $1.2(1.2910)=1.5492$ | 1.5492 | No: clipped, flat |
| 1.2910 | 0.7 | 0.9037 | $0.8(1.2910)=1.0328$ | 0.9037 | Yes |
| -0.7746 | 0.7 | -0.5422 | $0.8(-0.7746)=-0.6197$ | -0.6197 | No: clipped, flat |
| -0.7746 | 1.3 | -1.0070 | $1.2(-0.7746)=-0.9295$ | -1.0070 | Yes |

The pattern matches PPO: once a good answer's probability has already risen
by more than 20%, or a bad answer's probability has already fallen by more than
20%, that sample stops pushing. Moves in the *harmful* direction are never
capped, so the objective stays pessimistic.

### Two frozen distributions with two different jobs

`old_logs` stays fixed for two update epochs, then fresh answers are sampled.
The reference remains the original uniform distribution throughout training.
Old-policy ratios limit changes relative to collection; reference KL penalizes
long-term drift from the starting policy.

Because there are only four actions, the code computes **exact** categorical KL
by summing over all of them:

$$
D_{KL}(\pi_\theta\|\pi_{ref})=\sum_{a=0}^{3}\pi_\theta(a\mid x)
\log\frac{\pi_\theta(a\mid x)}{\pi_{ref}(a\mid x)},
$$

averaged over the two prompts. It does not use a sampled token-level KL estimator.
The actor receives gradients; sampled actions, rewards, and advantages do not
(they are computed under `torch.no_grad()`).

KL example: for a prompt whose policy is $[0.1,0.7,0.1,0.1]$ against the uniform
reference $0.25$,

$$
3(0.1)\ln\frac{0.1}{0.25}+0.7\ln\frac{0.7}{0.25}
=0.3(-0.9163)+0.7(1.0296)=-0.2749+0.7207\approx0.4458.
$$

The largest possible value here is $\ln4\approx1.386$, reached when all
probability sits on one answer. With $\beta=0.02$ the penalty is at most about
0.028, which is small compared with advantages near 1. In this toy run the
reference term therefore slows drift slightly but does not prevent the policy from
becoming nearly deterministic: the final probabilities give KL values of about
1.346 and 1.309.

## 5. Watch the script's first group, step by step

With `torch.manual_seed(7)`, the first iteration samples:

| Prompt | Sampled answers | Rewards | Advantages |
| --- | --- | --- | --- |
| 0 (correct = 1) | `[1, 1, 3, 2, 3, 3, 1, 2]` | `[1,1,0,0,0,0,1,0]` | 1.2910 for the three 1s, -0.7746 otherwise |
| 1 (correct = 3) | `[3, 3, 0, 0, 0, 0, 0, 3]` | `[1,1,0,0,0,0,0,1]` | 1.2910 for the three 3s, -0.7746 otherwise |

These are the exact values the script produces with this seed, matching the
3-of-8 row of the earlier table.

**Update epoch 1.** The policy has not moved yet, so $\pi_\theta=\pi_{old}$ and
every $\rho_i=1$. The surrogate mean is the mean advantage, which is 0, and the
KL to the uniform reference is 0. The printed loss would be about $10^{-8}$,
essentially zero. **A zero loss value does not mean a zero gradient.**

For a softmax policy, $\partial\log\pi(a_i)/\partial u_v=\mathbb 1[a_i=v]-\pi(v)$,
where $u_v$ is the logit of answer $v$. Because the group's advantages sum to zero,
the $-\pi(v)$ terms cancel, and the gradient of the loss for logit $v$ of one
prompt is

$$
\frac{\partial L}{\partial u_v}=-\frac1{16}\sum_{i:\,a_i=v}A_i .
$$

For prompt 0:

| Logit | Samples choosing it | Gradient |
| --- | --- | --- |
| $u_0$ | none | 0 |
| $u_1$ | three, each $A=1.2910$ | $-3(1.2910)/16=-0.2421$ |
| $u_2$ | two, each $A=-0.7746$ | $+2(0.7746)/16=+0.0968$ |
| $u_3$ | three, each $A=-0.7746$ | $+3(0.7746)/16=+0.1452$ |

The script's autograd values are $-0.24206$, $0.09682$, $0.14524$, matching.
Adam's first step moves each logit by about the learning rate in the direction
opposite its gradient sign, so $u_1$ becomes 0.04 and $u_2,u_3$ become -0.04.
($u_0$ moves by about 0.005 rather than 0 because its gradient is a tiny
floating-point residue, about $-1.4\times10^{-9}$, which Adam's per-coordinate
normalization amplifies. This is a numerical side effect, not part of GRPO.)

**Update epoch 2.** The probabilities for prompt 0 are now about
$[0.2533,0.2623,0.2422,0.2422]$, so the ratios are about $0.2623/0.25=1.049$
for correct answers and $0.2422/0.25=0.969$ for the wrong ones. Both lie inside
$[0.8,1.2]$, so no clipping happens. The loss is now about $-0.0388$.

In fact, in this run clipping **never** activates: the largest second-epoch ratio
deviation over all 100 iterations is about 0.066. The clip is present to show
the objective's form; this toy run does not exercise it. Larger learning rates
or more update epochs per group would.

**Over training.** The probabilities of the correct answers (prompt 0, prompt 1)
after selected iterations of the seeded run are:

| Iterations | $\pi(1\mid x_0)$ | $\pi(3\mid x_1)$ |
| --- | --- | --- |
| 10 | 0.583 | 0.545 |
| 25 | 0.914 | 0.892 |
| 50 | 0.976 | 0.976 |
| 100 | 0.994 | 0.988 |

## 6. Shapes and implementation mapping

`logits` has shape `[2,4]`: prompts by possible answers. Sampling with shape
`(8,)` initially gives `[8,2]`; transposing produces `[2,8]`, prompts by group
members. `gather(1,actions)` retrieves the selected answers' log-probabilities.
Normalization along the last dimension keeps each prompt's baseline separate.
Mixing rewards across prompts would compare different tasks rather than answers
to the same task.

## 7. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `logits = nn.Parameter(torch.zeros(2, 4))` | Trainable policy; zeros mean a uniform start |
| `reference_logs = torch.zeros(2, 4).log_softmax(-1)` | Frozen uniform reference $\pi_{ref}$ |
| `correct_answers = torch.tensor([1, 3])` | The verifier's answer key |
| `Categorical(logits=logits).sample((8,)).T` | Group of 8 answers per prompt, shape `[2,8]` |
| `old_logs = old_logs_all.gather(1, actions)` | $\log\pi_{old}(a_i\mid x)$, fixed for this group |
| `rewards = (actions == correct_answers[:, None]).float()` | Binary verifiable reward |
| `group_advantages(rewards)` | $(r_i-\bar r)/(s+10^{-6})$ within each prompt |
| `for _ in range(2):` | Two update epochs reuse the same group |
| `ratio = (new_logs-old_logs).exp()` | $\rho_i$ |
| `torch.minimum(ratio*advantages, ratio.clamp(0.8, 1.2)*advantages)` | Clipped surrogate |
| `kl = (logs_all.exp()*(logs_all-reference_logs)).sum(-1).mean()` | Exact KL to the reference, averaged over prompts |
| `loss = -surrogate.mean()+0.02*kl` | Minimized objective |
| `torch.optim.Adam([logits], lr=0.04)` | Optimizer and step size |

## 8. Extending to language requires additional decisions

Real completions have many tokens and different lengths. A sequence reward
must be assigned to token-level learning terms with explicit masks and reduction
rules. Length normalization, reward scaling, clipping variants, and KL estimators
all affect behavior. This small objective is not a claim to reproduce every
production GRPO variant or current trainer default. See the
[TRL GRPO documentation](https://huggingface.co/docs/trl/en/grpo_trainer) for those
distinctions, and the [DeepSeekMath paper](https://arxiv.org/abs/2402.03300)
listed in [SOURCES](../00_start/SOURCES.md) for the method's original setting.

What the script demonstrates: group-normalized advantages, a clipped ratio, and
an exact reference KL can steer a one-step categorical policy to the answers a
verifier rewards. What it does not demonstrate: anything about multi-token
generation, reasoning quality, sample efficiency, or comparisons with PPO.

## 9. Run it and read the output

Run from the course folder:

```bash
python3 04_alignment/04_grpo.py
```

The script prints the final answer probabilities, one row per prompt. With the
fixed seed, the output is approximately

```text
answer probabilities: [[0.0012, 0.9945, 0.0020, 0.0023], [0.0059, 0.0027, 0.0034, 0.9880]]
```

(the real print shows full floating-point digits). Expect high probability on
answer 1 for prompt 0 and answer 3 for prompt 1.

The assertion checks only that `group_advantages` returns all zeros when every
reward in a `[2,8]` group equals one. It does **not** check that training
converged or that the correct answers won; you read that from the printed table.

Experiments to try:

- Set all rewards to one: the policy-gradient term disappears; only reference
  regularization can act. From the uniform start nothing moves at all (the KL
  and its gradient are zero there); a policy that had already moved would be
  pulled back toward uniform.
- Test a flawed verifier that rewards a wrong action to see why optimizing
  reward does not guarantee optimizing true correctness.
- Raise the learning rate or the number of update epochs and count how often
  `ratio` leaves `[0.8, 1.2]`.

## 10. Practice calculations

1. Rewards `[1,1,0,0,0,0,0,0]`. What are $\bar r$ and $s$?
   **$\bar r=0.25$, $s=\sqrt{0.25\times0.75}\approx0.4330$.**
2. What advantages do the correct and wrong answers receive?
   **$0.75/0.4330\approx1.732$ and $-0.25/0.4330\approx-0.577$.**
3. Rewards `[0,0,0,0]`. What are the advantages, and why?
   **All zero: every reward equals the mean, so every deviation is zero.**
4. $A=1.732$ and $\rho=1.25$. Which surrogate term is selected?
   **$\min(2.165,\,1.2\times1.732=2.078)=2.078$, the clipped one; its gradient through $\rho$ is zero.**
5. $A=-0.577$ and $\rho=1.25$. Which term is selected?
   **$\min(-0.721,\,-0.693)=-0.721$, the unclipped one; the harmful increase is still penalized.**
6. Policy $[0.25,0.25,0.25,0.25]$ against the uniform reference. What is the KL?
   **Zero: the distributions are identical.**
7. Does GRPO in this script train a value network? **No. The group mean replaces
   the critic's baseline; there is no `critic` module anywhere in the file.**

Next: [SDPO, RLSD, and OPSD: learning from a privileged self-teacher](05_sdpo_rlsd_opsd.md).
