# RLHF and KL regularization: improve reward without unrestricted drift

Companion code: [02_rlhf_kl_penalty.py](02_rlhf_kl_penalty.py).
Prerequisites: [reward modeling](01_reward_model.md),
[PPO and GAE](../03_reinforcement_learning/06_ppo_and_gae.md),
[softmax and logs](../01_foundations/01_python_and_math.md), and
[supervised fine-tuning](../02_finetuning/04_huggingface_sft.md).
Read this explanation before running the code.

## 1. What problem are we solving now?

A common RLHF workflow has three stages:

```text
SFT model  --->  reward model from preferences  --->  policy optimization (e.g. PPO)
 (imitate)          (previous lesson)                 (maximize reward, stay close)
```

The frozen **reference policy** is usually the SFT model. The policy being
trained starts as a copy of it. If we simply told the policy “maximize the
reward model's score,” it could drift far from sensible language and exploit
the reward model's mistakes (remember the verbosity shortcut in the previous
lesson). So we add a penalty for moving away from the reference.

This script isolates **that objective** using three possible actions instead
of text. It computes every expectation exactly. It is not a PPO rollout trainer
or a complete autoregressive RLHF pipeline. [PPO](../03_reinforcement_learning/06_ppo_and_gae.md)
is the larger piece that would perform the optimization from sampled responses.

## 2. The tiny example: a three-action bandit

Think of three canned responses to one prompt:

| Action $i$ | Imagined response | Reference prob $q_i$ | Reward $r_i$ |
| --- | --- | --- | --- |
| 0 | bland, safe answer | 0.6 | 0.0 |
| 1 | decent answer | 0.3 | 0.5 |
| 2 | best-scoring answer | 0.1 | 1.0 |

The reference (SFT) model mostly gives the bland answer. The reward model
prefers action 2. The penalty strength is $\beta=0.2$.

Under the reference, expected reward is $0.6(0)+0.3(0.5)+0.1(1)=0.25$.

## 3. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $p_i$ or $\pi(i)$ | Trainable policy probability of action $i$ | `probs` |
| $q_i$ or $\pi_{ref}(i)$ | Frozen reference probability | `reference = [0.6, 0.3, 0.1]` |
| $r_i$ | Fixed reward of action $i$ | `reward = [0.0, 0.5, 1.0]` |
| $\beta$ | KL penalty strength | `beta = 0.2` |
| $z_i$ | Unconstrained logit; $p=\operatorname{softmax}(z)$ | `logits` |
| $D_{KL}(p\|q)$ | Forward KL from policy to reference | `kl` |
| $J(p)$ | Penalized objective to **maximize** | `objective` |
| $p^*$ | Exact maximizer of $J$ | `exact_optimum` |

## 4. Define the objective and the KL direction

Maximize expected reward minus a KL penalty:

$$\boxed{J(p)=\sum_i p_i r_i-\beta\, D_{KL}(p\|q)},\qquad
D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}.$$

- $\sum_i p_i r_i$ is the expected reward when actions are sampled from $p$.
- $\log(p_i/q_i)$ is the **log-ratio**: positive when the policy likes action $i$
  more than the reference did, negative when less.
- KL is the policy-weighted average of those log-ratios.

Assume $q_i>0$ for the allowed actions. KL is nonnegative and is zero when the
distributions agree. It is directional: $D_{KL}(p\|q)$ generally differs from
$D_{KL}(q\|p)$.

**Worked example.** Take a candidate policy $p=[0.2,0.3,0.5]$.

| $i$ | $p_i$ | $q_i$ | $\log(p_i/q_i)$ | $p_i\log(p_i/q_i)$ | $p_ir_i$ |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.2 | 0.6 | $\log(1/3)=-1.0986$ | $-0.2197$ | 0 |
| 1 | 0.3 | 0.3 | 0 | 0 | 0.15 |
| 2 | 0.5 | 0.1 | $\log 5=1.6094$ | 0.8047 | 0.5 |
| Sum | | | | KL $=0.5850$ | $E[r]=0.65$ |

$$J=0.65-0.2(0.5850)=0.65-0.1170=0.5330.$$

Notice that action 0's log-ratio is negative. A particular sampled log-ratio may
be negative even though its expectation under $p$ is nonnegative. The other
direction, $D_{KL}(q\|p)=0.6\log3+0+0.1\log0.2=0.4982$, is a different number.

## 5. Derive the exact best distribution

We maximize $J$ subject to $\sum_i p_i=1$. Introduce a Lagrange multiplier
$\lambda$ for that constraint:

$$
\mathcal L=\sum_i p_ir_i-\beta\sum_i p_i\log(p_i/q_i)
+\lambda\Big(\sum_i p_i-1\Big).
$$

Setting the derivative with respect to $p_i$ to zero gives:

$$
r_i-\beta[\log(p_i/q_i)+1]+\lambda=0
\quad\Longrightarrow\quad
\log\frac{p_i}{q_i}=\frac{r_i}{\beta}+\text{const}.
$$

Exponentiate, then normalize so the probabilities sum to one:

$$\boxed{p_i^*=\frac{q_i e^{r_i/\beta}}{\sum_jq_j e^{r_j/\beta}}
=\operatorname{softmax}_i(\log q+r/\beta)}$$

In words: **start from the reference and tilt it exponentially toward reward.**
The denominator $Z=\sum_j q_je^{r_j/\beta}$ is the normalizer (the DPO lesson
calls it $Z(x)$).

**Step-by-step for $\beta=0.2$:** $r/\beta=[0,\,2.5,\,5]$.

| $i$ | $q_i$ | $e^{r_i/\beta}$ | $q_ie^{r_i/\beta}$ | $p_i^*$ |
| --- | --- | --- | --- | --- |
| 0 | 0.6 | 1 | 0.6 | $0.6/19.0961=0.0314$ |
| 1 | 0.3 | 12.1825 | 3.6547 | 0.1914 |
| 2 | 0.1 | 148.4132 | 14.8413 | 0.7772 |
| Sum | | | $Z=19.0961$ | 1 |

The unnormalized weights are approximately `[0.6,3.6547,14.8413]`, and
normalizing gives `[0.0314,0.1914,0.7772]`. The high-reward action becomes
much more probable, but the reference affects how much probability each action
receives: action 1 keeps 19% rather than collapsing to zero.

At this optimum, $E[r]=0.8729$, $D_{KL}=1.4149$, and
$J^*=0.8729-0.2(1.4149)=0.5899$. A neat identity: $J^*=\beta\log Z=0.2\log19.0961=0.5899$.

## 6. Compare three policies

| Policy | $p$ | $E[r]$ | $D_{KL}(p\|q)$ | $J$ with $\beta=0.2$ |
| --- | --- | --- | --- | --- |
| Stay at reference | `[0.6, 0.3, 0.1]` | 0.25 | 0 | 0.25 |
| Candidate from section 4 | `[0.2, 0.3, 0.5]` | 0.65 | 0.5850 | 0.5330 |
| Always pick action 2 | `[0, 0, 1]` | 1.00 | $\log 10=2.3026$ | 0.5395 |
| Exact optimum | `[0.0314, 0.1914, 0.7772]` | 0.8729 | 1.4149 | **0.5899** |

The greedy policy gets the most reward but pays a large KL price. The optimum
gives up a little reward in exchange for staying closer to the reference.

## 7. How does beta change the answer?

Larger beta penalizes drift more strongly; smaller beta allows reward to
dominate more strongly. Exact optima from the boxed formula:

| $\beta$ | $p^*$ | $E[r]$ | $D_{KL}$ | $J^*$ |
| --- | --- | --- | --- | --- |
| 0.05 | `[0.0000, 0.0001, 0.9999]` | 0.9999 | 2.3011 | 0.8849 |
| 0.1 | `[0.0003, 0.0198, 0.9799]` | 0.9898 | 2.1806 | 0.7718 |
| 0.2 | `[0.0314, 0.1914, 0.7772]` | 0.8729 | 1.4149 | 0.5899 |
| 0.5 | `[0.2785, 0.3785, 0.3430]` | 0.5322 | 0.2970 | 0.3838 |
| 1.0 | `[0.4391, 0.3620, 0.1989]` | 0.3799 | 0.0677 | 0.3122 |
| 5.0 | `[0.5694, 0.3147, 0.1159]` | 0.2732 | 0.0023 | 0.2615 |

As $\beta\to0$ the optimum becomes greedy; as $\beta\to\infty$ it returns to
the reference. Always report reward **and** KL together: a higher reward with
a huge KL may simply mean the policy found a reward-model loophole.

**Reward scale and shift.** Only $r/\beta$ appears in $p^*$. If you multiply all
rewards by ten without changing beta, reward dominates more strongly; it is
exactly the same as using $\beta=0.02$, and the optimum becomes essentially
`[0, 0, 1]`. Reward scaling therefore changes the effective tradeoff. Adding the
same constant to every reward does not change the best policy: it multiplies
every weight $q_ie^{r_i/\beta}$ by the same factor, which cancels in the
normalization (equivalently, it adds only a constant to expected reward).

This regularization can preserve behavior statistically, but cannot guarantee
that all original capabilities remain intact.

## 8. Why optimize logits, and what is the first gradient?

The code stores unconstrained logits and converts them with log-softmax.
This automatically keeps $p_i>0$ and $\sum_i p_i=1$ without a separate
constrained optimizer. `probs * reward` computes the exact expected reward;
`probs * (log_probs-reference.log())` computes exact forward KL.
Adam minimizes `-objective`, which is the same as maximizing $J$.

The logits start at `reference.log()`, so the initial policy **is** the
reference: $p=q$, KL $=0$, $J=0.25$.

By the softmax chain rule, with $g_i=r_i-\beta\log(p_i/q_i)$,

$$
\frac{\partial J}{\partial z_i}=p_i\Big(g_i-\sum_j p_jg_j\Big).
$$

At the start, $p=q$ so $g=r$ and $\sum_jq_jr_j=0.25$:

$$
\nabla_z J=[0.6(0-0.25),\ 0.3(0.5-0.25),\ 0.1(1-0.25)]=[-0.15,\ 0.075,\ 0.075].
$$

Below-average-reward action 0 is pushed down; the other two are pushed up. Note
the KL term contributes zero gradient at $p=q$; it only starts resisting once
the policy has moved. After one Adam step the policy is
`[0.5758, 0.3182, 0.1061]` with $J=0.2649$.

## 9. What does training actually reach?

Training trajectory from the script's settings (Adam, lr 0.05, $\beta=0.2$):

| Step | Policy | $J$ |
| --- | --- | --- |
| 0 | `[0.6000, 0.3000, 0.1000]` | 0.2500 |
| 50 | `[0.0749, 0.1790, 0.7461]` | 0.5854 |
| 100 | `[0.0423, 0.1888, 0.7689]` | 0.5895 |
| 150 (printed) | `[0.0356, 0.1893, 0.7751]` | 0.58984 |
| exact optimum | `[0.0314, 0.1914, 0.7772]` | 0.58990 |

After 150 steps the policy is close to, but not exactly at, the optimum: action 0
still has 0.0356 instead of 0.0314. The objective is already within about
0.00006 of its maximum, because $J$ is flat near its peak. (Continuing the same
loop for 1000 steps in a scratch run matches the optimum to four decimals; the
script itself stops at 150.) The final `exact_optimum` comparison is an
independent mathematical check on the learned distribution.

## 10. From actions to text

For a complete response, autoregressive probability is a product over generated
tokens, so the log-ratio is a sum:

$$
\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)}
=\sum_t[\log\pi(y_t\mid x,y_{<t})-
\log\pi_{ref}(y_t\mid x,y_{<t})].
$$

**Example.** A two-token response where the policy assigns 0.5 and 0.6 and the
reference assigns 0.4 and 0.8:

| Token | $\log\pi$ | $\log\pi_{ref}$ | Log-ratio |
| --- | --- | --- | --- |
| 1 | $\log0.5$ | $\log0.4$ | $+0.2231$ |
| 2 | $\log0.6$ | $\log0.8$ | $-0.2877$ |
| Sum | | | $-0.0645$ |

This single-sample estimate is negative, even though expected KL is never
negative. That is why sampled KL penalties are noisy and only average out to
the true KL.

Mask out prompt and padding positions (as in
[instruction masks](../02_finetuning/02_instruction_data_and_masks.md)). In
sampled LLM training, this quantity can supply a reference penalty alongside a
sequence reward. PPO additionally needs an **old rollout policy** for importance
ratios. The reference policy and the old policy serve different roles and are
not interchangeable:

| Policy | Role | Updated? |
| --- | --- | --- |
| Reference $\pi_{ref}$ | Anchor for the KL penalty (usually SFT model) | Never |
| Old policy $\pi_{old}$ | Generated the current rollout batch; PPO ratio denominator | Replaced after each collection |
| Current policy $\pi_\theta$ | Being trained | Every step |

**What the script does not show:** sampling, token sequences, a value critic,
PPO clipping, or a learned reward model that can be hacked. The comment about
reward hacking is a general warning: higher learned reward need not imply
higher human quality.

## 11. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `reference = torch.tensor([0.6, 0.3, 0.1])` | Frozen reference $q$ |
| `reward = torch.tensor([0.0, 0.5, 1.0])` | Fixed rewards $r$ |
| `logits = torch.nn.Parameter(reference.log().clone())` | Trainable $z$, initialised so $p=q$ |
| `beta = 0.2` | KL penalty strength |
| `log_probs = logits.log_softmax(-1)` | $\log p_i$, always a valid distribution |
| `kl = (probs*(log_probs-reference.log())).sum()` | Exact $D_{KL}(p\|q)$ |
| `objective = (probs*reward).sum()-beta*kl` | $J(p)=E_p[r]-\beta D_{KL}$ |
| `(-objective).backward()` | Gradient ascent on $J$ via descent on $-J$ |
| `for _ in range(150):` | Fixed 150 Adam steps, lr 0.05 |
| `exact_optimum = (reference.log()+reward/beta).softmax(-1)` | Closed-form $p^*=\operatorname{softmax}(\log q+r/\beta)$ |

## 12. Run and experiment

Run from the course folder:

```bash
python3 04_alignment/02_rlhf_kl_penalty.py
```

Output:

```text
learned policy: [0.03564076125621796, 0.1892627477645874, 0.7750964760780334] analytic optimum: [0.03142009302973747, 0.19138753414154053, 0.7771924734115601]
```

The script has no command-line flags and no assertion. You compare the two lists
by eye: the learned distribution should approach the analytic optimum, and here
it agrees to about 0.004 in every entry.

Experiments (edit a copy of the script):

1. Sweep beta and compare reward and KL together; check against section 7.
2. Multiply the rewards by ten and confirm the result matches $\beta=0.02$.
3. Add 3 to every reward and confirm the optimum does not change.
4. Increase the step count and watch the gap to the optimum shrink.

## 13. Practice calculations

1. For $p=[0.3,0.3,0.4]$, what are $E[r]$, $D_{KL}(p\|q)$, and $J$ with $\beta=0.2$?
   **$E[r]=0.55$; KL $=0.3\log0.5+0+0.4\log4=0.3466$; $J=0.55-0.0693=0.4807$.**
2. What is the optimum for $\beta=1$?
   **Weights $[0.6,\,0.3e^{0.5},\,0.1e]=[0.6,0.4946,0.2718]$, sum 1.3664; $p^*=[0.4391,0.3620,0.1989]$.**
3. Why is KL zero at the start of training? **The logits are initialised to $\log q$, so $p=q$.**
4. A token has $\pi=0.2$ and $\pi_{ref}=0.5$. Its log-ratio? **$\log0.4=-0.9163$.**
5. Rewards become $r+7$. Does $p^*$ change? **No; the common factor $e^{7/\beta}$ cancels.**
6. Which policy is the PPO ratio denominator: reference or old? **The old rollout policy.**

Next: [DPO derives a preference loss from this optimum](03_dpo.md).
