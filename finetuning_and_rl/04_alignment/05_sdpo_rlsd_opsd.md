# SDPO, RLSD, and OPSD: learning from a privileged self-teacher

Companion code: [05_sdpo_rlsd_opsd.py](05_sdpo_rlsd_opsd.py), PyTorch objective labs
(loss functions on supplied tensors, not complete LLM trainers).
Prerequisites: causal language modeling, response masks, policy gradients, PPO,
GRPO, KL direction, and stop-gradient. Read
[tensors and autograd](../01_foundations/02_tensors_autograd.md),
[neural network training](../01_foundations/03_neural_network_training.md),
[tokens and attention](../01_foundations/04_tokens_attention.md),
[a tiny transformer](../01_foundations/05_tiny_transformer.md),
[instruction data and masks](../02_finetuning/02_instruction_data_and_masks.md),
[REINFORCE and actor-critic](../03_reinforcement_learning/05_reinforce_actor_critic.md),
[PPO and GAE](../03_reinforcement_learning/06_ppo_and_gae.md),
[RLHF and KL](02_rlhf_kl_penalty.md), and [GRPO](04_grpo.md) first. The
[math primer](../00_start/MATH_PRIMER.md) reviews KL divergence.

This chapter fixes the meanings of these acronyms to the linked paper versions
(also listed in [SOURCES](../00_start/SOURCES.md)); other papers can reuse the
same acronyms. Statements about what the papers propose are summaries of the
cited versions, not something the script verifies. Read the papers before
relying on any detail for your own experiments.

## 1. What problem are we solving now?

In [GRPO](04_grpo.md) the only learning signal was a single number per answer:
correct (1) or wrong (0). Every token of a long answer gets the same credit,
even though perhaps only one step was the actual mistake.

Suppose we could show the model something it will **not** see at test time:

- the error message from a failing test (feedback), or
- a reference solution to the problem.

With that **privileged information** $z$ in its context, the same model can
judge, token by token, which choices look sensible. That privileged copy acts as
a **self-teacher**. The **student** is the ordinary model, which sees only the
prompt. All three methods in this chapter use the teacher's per-token
probabilities to give the student a richer signal than one scalar reward:

| Method | Privileged $z$ | How the teacher is used |
| --- | --- | --- |
| SDPO | Environment feedback, or successful same-question attempts | Distill: minimize reverse KL, student ‖ teacher |
| OPSD | Reference answer or reasoning trace $y^*$ | Distill: a divergence such as forward KL, teacher ‖ student |
| RLSD | Teacher evidence alongside a verifier reward | Reweight the verifier's group advantage per token |

A tiny example used throughout: a vocabulary of 4 tokens at one position. The
student is uniform (zero logits, exactly as the script initializes it), and the
teacher prefers token 0:

```text
student p = [0.25, 0.25, 0.25, 0.25]
teacher q = [0.40, 0.30, 0.20, 0.10]
```

## 2. Notation and the implementation contract

For prompt $x$, student-generated response $y$, prefix $h_t=(x,y_{<t})$, and
privileged information $z$, define $p_t=\pi_\theta(\cdot\mid h_t)$ and
$q_t=\operatorname{sg}(\pi_{\bar\theta}(\cdot\mid x,z,y_{<t}))$.

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $x$ | Prompt | (outside the lab) |
| $y$, $y_t$ | Response sampled from the student; its $t$-th token | `tokens` |
| $h_t$ | Student prefix: prompt plus earlier response tokens | (outside the lab) |
| $z$ | Privileged information: feedback $f$ or reference $y^*$ | (outside the lab) |
| $p_t$ | Student next-token distribution | `student_logits.log_softmax(-1)` |
| $q_t$ | Teacher next-token distribution on the **same** prefix | `teacher_logits.detach().log_softmax(-1)` |
| $\operatorname{sg}$ | Stop-gradient: treat as a constant | `.detach()` / `torch.no_grad()` |
| $\pi_{old}$ | Student policy that generated the rollout | `old_token_logs` |
| $A_i$ | Verifier-derived group advantage of sequence $i$ | `sequence_advantages` |
| $\lambda$ | RLSD mixing weight | `mixing=0.5` |
| $\epsilon_w$ | RLSD evidence-weight clip | `weight_clip=0.2` |
| $\epsilon$ | PPO ratio clip | `policy_clip=0.2` |
| $|y|$ | Number of valid response tokens | `mask.sum(-1)` |

A teacher pass scores the **same student prefix**. Inference uses the student
context. In the lab, logits are supplied tensors; teacher prompting, autoregressive
collection, verification, and teacher synchronization are outside the example.

All losses average over valid response tokens per sequence, then sequences.
Prompt/padding positions are excluded; EOS remains a response token (a convention
for real data; the lab has no EOS token, only a mask). Computed rollouts and
prefixes are fixed during each update: no differentiation through sampling or
through teacher targets. `[B,T,V]` logits become `[B,T]` token losses.

## 3. Three ways to compare two distributions

KL divergence is **not symmetric**, so the order matters:

$$
D_{KL}(p\|q)=\sum_v p_v\log\frac{p_v}{q_v}\quad\text{(reverse: student first)},\qquad
D_{KL}(q\|p)=\sum_v q_v\log\frac{q_v}{p_v}\quad\text{(forward: teacher first)}.
$$

The Jensen-Shannon divergence (JSD) averages both against the mixture $m=(p+q)/2$:

$$
\boxed{\mathrm{JSD}(p,q)=\tfrac12 D_{KL}(q\|m)+\tfrac12 D_{KL}(p\|m)}.
$$

Worked example with the tiny $p$ and $q$:

| Token $v$ | $p_v$ | $q_v$ | $p_v\ln(p_v/q_v)$ | $q_v\ln(q_v/p_v)$ | $m_v$ |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.25 | 0.40 | $0.25\ln0.625=-0.1175$ | $0.40\ln1.6=0.1880$ | 0.325 |
| 1 | 0.25 | 0.30 | $0.25\ln0.833=-0.0456$ | $0.30\ln1.2=0.0547$ | 0.275 |
| 2 | 0.25 | 0.20 | $0.25\ln1.25=0.0558$ | $0.20\ln0.8=-0.0446$ | 0.225 |
| 3 | 0.25 | 0.10 | $0.25\ln2.5=0.2291$ | $0.10\ln0.4=-0.0916$ | 0.175 |
| **Sum** | | | **reverse KL ≈ 0.1218** | **forward KL ≈ 0.1064** | |

For JSD: $D_{KL}(q\|m)\approx0.0296$ and $D_{KL}(p\|m)\approx0.0261$, so
$\mathrm{JSD}\approx\tfrac12(0.0296+0.0261)\approx0.0279$. JSD is always between
0 and $\ln2\approx0.693$; both KLs are 0 exactly when $p=q$. These three values
match what `distillation_loss` returns for this single token with kinds
`'sdpo'`, `'opsd'`, and `'jsd'`.

## 4. SDPO: Self-Distillation Policy Optimization

Use feedback $z=f$ from the environment, such as a failing test, or successful
same-question attempts. The foundational reverse-KL objective is

$$
\boxed{L_{SDPO}=\mathbb E_{y\sim\pi_{old}}\frac1{|y|}\sum_t D_{KL}(p_t\|q_t)}.
$$

Read it right to left: at every response position, measure how far the student's
distribution is from the feedback-informed teacher's, average over the response's
tokens, then average over sampled responses.

For a fixed teacher and prefix, the gradient with respect to student logit $u_v$ is

$$
\frac{\partial L}{\partial u_v}=p_v\Big[\log\frac{p_v}{q_v}-D_{KL}(p\|q)\Big].
$$

Worked example (with $D_{KL}(p\|q)=0.1218$):

| $v$ | $\log(p_v/q_v)$ | $p_v[\log(p_v/q_v)-0.1218]$ | Effect of a descent step |
| --- | --- | --- | --- |
| 0 | -0.4700 | $0.25(-0.5918)=-0.1479$ | raise logit (teacher likes it more) |
| 1 | -0.1823 | $0.25(-0.3041)=-0.0760$ | raise logit |
| 2 | 0.2231 | $0.25(0.1014)=0.0253$ | lower logit |
| 3 | 0.9163 | $0.25(0.7945)=0.1986$ | lower logit (teacher likes it least) |

These match the autograd gradient of `distillation_loss(..., kind='sdpo')`.

**Sampled-token view.** The sampled-token policy-gradient view uses detached
advantage $\log q_t(y_t)-\log p_t(y_t)$; a constant baseline cancels in
expectation. In the example, sampling token 0 gives "advantage"
$\ln(0.40/0.25)=+0.470$ (the teacher approves more than the student), while
token 3 gives $\ln(0.10/0.25)=-0.916$. This connects SDPO to the policy
gradients of earlier lessons: the teacher's log-ratio plays the role of a dense,
per-token reward.

The lab evaluates **full-vocabulary** reverse KL, not this sampled estimate.
Practical variants can use JSD, teacher regularization, and importance correction
for reused rollouts.
[Paper v1, equation 1](https://arxiv.org/html/2601.20802v1).

## 5. OPSD: On-Policy Self-Distillation

Here $z=y^*$ is a reference answer or reasoning trace. The student samples;
the privileged self-teacher evaluates those prefixes. Minimize

$$
\boxed{L_{OPSD}=\mathbb E_{y\sim\pi_{old}}\frac1{|y|}\sum_t D(q_t\|p_t)}.
$$

"On-policy" means the prefixes come from the student's own samples, so the
student is corrected on the situations it actually produces, not on prefixes of
the reference answer.

The paper allows divergence choices. The lab uses forward KL, whose student-logit
gradient is simply

$$
\frac{\partial D_{KL}(q\|p)}{\partial u_v}=p_v-q_v,
$$

and also offers symmetric JSD with $m=(p+q)/2$:
$\tfrac12 KL(q\|m)+\tfrac12 KL(p\|m)$.
OPSD does not require a scalar verifier reward in its distillation loss.
The teacher is detached even if its parameters are shared with the student.
[Paper v1, section 3.2](https://arxiv.org/html/2601.18734v1).

Gradient comparison for the tiny example (computed by autograd on the script's
`distillation_loss`):

| Logit | Reverse KL (`sdpo`) | Forward KL (`opsd`), $p_v-q_v$ | JSD (`jsd`) |
| --- | --- | --- | --- |
| $u_0$ | -0.1479 | $0.25-0.40=-0.15$ | -0.0361 |
| $u_1$ | -0.0760 | $0.25-0.30=-0.05$ | -0.0152 |
| $u_2$ | 0.0253 | $0.25-0.20=0.05$ | 0.0099 |
| $u_3$ | 0.1986 | $0.25-0.10=0.15$ | 0.0413 |

All three push in the same directions here; they differ in magnitude, and they
can differ more when one distribution puts near-zero probability where the
other does not. This example does not show that any one divergence is better.

**Why must the teacher be detached?** If gradients reached the teacher, the
loss could be lowered by moving the *teacher* toward the student instead of the
student toward the teacher. Since the teacher may share parameters with the
student, that would also corrupt the student. `teacher_logits.detach()` makes
$q$ a fixed target.

## 6. Averaging correctly over masked tokens

`masked_mean` first averages each sequence's valid tokens, then averages
sequences. Using the script's mask `[[1,1,1],[1,1,0]]` and example token losses:

```text
sequence 0 token losses: [0.3, 0.1, 0.2]      mask [1, 1, 1]
sequence 1 token losses: [0.4, 0.6, 9.0]      mask [1, 1, 0]   (9.0 is padding)
```

1. Sequence 0: $(0.3+0.1+0.2)/3=0.2$.
2. Sequence 1: $(0.4+0.6)/2=0.5$. The padded 9.0 is multiplied by 0.
3. Mean over sequences: $(0.2+0.5)/2=0.35$.

A flat average over all five valid tokens would give $1.6/5=0.32$ instead:
that convention weights long responses more. Neither is universally correct;
the script chooses per-sequence averaging. A sequence with no valid token raises
`ValueError` rather than dividing by zero.

## 7. RLSD: RLVR with Self-Distillation

RLSD keeps the verifier (as in [GRPO](04_grpo.md)) and uses the teacher only to
decide **how much** each token shares in the sequence's credit. Use
verifier-derived group advantage $A_i$, retaining its sign while the teacher
modulates token credit:

$$\Delta_{it}=\operatorname{sg}(\log q_{it}(y_{it})-\log\pi_{old}(y_{it}\mid h_{it})),$$
$$w_{it}=\exp(\operatorname{sign}(A_i)\Delta_{it}),\qquad
\boxed{\tilde A_{it}=\operatorname{sg}\{A_i[(1-\lambda)+\lambda\,
\operatorname{clip}(w_{it},1-\epsilon_w,1+\epsilon_w)]\}}.$$

| Part | Interpretation |
| --- | --- |
| $\Delta_{it}$ | Teacher-vs-student log evidence for the sampled token |
| $\exp(\Delta)$ | Evidence ratio $q(y_t)/\pi_{old}(y_t)$ |
| $\operatorname{sign}(A_i)$ | For a failed sequence, invert the ratio |
| $\operatorname{clip}(w,0.8,1.2)$ | Teacher can change a token's credit by at most ±20% |
| $\lambda$ | Blend between uniform credit ($\lambda=0$) and teacher-weighted credit |

Insert $\tilde A$ into the PPO/GRPO clipped policy surrogate using a separate
current/old importance ratio. This follows Algorithm 1's credit construction;
the lab supplies the differentiable policy surrogate explicitly. According to the
cited paper version, teacher weights remain fixed across reuse epochs and no
auxiliary KL distillation loss is needed (the lab has no reuse epochs, so it
does not exercise the first point). $\lambda=0$ recovers uniform sequence credit.
Zero group advantage remains zero.
[Paper v1, equations 13–15 and Algorithm 1](https://arxiv.org/html/2604.03128v1).

### Hand calculation

Suppose $A=-1$, teacher probability 0.4, old student probability 0.2,
$\epsilon_w=0.2$, and $\lambda=0.5$.

1. $\Delta=\ln0.4-\ln0.2=\ln2$, so the evidence ratio is 2.
2. Negative advantage inverts it: $w=\exp(-\ln2)=0.5$.
3. Clip to $[0.8,1.2]$: 0.8.
4. $\tilde A=-1[0.5+0.5(0.8)]=-0.9$.

The token is still penalized, with a smaller magnitude: the teacher thinks this
token was reasonable, so it receives less blame for the failure. For $A=1$,
$w=2$ clips to 1.2 and gives $\tilde A=1[0.5+0.5(1.2)]=1.1$: a teacher-endorsed
token in a successful answer gets extra credit.

| $A$ | $q(y_t)$ | $\pi_{old}(y_t)$ | Ratio | $w$ | Clipped | $\tilde A$ ($\lambda=0.5$) |
| --- | --- | --- | --- | --- | --- | --- |
| +1 | 0.40 | 0.20 | 2.0 | 2.0 | 1.2 | 1.1 |
| -1 | 0.40 | 0.20 | 2.0 | 0.5 | 0.8 | -0.9 |
| +1 | 0.10 | 0.25 | 0.4 | 0.4 | 0.8 | 0.9 |
| -1 | 0.10 | 0.25 | 0.4 | 2.5 | 1.2 | -1.1 |
| +1 | 0.26 | 0.25 | 1.04 | 1.04 | 1.04 | 1.02 |
| 0 | 0.40 | 0.20 | 2.0 | 1 | 1 | 0 |

Every row has been checked against `rlsd_advantages`. Note that the sign of
$\tilde A$ always equals the sign of $A$: because the clipped weight is at least
0.8 > 0, the teacher can only rescale credit, never flip it. With the first
row's numbers, $\lambda=0$ gives $\tilde A=-1$ (uniform) and $\lambda=1$ gives
$-0.8$ (fully teacher-weighted). In code, `sign(0)=0`, so $w=e^0=1$ and
$\tilde A=0\times1=0$.

The script also clamps the signed gap to $[-30,30]$ before `exp`; this only
prevents overflow and does not change any result after the $[0.8,1.2]$ clip.

## 8. Two ratios with two different jobs

The teacher/student ratio measures evidence; the current/old ratio handles
policy reuse. They have different purposes.

| Ratio | Formula | Role | Gradient? |
| --- | --- | --- | --- |
| Evidence ratio | $q(y_t)/\pi_{old}(y_t)$ | Scales token credit inside $\tilde A$ | No, detached |
| Importance ratio | $\pi_\theta(y_t)/\pi_{old}(y_t)$ | PPO-style clipped update | Yes, through $\pi_\theta$ |

`rlsd_loss` computes `ratio = (new_token_logs-old_token_logs.detach()).exp()` and
the clipped surrogate $\min(\rho\tilde A,\operatorname{clip}(\rho,0.8,1.2)\tilde A)$,
then returns its negative masked mean.

## 9. Reproduce the script's RLSD number

The script samples tokens from a uniform old student (all zero logits, so
$\pi_{old}=0.25$ for every token) and scores them with random teacher logits.
Sequence 0 gets $A=+1$, sequence 1 gets $A=-1$, and the mask is
`[[1,1,1],[1,1,0]]`. With seed 7:

| Sequence, position | $q(y_t)$ | Evidence ratio $q/0.25$ | $w$ after sign | Clipped | $\tilde A$ |
| --- | --- | --- | --- | --- | --- |
| 0, 0 | 0.0539 | 0.215 | 0.215 | 0.8 | 0.9 |
| 0, 1 | 0.4274 | 1.709 | 1.709 | 1.2 | 1.1 |
| 0, 2 | 0.7033 | 2.813 | 2.813 | 1.2 | 1.1 |
| 1, 0 | 0.5063 | 2.025 | 0.494 | 0.8 | -0.9 |
| 1, 1 | 0.3131 | 1.252 | 0.798 | 0.8 | -0.9 |
| 1, 2 (padding) | 0.3179 | — | — | — | masked out |

Before any update the current policy equals the old one, so every $\rho=1$ and
the surrogate is just $\tilde A$:

1. Sequence 0: $(0.9+1.1+1.1)/3=1.0333$.
2. Sequence 1: $(-0.9-0.9)/2=-0.9$.
3. Mean: $(1.0333-0.9)/2=0.0667$; the loss is its negative, **-0.0667**.

This is exactly the printed `RLSD sampled-token loss: -0.06666666269302368`.

Gradient check for sequence 0, position 0 (sampled token 3, $\tilde A=0.9$):
each valid token of sequence 0 carries weight $\tfrac12\cdot\tfrac13=\tfrac16$, and
$\partial\log\pi(3)/\partial u_v=\mathbb 1[v=3]-0.25$. So the loss gradient is
$-\tfrac{0.9}{6}(0.75)=-0.1125$ for the sampled token and
$+\tfrac{0.9}{6}(0.25)=0.0375$ for the others, which matches autograd. The padded
position (sequence 1, position 2) receives a gradient of exactly zero.

## 10. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `masked_mean(values, mask)` | Per-sequence mean over valid tokens, then mean over sequences |
| `teacher = teacher_logits.detach().log_softmax(-1)` | $\log q_t$, a stop-gradient target |
| `(student.exp()*(student-teacher)).sum(-1)` | SDPO reverse KL $D_{KL}(p\|q)$ per token |
| `(teacher.exp()*(teacher-student)).sum(-1)` | OPSD forward KL $D_{KL}(q\|p)$ per token |
| `torch.logaddexp(student, teacher)-torch.log(torch.tensor(2.))` | $\log m$, mixture for JSD |
| `gap = teacher_token_logs-old_token_logs` | $\Delta_{it}$ (inside `torch.no_grad()`) |
| `sequence_advantages[:, None].sign()*gap` | $\operatorname{sign}(A_i)\Delta_{it}$ |
| `.clamp(-30, 30).exp().clamp(1-weight_clip, 1+weight_clip)` | $\operatorname{clip}(w_{it},0.8,1.2)$ with overflow guard |
| `sequence_advantages[:, None]*((1-mixing)+mixing*weights)` | $\tilde A_{it}$ |
| `ratio = (new_token_logs-old_token_logs.detach()).exp()` | Current/old importance ratio |
| `ratio.clamp(1-policy_clip, 1+policy_clip)*advantages` | PPO-clipped term |
| `teacher = torch.randn(2, 3, 4, requires_grad=True)` | Random teacher; `requires_grad` makes the "no teacher gradient" assertion meaningful |
| `optimizer = torch.optim.SGD([student], lr=0.5)` | 50 steps of plain gradient descent on student logits |

`distillation_loss` computes reverse KL, forward KL, or JSD. `rlsd_advantages`
constructs detached credit. `rlsd_loss` applies the current/old policy ratio.
`masked_mean` enforces valid tokens. Note that `rlsd_loss` does not expose
`weight_clip`, so through that function $\epsilon_w$ is always the default 0.2.

## 11. Run it and read the output

Run from the course folder:

```bash
python3 04_alignment/05_sdpo_rlsd_opsd.py
```

With the fixed seed the output is:

```text
sdpo loss: 0.29050520062446594 -> 0.02199612557888031
opsd loss: 0.24910327792167664 -> 0.02752235345542431
jsd loss: 0.06422306597232819 -> 0.03564245253801346
RLSD sampled-token loss: -0.06666666269302368
```

Each distillation line shows the loss before and after 50 SGD steps that move
fresh zero-initialized student logits toward the same random teacher. The
"before" values differ because they are different divergences of the same pair
of distributions; compare each line only with itself. The RLSD line is the
single loss value reproduced in section 9; no RLSD optimization steps are taken.

The assertions check that (1) each distillation loss decreased, (2) the teacher
tensor never received a gradient, and (3) the RLSD gradient on the student logits
is finite, with the teacher still gradient-free. They do not check convergence to
zero, the correctness of any particular number, or any downstream task benefit.
Random teacher logits illustrate the equations only; they do not demonstrate
improved reasoning.

## 12. What a real experiment would need

When designing an experiment, evaluate held-out prompts without privileged
information, correctness, answer leakage, response lengths, and performance on
prior tasks. A falling distillation loss alone does not establish better task
performance or justify general claims about the relative merits of these methods.

Answer leakage deserves special attention: the teacher sees $z$, and if the
student is pushed to copy tokens that are only predictable from $z$ (for example,
the reference answer's exact numbers), it may learn to guess rather than reason.
This lab cannot measure that risk.

## 13. Exercises

Exercise: verify a padded position has zero gradient. Set teacher equal to
student and check both KLs vanish. Compare positive and negative $A$ for the same
evidence ratio. Explain why detached teacher targets are necessary.

## 14. Practice calculations

1. Student $p=[0.5,0.5]$, teacher $q=[0.8,0.2]$. Reverse KL $D_{KL}(p\|q)$?
   **$0.5\ln(0.5/0.8)+0.5\ln(0.5/0.2)=0.5(-0.470)+0.5(0.916)\approx0.223$.**
2. Forward KL $D_{KL}(q\|p)$ for the same pair?
   **$0.8\ln1.6+0.2\ln0.4=0.376-0.183\approx0.193$.**
3. Forward-KL gradient for the two student logits? **$p-q=[-0.3,\,0.3]$.**
4. Token losses `[[0.2,0.4],[0.9,0.0]]` with mask `[[1,1],[1,0]]`. Masked mean?
   **$(0.3+0.9)/2=0.6$.**
5. RLSD with $A=+1$, $q(y_t)=0.15$, $\pi_{old}(y_t)=0.25$, $\lambda=0.5$, $\epsilon_w=0.2$.
   **Ratio 0.6, clipped to 0.8, $\tilde A=0.5+0.4=0.9$.**
6. Same token but $A=-1$? **$w=1/0.6\approx1.667$, clipped to 1.2,
   $\tilde A=-(0.5+0.6)=-1.1$: a token the teacher dislikes gets extra blame.**
7. $A=-2$, ratio 2, $\lambda=0.5$? **$w=0.5\to0.8$, $\tilde A=-2(0.9)=-1.8$.**
8. Can RLSD's teacher turn a negative advantage into a positive one?
   **No. The clipped weight lies in $[0.8,1.2]$, so $\tilde A$ keeps the sign of $A$.**

Next: [efficient training](../05_advanced/01_training_efficiency.md).
