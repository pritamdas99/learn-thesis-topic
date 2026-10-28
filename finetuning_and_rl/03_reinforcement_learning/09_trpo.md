# TRPO: constrain each policy update

Companion code: [09_trpo.py](09_trpo.py), a PyTorch equation lab on a three-action bandit.
Prerequisites: [policy gradients](05_reinforce_actor_critic.md),
[PPO and importance ratios](06_ppo_and_gae.md), and
[the math primer](../00_start/MATH_PRIMER.md) sections 2–6 (derivatives,
expectation, logarithms, KL divergence) and section 8 (linear systems and
curvature). You should know what a gradient, a Hessian, and a positive definite
matrix are. The lab computes exact action expectations, so it needs no sampling
and no critic.

## 1. What problem are we solving now?

A policy-gradient step says *which direction* improves the policy, but not
*how far* it is safe to go. A step that looks small in parameter space can
change action probabilities a lot, and a policy that suddenly collapses onto a
bad action collects bad data for the next update.

**Trust Region Policy Optimization (TRPO)** answers "how far?" by measuring the
step in terms of **how much the policy's action distribution changes**, using KL
divergence, and refusing steps that change it by more than a budget $\delta$.

The script uses the smallest possible setting: a **bandit** with one state and
three actions:

| Action | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Reward | 0 | 0.3 | 1 |

The policy is a softmax over three logits $\theta=(\theta_0,\theta_1,\theta_2)$,
starting at $\theta=0$, i.e. probabilities $(1/3,1/3,1/3)$. Because there is one
state and the rewards are known, every expectation can be computed exactly.
The goal is to watch TRPO move probability toward action 2 in 15 constrained steps.

## 2. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $\theta$ | Policy parameters (logits) | `theta`, shape `(3,)` |
| $\pi_\theta(a)$ | Action probability, $\operatorname{softmax}(\theta)_a$ | `x.log_softmax(-1)` |
| $\pi_{old}$ | Policy before the step, held fixed | `old_probs`, `old_logs` |
| $\hat A(a)$ | Advantage: reward minus expected reward under $\pi_{old}$ | `advantages` |
| $\rho_\theta(a)$ | Importance ratio $\pi_\theta(a)/\pi_{old}(a)$ | `ratios` |
| $L(\theta)$ | Surrogate objective | `surrogate(x)` |
| $D_{KL}$ | KL divergence $\pi_{old}\,\|\,\pi_\theta$ | `kl(x)` |
| $g$ | Gradient of $L$ at $\theta_{old}$ | `gradient` |
| $F$ | Hessian of KL at $\theta_{old}$ (Fisher matrix) | `fisher` |
| $\eta$ | Damping added to $F$ | `damping=1e-3` |
| $\delta$ | KL budget (trust-region size) | `delta=0.01` |
| $d$ | Proposed parameter step $\theta-\theta_{old}$ | `step` |

## 3. The constrained problem

TRPO maximises an old-policy surrogate while constraining average KL:

$$\boxed{\max_\theta\ \mathbb E_{s,a\sim\pi_{old}}
 [\rho_\theta(s,a)\hat A(s,a)]\quad\text{s.t.}\quad
\mathbb E_{s\sim d_{old}}D_{KL}(\pi_{old}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s))\le\delta}$$

with $\rho_\theta=\pi_\theta(a\mid s)/\pi_{old}(a\mid s)$. Read it piece by piece:

- The **surrogate** reweights old-policy experience by how much more (or less)
  likely the new policy makes each action. At $\theta=\theta_{old}$ every ratio
  is 1, so its gradient is the ordinary policy gradient.
- The **constraint** limits how different the new action distribution may be,
  averaged over states visited by the old policy.

In the bandit there is one state, so the "average KL" is just one KL, and the
expectations are exact sums over the three actions:

$$L(\theta)=\sum_a\pi_{old}(a)\,\frac{\pi_\theta(a)}{\pi_{old}(a)}\hat A(a),\qquad
D_{KL}=\sum_a\pi_{old}(a)\big[\log\pi_{old}(a)-\log\pi_\theta(a)\big].$$

These are exactly `surrogate` and `kl` in the code. Since the $\pi_{old}$ factors
cancel, $L(\theta)=\sum_a\pi_\theta(a)\hat A(a)$, which in this exact bandit
equals the expected reward of $\pi_\theta$ minus that of $\pi_{old}$.

## 4. Local approximation and the step formula

Let $d=\theta-\theta_{old}$. Near $\theta_{old}$:

- the surrogate is approximately linear: $L\approx g^Td$, with $g=\nabla L$;
- the KL is approximately quadratic: $D_{KL}\approx\tfrac12d^TFd$, with
  $F=\nabla^2 D_{KL}$.

Why no linear KL term? KL is minimised (equal to zero) when the policies match,
so its first derivative there is zero; the Hessian describes the local curvature.
This Hessian is the **Fisher information matrix** of the policy, which is why
the code names it `fisher`.

The local problem is

$$\max_d\ g^Td\quad\text{subject to}\quad\tfrac12d^TFd\le\delta.$$

The best direction is $d=\beta F^{-1}g$ (the "natural gradient"). Plugging it into
the constraint at equality, $\tfrac12\beta^2\,g^TF^{-1}g=\delta$, gives

$$\boxed{d=\sqrt{\frac{2\delta}{g^TF^{-1}g}}\;F^{-1}g}.$$

**Scalar check.** $g=0.2$, $F=0.5$, $\delta=0.01$: $F^{-1}g=0.4$,
$g^TF^{-1}g=0.2(0.4)=0.08$, scale $\sqrt{0.02/0.08}=0.5$, so $d=0.5(0.4)=0.2$.
Its quadratic KL is $\tfrac12(0.2)^2(0.5)=0.01=\delta$, exactly on the boundary.
The actual (nonlinear) KL can differ, which motivates the line search below.

## 5. Why damping? The softmax redundancy

Adding the same constant to every logit does not change a softmax. So moving
$\theta$ along $(1,1,1)$ changes nothing, the KL is flat in that direction, and
$F(1,1,1)^T=0$: **$F$ is singular**. The code therefore solves

$$(F+\eta I)\,x=g,\qquad \eta=10^{-3},$$

with a dense `torch.linalg.solve`. Damping makes the matrix invertible. Large
networks cannot form $F$ at all; they use conjugate gradient with
Hessian-vector products instead. The dense solve is appropriate only at toy scale.

The code then measures `curvature = direction@metric@direction`, which is
$x^T(F+\eta I)x=g^T(F+\eta I)^{-1}g$, the damped version of $g^TF^{-1}g$. If it is
essentially zero (gradient vanishes) the step is rejected.

## 6. Worked example: the first TRPO step

**Old policy and advantages.** At $\theta=0$, $\pi_{old}=(1/3,1/3,1/3)$. Expected
reward is $(0+0.3+1)/3=0.43333$, so

$$\hat A=(0-0.43333,\ 0.3-0.43333,\ 1-0.43333)=(-0.43333,\ -0.13333,\ 0.56667).$$

**Gradient.** For a softmax, $\partial L/\partial\theta_j=\pi_j(\hat A_j-\sum_a\pi_a\hat A_a)$.
The advantages already average to zero under $\pi_{old}$, so $g_j=\pi_j\hat A_j$:

$$g=(-0.14444,\ -0.04444,\ 0.18889).$$

**Fisher matrix.** For softmax, $F=\operatorname{diag}(\pi)-\pi\pi^T$:

$$F=\begin{pmatrix}0.22222&-0.11111&-0.11111\\-0.11111&0.22222&-0.11111\\
-0.11111&-0.11111&0.22222\end{pmatrix}.$$

Each row sums to zero, confirming the singular direction $(1,1,1)$.

**Solve.** Because $g$ sums to zero, $F$ acts on it like $\tfrac13I$, so
$(F+0.001I)x=g$ gives $x=g/0.334333$:

$$x=(-0.43204,\ -0.13293,\ 0.56497).$$

**Scale to the boundary.** $g^Tx=0.058519/0.334333=0.17503$, so the scale is
$\sqrt{2(0.01)/0.17503}=\sqrt{0.11427}=0.33803$ and

$$d=0.33803\,x=(-0.14604,\ -0.04494,\ 0.19098).$$

Its quadratic KL is $\tfrac12d^TFd\approx0.00997$ (slightly under 0.01 because of damping).

**Backtracking line search.** The code tries $\theta_{old}+0.5^j d$ for
$j=0,1,\dots,14$ and accepts the first candidate whose **actual** KL is at most
$\delta$ **and** whose surrogate improves on the old value (which is 0):

| $j$ | Step fraction | New probabilities | Actual KL | Surrogate | Accepted? |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | (0.2851, 0.3155, 0.3994) | 0.010153 | 0.060709 | No: KL > 0.01 |
| 1 | 0.5 | (0.3091, 0.3251, 0.3658) | 0.002517 | 0.030010 | **Yes** |

The quadratic model predicted 0.00997, but the real KL of the full step is
0.010153, just over the budget. Halving the step gives roughly a quarter of the
KL (0.0025), as a quadratic would suggest. Expected reward rises from 0.43333 to
0.46334, and $0.46334-0.43333=0.03001$ matches the surrogate, as Section 3 predicted
for this exact bandit.

If all 15 candidates fail, `trpo_step` returns the unchanged parameters with
`accepted=False`. Rejection is a valid, safe outcome.

## 7. Watch the whole run

The script runs `trpo_step` 15 times with the same rewards. We instrumented a
copy to log each accepted step:

| Step | Actual KL | $\pi(0)$ | $\pi(1)$ | $\pi(2)$ | Expected reward | $\lVert d\rVert$ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.002517 | 0.3091 | 0.3251 | 0.3658 | 0.4633 | 0.122 |
| 2 | 0.002510 | 0.2855 | 0.3156 | 0.3989 | 0.4936 | 0.121 |
| 3 | 0.002502 | 0.2625 | 0.3049 | 0.4326 | 0.5241 | 0.120 |
| 4 | 0.009975 | 0.2189 | 0.2805 | 0.5006 | 0.5848 | 0.238 |
| 8 | 0.009482 | 0.0827 | 0.1604 | 0.7569 | 0.8050 | 0.269 |
| 12 | 0.008692 | 0.0147 | 0.0503 | 0.9350 | 0.9501 | 0.430 |
| 15 | 0.007056 | 0.0009 | 0.0062 | 0.9930 | 0.9948 | 0.918 |

Things to notice:

- Every step was accepted. Steps 1–3 needed one halving; from step 4 on the full
  step already satisfied the KL budget.
- The KL per step stays at or below 0.01 throughout, as the assertion requires.
- The **Euclidean size** $\lVert d\rVert$ of the accepted step grows from about 0.12
  to 0.92. As the policy becomes peaked, the same change in logits moves the
  probabilities less, so TRPO allows larger parameter steps for the same KL.
  This is the point of measuring steps in distribution space rather than
  parameter space.
- The logits stay centred (their sum remains 0), since every step is a multiple
  of a zero-sum direction.

## 8. Caveats: what the lab does and does not show

- **Exact expectations.** Real TRPO estimates the surrogate and KL from sampled
  trajectories; this lab sums over all three actions exactly.
- **No trainer.** A complete TRPO implementation also needs environment
  collection, advantage estimation (often GAE), and critic fitting. None of that
  is here.
- **Dense Hessian.** `torch.autograd.functional.hessian` and `torch.linalg.solve`
  are fine for 3 parameters, not for a neural network.
- **No unconditional guarantee.** TRPO's theory motivates monotonic improvement
  for a related, more conservative objective. Practical sampled average-KL TRPO
  does not inherit an unconditional monotonic-return guarantee. See the
  [primary paper](https://arxiv.org/abs/1502.05477).
- **Relation to PPO.** [PPO](06_ppo_and_gae.md) replaces this constrained
  second-order step with a clipped first-order surrogate. Its clipping does not
  enforce a hard KL bound; TRPO's line search does check one (for the
  estimated KL).

## 9. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `trpo_step(theta, rewards, delta=0.01, damping=1e-3)` | One constrained update; returns `(theta, kl, accepted)` |
| `old_logs = theta.detach().log_softmax(-1)` | $\log\pi_{old}$, frozen |
| `advantages = (rewards-(old_probs*rewards).sum()).detach()` | $\hat A(a)=r(a)-\mathbb E_{\pi_{old}}[r]$ |
| `ratios = (x.log_softmax(-1)-old_logs).exp()` | $\rho_\theta(a)$ |
| `(old_probs*ratios*advantages).sum()` | Exact surrogate $L(\theta)$ |
| `(old_probs*(old_logs-x.log_softmax(-1))).sum()` | Exact $D_{KL}(\pi_{old}\|\pi_\theta)$ |
| `torch.autograd.functional.jacobian(surrogate, theta)` | $g$ |
| `torch.autograd.functional.hessian(kl, theta)` | $F$ |
| `fisher+damping*torch.eye(...)` | $F+\eta I$ |
| `torch.linalg.solve(metric, gradient)` | $x=(F+\eta I)^{-1}g$ |
| `curvature = direction@metric@direction` | $g^T(F+\eta I)^{-1}g$ |
| `(2*delta/curvature).sqrt()*direction` | Step scaled to the quadratic KL boundary |
| `for j in range(15)`, `(0.5**j)*step` | Backtracking line search |
| `candidate_kl <= delta and surrogate(candidate) > before` | Acceptance test: actual KL and improvement |
| `return theta.detach().clone(), 0., False` | Reject: parameters unchanged |

## 10. Run it

From the course folder:

```bash
python3 03_reinforcement_learning/09_trpo.py
```

Output on our machine:

```text
TRPO action probabilities: [0.0008788435235311761, 0.006167071159787495, 0.9929540853166813]
```

The computation is deterministic (no random numbers), so the digits should
repeat, up to tiny floating-point differences across PyTorch versions. The
assertions check that each step's reported KL is at most $0.01+10^{-10}$ and
that the final probability of the best action exceeds 0.8. They do not check
that every step was accepted or that the policy is optimal.

## 11. Practice calculations

1. Scalar case: $g=0.3$, $F=2$, $\delta=0.01$. What is the step $d$?
   **$F^{-1}g=0.15$, $g^TF^{-1}g=0.045$, scale $\sqrt{0.02/0.045}\approx0.667$,
   $d=0.1$.**
2. What is its quadratic KL? **$\tfrac12(0.1)^2(2)=0.01$.**
3. A two-action bandit with rewards $(0,1)$ starts at $\theta=0$. What are the
   advantages and the gradient $g$? **$\pi=(0.5,0.5)$, $\hat A=(-0.5,0.5)$,
   $g=(-0.25,0.25)$.**
4. If a full step has actual KL 0.05 and KL scales roughly quadratically with
   step size, how many halvings until it is under $\delta=0.01$? **Two: one halving
   gives about $0.05/4=0.0125$, two give about $0.05/16\approx0.0031$.**
5. Why is $F$ singular for a softmax policy? **Adding a constant to all logits
   leaves the probabilities unchanged, so $F(1,\dots,1)^T=0$.**
6. Why does $\lVert d\rVert$ grow in later steps? **A peaked softmax is less
   sensitive to logit changes, so a larger parameter step fits in the same KL.**

Exercise: increase $\delta$ and log accepted step sizes, actual KL, and reward.
Explain why the same Euclidean parameter step can change different policies by
different amounts.

This completes the reinforcement learning section. Next:
[Reward models: learning scores from pairwise preferences](../04_alignment/01_reward_model.md).
