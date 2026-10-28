# PPO and GAE: from rewards to a stable policy update

Companion code: [06_ppo_and_gae.py](06_ppo_and_gae.py). Read this explanation before running the code.

## 1. The problem we are solving

Imagine an agent in a five-position corridor, numbered 0 through 4. It starts
at position 2 and chooses left or right. Position 0 ends the episode with zero
reward; entering position 4 ends it with reward 1. Other transitions reward 0.
We want the agent to learn to move right without supplying the correct action
as a supervised label.

The policy produces probabilities, not a command that is always correct.
Early on it may choose right only half the time. Learning should increase the
probability of actions that led to better-than-expected outcomes. However,
large changes based on a few lucky episodes can destroy useful behavior.

Two components address different parts of this problem:

- **GAE, generalized advantage estimation:** estimate how much better an action
  was than the policy's usual expected outcome at that state.
- **PPO, proximal policy optimization:** improve action probabilities using those
  estimates while reducing the incentive for excessively large updates.

GAE is an estimator, not an optimizer. PPO can use other advantage estimators;
GAE can be used with other policy-gradient algorithms.

Prerequisites: [returns and policy gradients](05_reinforce_actor_critic.md),
[MDPs and values](02_mdp_value_iteration.md), and
[softmax and derivatives](../01_foundations/01_python_and_math.md).

## 2. Notation: read the symbols before the equations

| Symbol | Meaning | Code |
| --- | --- | --- |
| $s_t$ | State before taking action at step $t$ | `state` |
| $a_t$ | Sampled action, 0=left or 1=right | `action` |
| $r_t$ | Reward from that action | `rewards[t]` |
| $\pi_\theta(a\mid s)$ | Action probability under actor parameters $\theta$ | `Categorical(logits=actor(...))` |
| $V_\phi(s)$ | Critic's estimate of expected future return | `critic(...)` |
| $\gamma$ | Discount factor, here 0.95 | `gamma` |
| $\lambda$ | GAE trace-decay setting, here 0.95 | `lam` |
| $\hat A_t$ | Estimated advantage | `advantages[t]` |
| $\rho_t$ | New action probability divided by old probability | `ratio[t]` |
| $\epsilon$ | PPO clip width, here 0.2 | bounds `0.8, 1.2` |
| $d_t$ | 1 if transition truly terminates the task | `terminated[t]` |
| $e_t$ | 1 if the collected episode ends, including truncation | `episode_end[t]` |

$\mathbb E$ means an average weighted by probabilities. A hat, as in $\hat A$,
means an estimate. $\nabla_\theta$ means derivatives with respect to all entries
of the actor's parameter vector.

## 3. Rewards, returns, values, and advantages

A reward describes one transition. A return includes future rewards:

$$
G_t = r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.
$$

If rewards are `[0, 0, 1]` and $\gamma=0.9$, the three returns are
`[0.81, 0.9, 1.0]`. Earlier actions receive credit because they made the later
reward possible. Discounting also prefers a sooner reward to an equally large
later reward.

The state value is an average over possible continuations:

$$
V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s].
$$

The action value fixes the first action before averaging the continuation:

$$
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
$$

Their difference is the advantage:

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

If the state normally yields return 0.4, but taking a particular action yields
expected return 0.7, its advantage is 0.3. Positive advantage means better than
the state baseline; negative advantage means worse. It does not mean the
immediate reward itself must be positive or negative.

## 4. Why a policy gradient can improve behavior

The basic objective is expected return, $J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G_0]$,
where $\tau$ is a trajectory. For a probability $p_\theta$:

$$
\nabla_\theta p_\theta
=p_\theta\nabla_\theta\log p_\theta.
$$

Substituting this identity into the derivative of an expectation gives the
score-function estimator. The probability of a trajectory contains a product
of action probabilities; taking its logarithm turns this product into a sum.
For a discounted episodic objective, one exact reward-to-go form is:

$$
\nabla_\theta J
=\mathbb E\left[\sum_t\gamma^t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t\right].
$$

In practical PPO, rollout transitions are typically averaged uniformly, using
discounted advantage estimates, rather than inserting explicit $\gamma^t$
factors. This is the convention used in our code and its surrogate objective.

For a fixed state, subtracting an action-independent baseline does not change
the expected score-function term:

$$
\sum_a\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)b(s)
=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0.
$$

That is the mathematical reason a value baseline can reduce noise without
introducing baseline bias. Approximate bootstrapped advantage estimates can
still introduce bias; that is a separate issue.

A sample with positive advantage rewards increasing its log-probability. A
negative-advantage sample rewards decreasing it. The advantages are held fixed
while differentiating the actor loss, so the actor cannot improve that loss by
merely changing the critic's estimate.

## 5. First build a one-step advantage estimate

The one-step temporal-difference residual is:

$$
\delta_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t).
$$

It compares an updated estimate (reward plus discounted next value) with the
old estimate at the current state. If the value function is accurate, averaging
this residual conditional on $(s_t,a_t)$ gives the action's advantage.

Example: $r_t=0$, $V(s_t)=0.4$, $V(s_{t+1})=0.7$, $\gamma=0.9$, and no termination:

$$
\delta_t=0+0.9(0.7)-0.4=0.23.
$$

Only one observed transition is used, so this can be relatively low-variance,
but it depends strongly on the accuracy of the critic's next-state estimate.

Code:

```python
deltas = rewards + gamma * (1 - terminated) * next_values - values
```

## 6. Deriving GAE by combining several horizons

An $n$-step estimate uses more observed rewards before bootstrapping:

$$
\hat A_t^{(n)}=
\sum_{l=0}^{n-1}\gamma^l r_{t+l}
+\gamma^nV(s_{t+n})-V(s_t).
$$

Away from terminal boundaries, expand the TD residuals and cancel intermediate
value terms to obtain:

$$
\hat A_t^{(n)}=\sum_{l=0}^{n-1}\gamma^l\delta_{t+l}.
$$

GAE mixes these estimates with geometrically decreasing weights. In the
infinite-horizon expression for $0\leq\lambda<1$:

$$
\hat A_t^{\mathrm{GAE}}
=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}\hat A_t^{(n)}
=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}.
$$

Why does the second equality hold? The residual $\delta_{t+l}$ occurs in every
$n$-step estimate with $n\ge l+1$. Its total coefficient is:

$$
\gamma^l(1-\lambda)\sum_{n=l+1}^{\infty}\lambda^{n-1}
=\gamma^l\lambda^l.
$$

For a finite rollout, compute the available residual sum backwards:

$$
\hat A_t=\delta_t+\gamma\lambda(1-e_t)\hat A_{t+1}.
$$

This recurrence is the `carry` variable in the code. Initialize the carry to
zero at the end of the collected buffer. If that boundary is not terminal,
the final residual still contains the next-state bootstrap value.

- $\lambda=0$: use only the one-step TD residual.
- $\lambda$ closer to 1: use longer observed reward sequences, usually reducing
  reliance on intermediate value predictions while increasing sampling noise.
- $\lambda=1$ on a complete terminal episode: telescoping gives $G_t-V(s_t)$.
  On a truncated rollout, it gives a bootstrapped return minus the value instead.

These are bias/variance tendencies, not guarantees for every environment.

## 7. A full numerical GAE example

Use a three-step terminal episode. To keep the arithmetic easy, take
$\gamma=0.9$, $\lambda=0.8$ (the script uses 0.95 for both).

| $t$ | Reward | $V(s_t)$ | $V(s_{t+1})$ | Terminal? | $\delta_t$ |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.4 | 0.5 | No | $0+0.9(0.5)-0.4=0.05$ |
| 1 | 0 | 0.5 | 0.6 | No | $0+0.9(0.6)-0.5=0.04$ |
| 2 | 1 | 0.6 | ignored | Yes | $1-0.6=0.4$ |

Since $\gamma\lambda=0.72$, work backwards:

$$
\hat A_2=0.4,
\qquad \hat A_1=0.04+0.72(0.4)=0.328,
$$

$$
\hat A_0=0.05+0.72(0.328)=0.28616.
$$

The critic regression targets are:

$$
\hat R_t=\hat A_t+V_{\mathrm{old}}(s_t)
=[0.68616,\ 0.828,\ 1.0].
$$

These are GAE-derived lambda-return targets, not necessarily the raw Monte
Carlo returns `[0.81, 0.9, 1.0]`. `gae()` returns both arrays before the script
normalizes the advantages for the actor.

## 8. Why the code needs two episode masks

**Termination:** the task actually ends. Future task value is zero.

**Truncation:** collection stops at a time limit even though the underlying
continuing task could proceed. We usually bootstrap from the final observation.
The next stored transition may belong to a newly reset episode, so we must not
propagate its GAE residual backwards into the old one.

Thus:

| Transition | Bootstrap next value? | Continue GAE into next stored transition? |
| --- | --- | --- |
| Ordinary step | Yes | Yes, if available in buffer |
| True terminal | No | No |
| Time-limit reset | Yes | No |

Example: a truncated step has reward 1, current value 0, next value 2, and
$\gamma=0.5$. Its residual is $1+0.5(2)=2$. Do not add a following reset episode's
reward of 100 to that estimate. Use the observation from **before resetting**
for `next_values`.

If a finite deadline is genuinely part of the task definition, its endpoint
is terminal instead. Remaining time may also need to be part of the state.
See [Gymnasium's explanation of time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## 9. Why PPO stores old log-probabilities

Collect experience using a fixed rollout policy $\pi_{\mathrm{old}}$. After one
optimization step, the actor has changed, but we would like to reuse this batch
for a few more steps.

For a fixed state, importance weighting gives:

$$
\sum_a\pi_\theta(a\mid s)f(a)
=\sum_a\pi_{\mathrm{old}}(a\mid s)
\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}f(a),
$$

assuming the old distribution has support wherever the new one needs it.
Define the sampled-action ratio:

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}
=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)).
$$

That last expression is exactly:

```python
ratio = (new_logs - old_logs).exp()
```

The simple surrogate is $\mathbb E_{\mathrm{old}}[\rho_t\hat A_t]$.
It reweights sampled actions, but does not fully correct the changing state
visitation distribution. Consequently, it is a local approximation to policy
improvement, not an exact new-policy return evaluator.

At the beginning of an update the policies agree, so ratios are 1. They change
as optimization proceeds. `old_logs` must remain fixed throughout those updates.

## 10. Deriving the clipped objective's behavior

The PPO-Clip objective to **maximize** is:

$$
L^{\mathrm{clip}}(\theta)
=\frac1N\sum_t\min\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right).
$$

The minimum chooses the more pessimistic of the ordinary and clipped surrogate.
The sign of advantage determines which direction should stop receiving extra
incentive.

Take $\epsilon=0.2$. The clip interval is `[0.8, 1.2]`:

| Advantage | Ratio | Ordinary term | Clipped term | Minimum |
| --- | --- | --- | --- | --- |
| +2 | 1.1 | 2.2 | 2.2 | 2.2 |
| +2 | 1.5 | 3.0 | 2.4 | 2.4 |
| +2 | 0.5 | 1.0 | 1.6 | 1.0 |
| -2 | 0.5 | -1.0 | -1.6 | -1.6 |
| -2 | 1.5 | -3.0 | -2.4 | -3.0 |

For a good action ($\hat A>0$), increasing its probability helps until the ratio
exceeds 1.2; beyond that, this sample's clipped surrogate stops giving additional
benefit. Decreasing a good action's probability remains penalized.

For a bad action ($\hat A<0$), decreasing its probability helps until the ratio
falls below 0.8. Increasing a bad action's probability remains penalized.

For example, if the old action probability was 0.4 and the new probability is
0.6, the ratio is 1.5. It is the **ratio**, not the probability, that is clipped.
A ratio bound of 1.2 would correspond here to probability 0.48.

PyTorch minimizes losses, so the script negates this maximization objective:

```python
policy_loss = -torch.minimum(unclipped, clipped).mean()
```

Clipping does not force all ratios to remain in the interval. Shared parameters,
other samples, and optimizer momentum can still move them outside. PPO does not
guarantee monotonic return improvement or a strict KL bound. The clipped
surrogate is the central mechanism described in the [PPO paper](https://arxiv.org/abs/1707.06347).

## 11. Train the critic and maintain exploration

The critic learns the fixed targets computed from the rollout:

$$
L_V(\phi)=\frac1N\sum_t(V_\phi(s_t)-\hat R_t)^2.
$$

The entropy of a discrete policy is:

$$
H(\pi(\cdot\mid s))=-\sum_a\pi(a\mid s)\log\pi(a\mid s).
$$

For two equally likely actions, entropy is $\log 2\approx0.693$ nats. For a
nearly deterministic policy it is near zero. An entropy bonus discourages
premature collapse to one action.

The script minimizes:

$$
L_{\mathrm{total}}=-L^{\mathrm{clip}}+0.5L_V-0.01\overline H.
$$

The signs matter: decrease policy loss and value error, but increase entropy.
The coefficients are teaching choices, not universal optimal settings.

The actor advantages are standardized:

$$
\tilde A_t=\frac{\hat A_t-\operatorname{mean}(\hat A)}
{\operatorname{std}(\hat A)+10^{-8}}.
$$

This controls their scale within an update. It is a practical heuristic, not
required by the definition of GAE; finite-batch centering may change individual
advantage signs. The **critic targets must use the unnormalized advantages**,
which the code computes before this step.

## 12. Map the entire script to the math

| Code section | Mathematical role |
| --- | --- |
| `features = torch.eye(5)` | One-hot state encoding; each state has its own input coordinate |
| `actor = nn.Linear(5, 2)` | Two logits per state; the categorical distribution applies softmax |
| `critic = nn.Linear(5, 1)` | One scalar value prediction per state |
| `with torch.no_grad()` collection | Store fixed old-policy statistics without a backward graph |
| 64 collection steps | One rollout batch, potentially containing several episodes |
| `next_values` before reset | Bootstrap from the actual next state |
| `gae(...)` | TD residuals, backward advantage recursion, critic targets |
| Four update epochs | Reuse this batch briefly, always comparing with original old logs |
| `clip_grad_norm_(..., 0.5)` | Separate safeguard on gradient magnitude; not PPO ratio clipping |
| Next outer iteration | Collect fresh transitions using the newly updated actor |

The example uses one full rollout batch per update epoch, separate linear
actor/critic networks, and one shared optimizer. Larger implementations often
use shuffled minibatches, separate learning rates, and additional diagnostics.
This code has no KL early stopping or clipped value loss.

## 13. What to expect and what can go wrong

Run from the course directory:

```bash
python 03_reinforcement_learning/06_ppo_and_gae.py
```

The final probability of moving right should become high for the nonterminal
states under the supplied seed. Intermediate “successful terminal transitions”
are counts within 64 collection steps, not an evaluation success percentage.
Shorter successful episodes can increase that count.

Common mistakes include recomputing `old_logs` during updates, letting gradients
flow into advantage targets, failing to reset after truncation, using the reset
observation for bootstrap, normalizing critic targets accidentally, or treating
PPO clipping as ordinary gradient clipping.

Useful extra diagnostics: average evaluation return, entropy, fraction of
ratios outside the clip interval, critic explained variance, and approximate
KL. One sample-based KL diagnostic is the average of
$\rho-1-\log\rho$ over old-policy samples; its population expectation equals
$D_{KL}(\pi_{old}\|\pi_{new})$ at fixed states under matching support. It is
still only a finite-sample estimate, not a hard constraint.

## 14. Exercises with checks

1. Set $\lambda=0$ in the numerical example. Expected advantages:
   `[0.05, 0.04, 0.4]`.
2. Set $\lambda=1$. Expected advantages: `[0.41, 0.4, 0.4]`; adding the old
   values gives the Monte Carlo returns `[0.81, 0.9, 1.0]`.
3. Use advantage -3 and ratio 0.6 with clip width 0.2. The selected surrogate
   is `min(-1.8, -2.4) = -2.4`; the minimized policy loss contribution is +2.4.
4. Explain why a positive reward can still produce negative advantage.
   Check: the outcome can be worse than an even higher expected baseline.
5. Compare one, four, and twenty update epochs across several seeds. Measure
   policy change and held-out returns rather than assuming more updates help.

For the estimator's original motivation see the [GAE paper](https://arxiv.org/abs/1506.02438).
For an implementation-oriented reference see [Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
The corridor arithmetic and code walkthrough above are worked specifically for
this learning folder.
