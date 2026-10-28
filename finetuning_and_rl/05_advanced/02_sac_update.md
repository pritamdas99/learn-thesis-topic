# Soft Actor-Critic: continuous actions, entropy, and twin value estimates

Companion code: [02_sac_update.py](02_sac_update.py).
Prerequisites: [Q-learning targets](../03_reinforcement_learning/03_q_learning_sarsa.md),
[DQN targets and target networks](../03_reinforcement_learning/04_dqn.md),
[policy gradients and actor-critic](../03_reinforcement_learning/05_reinforce_actor_critic.md),
and [autograd](../01_foundations/02_tensors_autograd.md).
Read this explanation before running the code.

## 1. What problem are we solving now?

The corridor in earlier lessons had two actions: left or right. Many control
problems instead have **continuous** actions: a steering angle, a torque, or
a joint velocity. Here each action is a vector of two numbers, each in $(-1,1)$.

Two consequences follow:

- DQN's `max` over actions is no longer a lookup over a short list. We cannot
  enumerate infinitely many actions, so a separate **actor** network proposes them.
- Exploring by occasionally choosing a uniformly random action (epsilon-greedy)
  is clumsy. SAC instead uses a **stochastic** policy and adds a reward for
  keeping it random, which encourages exploration and avoids choosing one
  narrow action too early.

SAC learns a stochastic continuous-action policy using **off-policy replay**:
it reuses stored transitions collected by older versions of the policy.

**What this file actually does.** It executes one full set of SAC update
equations on 32 **synthetic** replay transitions: random states with 3 numbers,
random actions with 2 numbers, random rewards. It does not collect real
transitions, run an environment, or demonstrate a learned control policy.
It is an equation lab: you can check every tensor shape and gradient path.

## 2. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $s$, $s'$ | State and next state, 3 numbers each | `states`, `next_states`, shape `[32,3]` |
| $a$ | Stored replay action, 2 numbers in $[-1,1)$ | `actions`, shape `[32,2]` |
| $r$ | Reward | `rewards`, shape `[32,1]` |
| $d$ | 1 if the transition terminated the task | `terminated` (all zeros here) |
| $\gamma$ | Discount | `0.99` |
| $\pi_\theta(a\mid s)$ | Actor's action density | `actor.sample(...)` |
| $\mu,\sigma$ | Gaussian mean and standard deviation before squashing | `mean`, `log_std.clamp(-5, 2).exp()` |
| $u$ | Unsquashed Gaussian sample | `raw` |
| $Q_{\phi_1},Q_{\phi_2}$ | Two online critics | `q1`, `q2` |
| $Q_{\bar\phi_1},Q_{\bar\phi_2}$ | Slow target critics | `target1`, `target2` |
| $\alpha$ | Entropy temperature, $\alpha=e^{\ell_\alpha}$ | `log_alpha.exp()`, starts at $e^{-1}\approx0.3679$ |
| $H_{target}$ | Target entropy | `target_entropy = -2.0` |
| $\tau$ | Polyak averaging rate | `0.005` |

Alpha here controls the reward/exploration balance. It is unrelated to LoRA's
adapter scale and to the learning rate $\alpha$ of the Q-learning lesson.

## 3. The maximum-entropy objective

Conceptually SAC optimizes discounted reward **plus** entropy:

$$
\boxed{J(\pi)=\mathbb E\left[\sum_t\gamma^t
\{r_t+\alpha H(\pi(\cdot\mid s_t))\}\right]}
$$

Entropy $H$ measures how spread out the action distribution is. Because
entropy is $-\mathbb E[\log\pi(a\mid s)]$, sampled objectives use
$-\alpha\log\pi(a\mid s)$ as a per-sample bonus: an unlikely (low $\log\pi$)
action gets a larger bonus.

In continuous spaces this is **differential** entropy, which can be negative
and changes under action rescaling. A density can exceed 1, so $\log\pi$ can be
positive and the "entropy bonus" can be a penalty. You will see this in the
worked example of section 5.

## 4. A differentiable stochastic actor

The actor's single linear layer maps the 3 state numbers to 4 outputs: two
action means and two log standard deviations. It then samples:

$$
\boxed{u=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon,
\quad\epsilon\sim\mathcal N(0,I),\qquad a=\tanh(u).}
$$

This is **reparameterization**: randomness enters through independent noise
$\epsilon$, while the sampled value remains differentiable with respect to the
actor's parameters. `Normal.rsample()` supplies that path. Ordinary `.sample()`
would not preserve this pathwise gradient, and the actor update in section 7
would receive no signal from the critic.

The code clamps the log standard deviation to $[-5,2]$, so $\sigma$ stays
between $e^{-5}\approx0.0067$ and $e^{2}\approx7.39$.

## 5. Why the tanh squash needs a log-probability correction

Tanh bounds each action to $(-1,1)$. But the Gaussian density is a density over
$u$, not over the transformed $a$. Squashing piles probability up near $\pm1$,
so the density of $a$ is different. Change of variables gives, per dimension $j$:

$$
\boxed{\log\pi(a\mid s)=\sum_j\left[
\log\mathcal N(u_j;\mu_j,\sigma_j)-\log(1-\tanh^2u_j)\right]}
$$

because $\frac{d}{du}\tanh u=1-\tanh^2u$. The script evaluates the correction
with the stable identity

$$
\boxed{\log(1-\tanh^2u)=2[\log2-u-\operatorname{softplus}(-2u)]}
$$

which follows from $1-\tanh^2u=\frac{4e^{-2u}}{(1+e^{-2u})^2}$ and
$\operatorname{softplus}(x)=\log(1+e^x)$.

### Worked example: one two-dimensional sample

Suppose for some state the actor outputs $\mu=(0.2,-0.4)$ and
$\log\sigma=(-0.5,0)$, so $\sigma=(0.6065,1)$, and the noise is
$\epsilon=(0.5,-1.0)$.

Gaussian log-density per dimension is
$\log\mathcal N=-\tfrac12\epsilon^2-\log\sigma-\tfrac12\log(2\pi)$,
with $\tfrac12\log(2\pi)=0.9189$.

| Step | Dimension 1 | Dimension 2 |
| --- | --- | --- |
| $u=\mu+\sigma\epsilon$ | $0.2+0.6065(0.5)=0.5033$ | $-0.4+1(-1)=-1.4$ |
| $a=\tanh u$ | 0.4647 | -0.8854 |
| $\log\mathcal N(u)$ | $-0.125+0.5-0.9189=-0.5439$ | $-0.5-0-0.9189=-1.4189$ |
| $\log(1-\tanh^2u)$ | -0.2433 | -1.5318 |
| corrected $\log\mathcal N-\log(1-\tanh^2u)$ | -0.3007 | +0.1128 |

Sum over dimensions: $\log\pi(a\mid s)=-0.3007+0.1128=-0.1879$.
Without the correction you would have used $-1.9629$, which is wrong by 1.775.
Note dimension 2's corrected log-density is **positive**: near $a=-0.885$ the
squashed density is above 1. That is the negative-differential-entropy effect
from section 3.

### Why the stable identity matters

For $u=10$ in float32, $\tanh(10)$ rounds to exactly 1, so the naive
$\log(1-\tanh^2u)=\log0=-\infty$. The identity gives
$2[0.6931-10-\operatorname{softplus}(-20)]=-18.6137$, a finite correct value.
It avoids subtracting nearly equal floating-point numbers when tanh saturates.

If an environment requires actions outside $[-1,1]$, affine rescaling
and the corresponding density correction need to be handled explicitly.

## 6. Critic target: bootstrap value, entropy, and the twin minimum

SAC uses two critics and slow target copies of each. For a replay transition,
sample a **fresh** next action $a'$ from the current actor at $s'$ (not the
action stored in replay), then compute:

$$
\boxed{y=r+\gamma(1-d)\left[
\min_{i=1,2}Q_{\bar\phi_i}(s',a')-\alpha\log\pi_\theta(a'\mid s')
\right]}
$$

Each critic minimizes $\mathbb E[(Q_{\phi_i}(s,a)-y)^2]$ with a **detached**
target (the script builds $y$ under `torch.no_grad()`). The minimum of two
estimates can reduce the optimistic value errors that a single critic's
maximization tends to accumulate, the same overestimation issue seen with DQN.

### Worked example (earlier toy numbers)

Target critics predict 3 and 5, $\log\pi(a'\mid s')=-1$, $\alpha=0.2$,
$r=1$, $\gamma=0.9$, nonterminal:

$$y=1+0.9[\min(3,5)-0.2(-1)]=1+0.9(3.2)=3.88.$$

For a terminal transition ($d=1$) the bracket is multiplied by zero and $y=1$.

### Same example with the script's settings

Now use $\gamma=0.99$ and the initial $\alpha=e^{-1}=0.36788$:

1. Twin minimum: $\min(3,5)=3$.
2. Entropy term: $-\alpha\log\pi=-0.36788(-1)=+0.36788$.
3. Soft next value: $3+0.36788=3.36788$.
4. Target: $y=1+0.99(3.36788)=4.3342$.

| Variant of the next-value estimate | Target $y$ |
| --- | --- |
| Twin minimum (what SAC uses) | 4.3342 |
| Average of the two critics, 4 | 5.3242 |
| Larger critic, 5 | 6.3142 |
| Twin minimum but $\alpha=0$ (no entropy term) | 3.9700 |

The min choice gives the most cautious target; the entropy term adds value to
next states where the policy can remain random.

### An implementation limitation you must notice

In the script `q2` is a deep copy of `q1`, and both critics receive the same
data, target, and symmetric optimizer updates. They remain **identical** in
this toy update, so `torch.minimum(q1(...), q2(...))` always equals `q1(...)`
and the printed critic loss is exactly twice one critic's MSE
($1.18791/2\approx0.594$ each). The code shows the twin-critic formula, but does
not actually demonstrate its error-diversification benefit. A practical
extension should initialize critics independently and evaluate whether their
errors differ.

## 7. Actor update: increase soft value through the sampled action

The actor minimizes:

$$
\boxed{L_\pi=\mathbb E_s\left[\alpha\log\pi_\theta(a\mid s)
-\min_iQ_{\phi_i}(s,a)\right],\quad a=f_\theta(s,\epsilon).}
$$

Read it as: prefer actions the critics value highly, but pay $\alpha$ for every
unit of $\log\pi$ (i.e. reward staying random). For one sample with
$\alpha=0.36788$, $\log\pi=-1$, and critic values 3 and 5, the contribution is
$0.36788(-1)-3=-3.3679$. The script's printed actor loss is the average of such
terms over 32 states.

Gradients reach $\theta$ along two paths: through $\log\pi_\theta$, and through
the critic's output with respect to the sampled action $a=f_\theta(s,\epsilon)$.

Freeze critic **parameters**, but retain the derivative of the critic output
with respect to the sampled action. Otherwise the actor cannot learn which
actions increase value. The script's `requires_grad_(False)` does this; wrapping
the entire actor-loss critic computation in `no_grad()` would remove the useful
path too. The actor loss also uses `log_alpha.exp().detach()`, so this step
does not change alpha.

## 8. Temperature update: aim for a target entropy

Alpha must stay positive, so the script learns $\ell_\alpha=\log\alpha$ and
uses $\alpha=e^{\ell_\alpha}$. Its common log-temperature surrogate is:

$$
\boxed{L_\alpha=-\mathbb E\left[\ell_\alpha\,
\operatorname{stopgrad}(\log\pi(a\mid s)+H_{target})\right]}
$$

Here $H_{target}=-2$ for two action dimensions (the common "negative action
dimensionality" heuristic). The derivative is

$$\frac{\partial L_\alpha}{\partial\ell_\alpha}=-\mathbb E[\log\pi+H_{target}]
=\hat H-H_{target},$$

where $\hat H=-\mathbb E[\log\pi]$ is the sampled entropy estimate. Gradient
descent therefore lowers alpha when entropy is above target and raises it when
entropy is below target.

### Worked example: two cases, starting from $\ell_\alpha=-1$

| Mean $\log\pi$ | Entropy estimate $\hat H$ | Gradient $\hat H-(-2)$ | Direction | $\ell_\alpha$ after one Adam step (lr 0.001) | New $\alpha$ |
| --- | --- | --- | --- | --- | --- |
| -1 | 1 | +3 | entropy above target: lower $\alpha$ | -1.001 | 0.36751 |
| 3 | -3 | -1 | entropy below target: raise $\alpha$ | -0.999 | 0.36825 |

Why exactly $\pm0.001$? On Adam's **first** step, bias correction makes the
update approximately $\text{lr}\times\operatorname{sign}(g)$, regardless of how
large $g$ is (a gradient of 300 also moves $\ell_\alpha$ by 0.001). This is an
Adam property, not a SAC property.

The log-alpha surrogate has a different gradient scaling from optimizing the
exact alpha-parameterized dual objective, but the same desired balance point.
In the script the alpha update uses the `log_prob` sampled **before** the actor
step, detached.

## 9. Target networks: Polyak averaging

Targets move slowly by Polyak averaging:

$$
\boxed{\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi,\qquad\tau=0.005.}
$$

If a target weight is 2.0 and the online weight is 3.0, the new target weight
is $0.995(2)+0.005(3)=2.005$. `target_p.lerp_(p, 0.005)` computes exactly this.
Repeated updates close the gap geometrically: half of the gap to an unchanging online weight disappears
after $\ln0.5/\ln0.995\approx138$ updates. This slow movement keeps the critic's
regression target from chasing itself, as with DQN's target network.

## 10. Code sequence and shapes

The code updates critics first, actor second, temperature third, then target
networks. States have shape `[32,3]`, actions `[32,2]`, and concatenated critic
inputs `[32,5]`. Rewards, log-probabilities, and values are `[32,1]` (note
`keepdim=True` in the log-prob sum) to avoid accidental broadcasting: a `[32]`
tensor minus a `[32,1]` tensor would silently become `[32,32]`.

## 11. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `self.net = nn.Linear(3, 4)` | State to two means and two log standard deviations |
| `mean, log_std = self.net(states).chunk(2, dim=-1)` | Split the 4 outputs into $\mu$ and $\log\sigma$ |
| `Normal(mean, log_std.clamp(-5, 2).exp())` | Gaussian with $\sigma\in[e^{-5},e^2]$ |
| `raw = distribution.rsample()` | Reparameterized sample $u=\mu+\sigma\epsilon$ |
| `actions = raw.tanh()` | Squash to $(-1,1)$ |
| `correction = 2*(torch.log(torch.tensor(2.0))-raw-nn.functional.softplus(-2*raw))` | Stable $\log(1-\tanh^2u)$ |
| `log_prob = (distribution.log_prob(raw)-correction).sum(-1, keepdim=True)` | Corrected $\log\pi(a\mid s)$, shape `[32,1]` |
| `q2 = copy.deepcopy(q1)` | Second critic, **identical** initialization |
| `target1, target2 = copy.deepcopy(q1).requires_grad_(False), ...` | Frozen target critics |
| `log_alpha = nn.Parameter(torch.tensor(-1.0))` | $\ell_\alpha$, so $\alpha$ starts at 0.3679 |
| `next_actions, next_logs = actor.sample(next_states)` | Fresh $a'$ and $\log\pi(a'\mid s')$ under `no_grad` |
| `next_q = torch.minimum(target1(inputs), target2(inputs))-log_alpha.exp()*next_logs` | Soft next value with twin minimum |
| `target = rewards + 0.99*(1-terminated)*next_q` | Bellman target $y$ |
| `critic_loss = (q1(inputs)-target).square().mean()+(q2(inputs)-target).square().mean()` | Sum of both critics' MSE |
| `p.requires_grad_(False)` | Freeze critic parameters, keep action gradients |
| `actor_loss = (log_alpha.exp().detach()*log_prob-torch.minimum(q1(inputs), q2(inputs))).mean()` | $L_\pi$ |
| `alpha_loss = -(log_alpha*(log_prob.detach()+target_entropy)).mean()` | $L_\alpha$ |
| `target_p.lerp_(p, 0.005)` | Polyak update with $\tau=0.005$ |

## 12. Run it

Run from the course folder:

```bash
python3 05_advanced/02_sac_update.py
```

On the CPU used to verify this lesson (PyTorch 2.14, seed 7) the output was:

```text
critic loss: 1.1879132986068726 actor loss: -0.4474215507507324
```

Exact digits can differ across PyTorch versions or hardware. There is no
assertion. Printed losses only confirm execution of a synthetic update; they
are not learning curves, and one number cannot show whether SAC "works".
The script needs no GPU.

Extend the example with independently initialized critics, a real environment,
replay collection, a warmup policy, and held-out episode evaluation before
claiming control learning.

## 13. Practice calculations

1. Target critics give 2 and 2.5, $\log\pi(a'\mid s')=-0.5$, $\alpha=0.2$,
   $r=0$, $\gamma=0.99$, nonterminal. Target?
   **$0.99[2-0.2(-0.5)]=0.99(2.1)=2.079$.**
2. Same transition but terminal with $r=0$? **$y=0$.**
3. One dimension with $u=0$: what is $\log(1-\tanh^2 0)$, and from the identity?
   **$\log1=0$; $2[\log2-0-\log2]=0$.**
4. Batch mean $\log\pi=-2.5$ with $H_{target}=-2$. Sign of
   $\partial L_\alpha/\partial\ell_\alpha$, and does alpha go up or down?
   **$\hat H-H_{target}=2.5+2=+4.5>0$, so descent lowers alpha.**
5. Target weight 1.0, online weight 0.0, $\tau=0.005$. After one update?
   **0.995.** After two (online still 0)? **$0.995^2=0.990025$.**
6. Why can't the actor loss be computed inside `torch.no_grad()`?
   **The gradient through $Q(s,a)$ with respect to the sampled action would vanish,
   leaving only the entropy term.**

Reference: [SAC equations and algorithm](https://spinningup.openai.com/en/latest/algorithms/sac.html).
Next: [offline RL and the conservative Q-learning penalty](03_offline_rl_cql.md).
