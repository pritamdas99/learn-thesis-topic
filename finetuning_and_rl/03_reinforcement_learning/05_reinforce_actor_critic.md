# Policy gradients, REINFORCE, and a learned baseline

Companion code: [05_reinforce_actor_critic.py](05_reinforce_actor_critic.py). Requires PyTorch.
Prerequisites: [MDPs, returns, and values](02_mdp_value_iteration.md),
[Q-learning and SARSA](03_q_learning_sarsa.md),
[DQN](04_dqn.md), and
[tensors and autograd](../01_foundations/02_tensors_autograd.md).
Read this explanation before running the code.

## 1. What problem are we solving now?

Same corridor, same rewards, $\gamma=0.95$:

```text
       0          1          2          3          4
    [END] <--> [room] <--> [START] <--> [room] <--> [GOAL]
     reward 0                                     reward +1
```

Q-learning and DQN learn **values** and then act by choosing the action with
the largest value. A **policy-gradient** method instead learns the **action
probabilities** directly. It parameterizes a policy $\pi_\theta(a\mid s)$ and
adjusts $\theta$ so that actions which led to high return become more probable.

Why bother, when the value methods already work here?

- The policy is naturally stochastic, so exploration is built in.
- It extends directly to continuous actions (output a Gaussian mean, say),
  where $\max_a Q(s,a)$ would itself be an optimization problem.
- It is the family that PPO, and LLM fine-tuning with RL, belong to.

The script trains two small networks:

- an **actor** that outputs two logits per state, turned into a categorical
  distribution over left/right by softmax;
- a **critic** that outputs one number $V_\phi(s)$, a prediction of return used
  only as a **baseline**. The critic never chooses actions.

The goal of the run is a high probability of moving right in states 1–3.

## 2. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $\pi_\theta(a\mid s)$ | Probability of action $a$ in state $s$ | `Categorical(logits=actor(features[state]))` |
| $\theta$ | Actor parameters | `actor = nn.Linear(5, 2)` |
| $z_{left}, z_{right}$ | The two logits; $\pi(\text{right})=e^{z_R}/(e^{z_L}+e^{z_R})$ | `actor(features[state])` |
| $V_\phi(s)$ | Critic's estimate of return from $s$ | `critic(features[state]).squeeze(-1)` |
| $\phi$ | Critic parameters | `critic = nn.Linear(5, 1)` |
| $r_t$ | Reward after the action at step $t$ | `rewards[t]`, 1 only on entering 4 |
| $G_t$ | Discounted return from step $t$ to the end of the episode | `returns[t]` |
| $\hat A_t$ | Advantage estimate $G_t-V_\phi(s_t)$ | `advantage[t]` |
| $H(\pi_t)$ | Entropy of the action distribution at step $t$ | `distribution.entropy()` |
| $J(\theta)$ | Expected return of the policy: what we want to increase | (not computed directly) |

## 3. What does the actor look like at the start?

`actor = nn.Linear(5, 2)` on a one-hot input means the logits for state $s$ are
column $s$ of the weight matrix plus a bias vector that **every state shares**.
With `torch.manual_seed(7)`, the initial probabilities of moving right are

| State | 1 | 2 | 3 |
| --- | --- | --- | --- |
| Initial $\pi(\text{right}\mid s)$ | 0.434 | 0.426 | 0.370 |

So the agent begins slightly biased toward left. The initial critic values are
all negative (about $-0.64$, $-0.43$, $-0.51$ for states 1–3), although every
true return is between 0 and 1. Both networks have to learn.

## 4. The log-probability trick on a one-step problem

Start with a single decision, like a bandit. Let $R(a)$ be the reward of action
$a$. The expected reward is

$$J(\theta)=\sum_a\pi_\theta(a)R(a).$$

Its gradient is $\sum_a\nabla\pi_\theta(a)R(a)$. Using
$\nabla\pi=\pi\nabla\log\pi$ (just the chain rule for $\log$):

$$\boxed{\nabla J(\theta)=\sum_a\pi_\theta(a)\,\nabla\log\pi_\theta(a)\,R(a)
=\mathbb E_{a\sim\pi}\big[\nabla\log\pi_\theta(a)\,R(a)\big]}$$

It is an **expectation under the policy**, so one sampled action gives an
unbiased estimate $\nabla\log\pi_\theta(a)R(a)$ without differentiating through
the environment. A high-reward sampled action becomes more probable; a poor one
less probable.

For a softmax policy the score has a simple form:
$\partial\log\pi(a)/\partial z_k=\mathbb 1[k=a]-\pi(k)$.

**Worked example.** $\pi(\text{right})=0.4$, $R(\text{right})=1$,
$R(\text{left})=0$. Look at the derivative with respect to $z_R$:

| Sampled action | Probability | Score $\partial\log\pi(a)/\partial z_R$ | Score $\times R(a)$ |
| --- | --- | --- | --- |
| Right | 0.4 | $1-0.4=0.6$ | $0.6$ |
| Left | 0.6 | $0-0.4=-0.4$ | $0$ |

Average: $0.4(0.6)+0.6(0)=0.24$. Directly:
$J=\pi_R$, and $\partial\pi_R/\partial z_R=\pi_R(1-\pi_R)=0.4(0.6)=0.24$. They agree.

## 5. From one step to whole episodes

An episode's probability is a product of action probabilities (times
environment probabilities that do not depend on $\theta$), so its log is a sum
of $\log\pi_\theta(a_t\mid s_t)$ terms. An action at time $t$ cannot cause
rewards that arrived before it, so its score term can be weighted by the
**future** return only:

$$\boxed{G_t=r_t+\gamma r_{t+1}+\gamma^2 r_{t+2}+\cdots=r_t+\gamma G_{t+1}}$$

`discounted_returns` computes this backwards. For rewards `[0, 0, 1]`:

| Step (backwards) | Calculation | $G_t$ |
| --- | --- | --- |
| $t=2$ | $1+0.95(0)$ | 1 |
| $t=1$ | $0+0.95(1)$ | 0.95 |
| $t=0$ | $0+0.95(0.95)$ | 0.9025 |

Returns: `[0.9025, 0.95, 1]`. An episode ending at state 0 has all rewards zero
and all returns zero. An episode is collected **completely** before any update,
which is what makes this a **Monte Carlo** method.

Estimator convention: for the exact discounted start-state objective, the
expected sum includes an outer $\gamma^t$ as well as discounting inside $G_t$.
Like many practical small implementations, this script instead averages
per-step score terms with discounted returns and no explicit outer $\gamma^t$.
That is its estimator convention, not a claim of exact equivalence to every
choice of discounted objective.

## 6. Why subtract a baseline?

A return of 1 is impressive in a state normally worth 0.1, but disappointing in
a state normally worth 5. Without a reference point, every positive return
pushes its action up, and only the *size* of the push tells good from great.
Subtract a state-dependent baseline:

$$\boxed{\hat A_t=G_t-V_\phi(s_t)}$$

For a baseline that does not depend on the sampled action:

$$\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)\,V(s)]
=V(s)\sum_a\nabla\pi(a\mid s)=V(s)\nabla\sum_a\pi(a\mid s)=V(s)\nabla 1=0.$$

So subtracting it does not change the expected gradient at a fixed state, while
it can greatly reduce variance.

**Worked example.** Repeat section 4 with baseline $b=0.4$ (the true expected
reward):

| Sampled action | Probability | Score w.r.t. $z_R$ | $R-b$ | Product |
| --- | --- | --- | --- | --- |
| Right | 0.4 | 0.6 | $0.6$ | 0.36 |
| Left | 0.6 | $-0.4$ | $-0.4$ | 0.16 |

| | Mean | $E[x^2]$ | Variance |
| --- | --- | --- | --- |
| No baseline (values 0.6, 0) | 0.24 | $0.4(0.36)=0.144$ | $0.144-0.0576=0.0864$ |
| Baseline 0.4 (values 0.36, 0.16) | $0.144+0.096=0.24$ | $0.4(0.1296)+0.6(0.0256)=0.0672$ | $0.0672-0.0576=0.0096$ |

Same mean, nine times smaller variance. Notice the left action now also pushes
$z_R$ **up**: left was worse than average, so the update makes it less likely.

The critic is trained to approximate returns so that it can play the role of $b$.
It does not need to be accurate for the expected gradient to stay unbiased; an
accurate critic just reduces variance more.

## 7. Losses and gradient flow

The script minimizes, over the $T$ steps of one episode:

$$\boxed{L_{actor}=-\frac1T\sum_t\log\pi_\theta(a_t\mid s_t)\,\operatorname{stopgrad}(\hat A_t)}$$

$$L_{critic}=\frac1T\sum_t(G_t-V_\phi(s_t))^2,\qquad
\boxed{L=L_{actor}+0.5L_{critic}-0.01\frac1T\sum_t H(\pi_t)}$$

| Term | Purpose |
| --- | --- |
| Minus sign in $L_{actor}$ | Optimizers minimize; minimizing $-\log\pi\cdot\hat A$ raises $\log\pi$ when $\hat A>0$ |
| `advantage.detach()` | Stop-gradient: the actor loss must not change the critic |
| $0.5L_{critic}$ | Squared-error regression of $V_\phi$ toward observed returns |
| $-0.01H$ | Entropy bonus: rewards keeping some randomness, encouraging exploration |

Without `detach()`, the actor loss would also send gradients into the critic
through $\hat A$, encouraging it to manipulate the baseline rather than just
predict returns. Too large an entropy weight keeps the policy too random.

One-term example: advantage 0.6 and selected-action probability 0.25 give
$-\log(0.25)(0.6)\approx0.8318$. Raising that probability to 0.5 reduces the
term to $-\log(0.5)(0.6)\approx0.4159$. For negative advantage the incentive
reverses: the loss falls when the probability falls.

## 8. Trace one update by hand

To see every number, take the initial networks from section 3 and **suppose**
the first episode were $2\to3\to4$ (right, right). This is a chosen example,
not the episode the seeded script actually samples first.

**Collected data.**

| $t$ | $s_t$ | $\pi(\text{right}\mid s_t)$ | $\log\pi(a_t\mid s_t)$ | $H(\pi_t)$ | $V_\phi(s_t)$ | $r_t$ |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2 | 0.4260 | $-0.8534$ | 0.6821 | $-0.4310$ | 0 |
| 1 | 3 | 0.3699 | $-0.9946$ | 0.6589 | $-0.5077$ | 1 |

**Returns and advantages.** $G=[0.95,\,1]$, so
$\hat A_0=0.95-(-0.4310)=1.3810$ and $\hat A_1=1-(-0.5077)=1.5077$. Both are
positive: reaching the goal was much better than the (pessimistic) critic predicted.

**Losses.**

$$L_{actor}=-\tfrac12[(-0.8534)(1.3810)+(-0.9946)(1.5077)]=\tfrac12(1.1785+1.4995)\approx1.3390$$

$$L_{critic}=\tfrac12(1.3810^2+1.5077^2)=\tfrac12(1.9071+2.2733)\approx2.0902,\qquad
\bar H=\tfrac12(0.6821+0.6589)\approx0.6705$$

$$L=1.3390+0.5(2.0902)-0.01(0.6705)\approx2.3774.$$

**Actor gradient.** Using $\partial\log\pi/\partial z_R=1-\pi_R$ for a right
action, the gradient of $L_{actor}$ with respect to state 2's right logit is
$-\tfrac12(1.3810)(1-0.4260)\approx-0.3964$, and for state 3 it is
$-\tfrac12(1.5077)(1-0.3699)\approx-0.4750$. Left logits get the opposite signs.
Gradient descent therefore **raises** the right logits. The shared right-logit
bias receives the sum, $-0.8714$. Weight columns for states 0, 1, 4 get zero
gradient because their one-hot inputs were zero.

**Critic gradient.** $\partial(0.5L_{critic})/\partial V_t=-\tfrac12\hat A_t$:
$-0.6905$ and $-0.7539$, so descent raises both values toward the returns.

**One Adam step (lr 0.03).** On its first step Adam moves each parameter with a
nonzero gradient by almost exactly $0.03$ against the gradient's sign. The
right-minus-left logit gap for states 2 and 3 grows by $4\times0.03=0.12$
(weight and bias, on both logits); for state 1 only the shared bias moves, so
the gap grows by 0.06. Running this in PyTorch gives:

| State | $\pi(\text{right})$ before | after one step |
| --- | --- | --- |
| 1 (not visited) | 0.434 | 0.449 |
| 2 | 0.426 | 0.456 |
| 3 | 0.370 | 0.398 |

State 1 changed even though it was never visited, because of the shared bias.
That is generalization through shared parameters, the same effect as in DQN.

## 9. What kind of algorithm is this?

This is **Monte Carlo REINFORCE with a learned baseline**, not TD-based A2C:
the advantage uses complete observed returns $G_t$, and the critic is never
bootstrapped from $V(s_{t+1})$. Bootstrapped advantages appear in
[A2C/A3C](08_a2c_a3c.md) and [PPO and GAE](06_ppo_and_gae.md).

The episode loop runs at most 30 steps. Here that horizon is treated as the
endpoint of the finite collection task: there is no bootstrap at that endpoint,
and the return of a 30-step episode that never reaches a terminal state is just
its (zero) observed rewards. Since the actor and critic observe only position
and not remaining time, this finite-horizon setup does not represent a fully
time-aware optimal value function. It is a simplifying choice in this toy
demonstration. For a continuing-task time limit, use bootstrap as taught in
[PPO and GAE](06_ppo_and_gae.md).

Each episode is used for exactly **one** gradient step and then discarded.
Unlike DQN there is no replay: the score-function estimator requires actions
sampled from the current policy, so old episodes would need importance
correction (one motivation for PPO's probability ratio).

## 10. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `discounted_returns(rewards, gamma=0.95)` | Backward recursion $G_t=r_t+\gamma G_{t+1}$ |
| `actor, critic = nn.Linear(5, 2), nn.Linear(5, 1)` | Logits for 2 actions; one value per state |
| `torch.optim.Adam(list(actor.parameters())+list(critic.parameters()), lr=0.03)` | One optimizer for both networks |
| `features = torch.eye(5)` | One-hot state encodings |
| `for episode in range(250):` | 250 episodes, one update each |
| `for _ in range(30):` | At most 30 steps per episode |
| `distribution = Categorical(logits=actor(features[state]))` | Softmax policy $\pi_\theta(\cdot\mid s)$ |
| `action = distribution.sample()` | Sample, do not take the argmax |
| `logs.append(distribution.log_prob(action))` | Store $\log\pi_\theta(a_t\mid s_t)$ with its graph |
| `entropies.append(distribution.entropy())` | Store $H(\pi_t)$ |
| `values.append(critic(features[state]).squeeze(-1))` | Store $V_\phi(s_t)$ as a scalar |
| `state += 1 if action.item() else -1` | Move right for action 1, left for 0 |
| `rewards.append(float(state == 4))` | Reward 1 only on entering the goal |
| `if state in (0, 4): break` | Terminal states end the episode |
| `advantage = returns-values` | $\hat A_t=G_t-V_\phi(s_t)$ |
| `actor_loss = -(torch.stack(logs)*advantage.detach()).mean()` | $L_{actor}$ with stop-gradient |
| `critic_loss = advantage.square().mean()` | $L_{critic}$ |
| `loss = actor_loss + 0.5*critic_loss - 0.01*entropy` | Total loss $L$ |
| `actor(features[1:4]).softmax(-1)[:, 1].tolist()` | Final $\pi(\text{right})$ for states 1–3 |

## 11. Run and experiment

Run from the course folder:

```bash
python3 03_reinforcement_learning/05_reinforce_actor_critic.py
```

With the supplied seed the output is

```text
probability of right: [0.9134044647216797, 0.9554551839828491, 0.9237444400787354]
```

These are $\pi(\text{right}\mid s)$ for states 1, 2, 3, up from 0.434, 0.426,
0.370 at initialization. The policy remains stochastic (about 4–9% left), partly
because the entropy bonus resists full determinism and partly because training
stops after 250 episodes. The script contains **no assertion**; it prints the
probabilities for inspection. Separately, `tests/test_learning_invariants.py`
checks `discounted_returns([0., 0., 1.], 0.5)` equals `[0.25, 0.5, 1.]`.

Experiments:

- Replace `advantage.detach()` in the actor loss with `returns` to obtain a
  no-baseline REINFORCE comparison.
- Compare variability across several seeds, not one final probability.
- Change the entropy weight 0.01 to 0 and to 0.1 and compare final probabilities.
- Explain why subtracting a baseline that depends arbitrarily on the action
  would invalidate the zero-expectation argument in section 6.

## 12. Practice calculations

1. Rewards `[0, 0, 0, 1]`, $\gamma=0.95$. Returns?
   **`[0.857375, 0.9025, 0.95, 1]`.**
2. $G_t=0.9025$ and $V(s_t)=0.5$. Advantage? **$0.4025$.**
3. Selected-action probability 0.8, advantage $-0.5$. Actor loss term
   $-\log\pi\cdot\hat A$? **$-(-0.2231)(-0.5)\approx-0.1116$; minimizing it
   pushes the probability down.**
4. Softmax with $\pi(\text{right})=0.3$, right sampled, advantage 2. Gradient of
   $-\log\pi\cdot\hat A$ with respect to $z_R$? **$-(1-0.3)(2)=-1.4$; descent raises $z_R$.**
5. In section 4's example, use the action-dependent "baseline"
   $b(\text{right})=1$, $b(\text{left})=0$. Expected gradient?
   **$0.4(0.6)(0)+0.6(-0.4)(0)=0\neq0.24$: biased.**
6. Why is there no replay buffer here? **REINFORCE needs samples from the current
   policy; old episodes would need importance weights.**

Next: [PPO and GAE: from rewards to a stable policy update](06_ppo_and_gae.md).
