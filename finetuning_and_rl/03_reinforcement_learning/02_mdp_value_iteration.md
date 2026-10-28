# MDPs, value iteration, and policy iteration: planning a route to reward

Companion code: [02_mdp_value_iteration.py](02_mdp_value_iteration.py).
Prerequisites: the [bandits chapter](01_bandits.md), probability-weighted averages,
and basic Python loops. This chapter explains each new symbol before using it.

## 1. What problem are we solving?

Imagine a robot standing in a corridor:

```text
State:       0          1          2          3          4
          [END] <--> [room] <--> [START] <--> [room] <--> [GOAL]
Arrival
reward:      0          0          0          0         +1
```

The robot starts at state 2. It can move left or right. Entering either edge,
0 or 4, ends the episode. Only entering state 4 gives reward 1.
An **episode** is one run from a starting state until the task ends.

We want to answer two questions:

1. How valuable is each starting room, considering rewards we can obtain later?
2. Which action should we choose in each room?

In a bandit, pressing a button affects the current reward. Here an action also
changes our location and therefore our future opportunities. Moving right from
2 gives no immediate reward, but puts us one step away from the goal.

**The solution is a policy: choose right in states 1, 2, and 3.** Value iteration
works out why these actions are best by calculating their future consequences.

## 2. The vocabulary of an MDP

An MDP, or **Markov decision process**, specifies a sequential decision problem.

| Term | Symbol | Meaning in our corridor |
| --- | --- | --- |
| State | $s$ | Current position: 0, 1, 2, 3, or 4 |
| Action | $a$ | Left (`0`) or right (`1`) |
| Next state | $s'$ | Position after the move; the prime means “next” |
| Reward | $r$ | 1 when entering 4; otherwise 0 |
| Transition model | $p(s',r\mid s,a)$ | Probabilities of next positions and rewards |
| Discount factor | $\gamma$ | How strongly future rewards count; code uses 0.95 |
| Policy | $\pi(a\mid s)$ | Rule giving action probabilities in each state |
| Terminal state | — | State 0 or 4; no further decisions or rewards |

“Markov” means the present state contains the information needed to predict the
next transition, given our action. For this corridor, knowing the current position
is enough; we need not remember how we got there.

The model is deterministic: from state 2, action right always produces
`(next_state=3, reward=0, terminated=False)`. From state 3, right produces
`(4, 1, True)`. Deterministic transitions are a special case of probabilities:
the stated outcome has probability 1 and all other outcomes have probability 0.

An MDP describes the problem. **Planning** uses a known model to solve it.
**Reinforcement learning** learns from experience, often without knowing that
model. This script performs planning: it already has `transition(state, action)`.
It does not need to explore the corridor and collect training episodes.

## 3. Reward and return are different

A reward is feedback for one move. A **return** adds rewards over future moves.
Let $r_t$ mean the reward immediately after the action at time $t$:

$$G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots.$$

$G_t$ is the return from time $t$. The discount $\gamma$ multiplies each reward
once for each extra step we wait. With $\gamma=0.95$:

| Future timing | Contribution of reward 1 to current return |
| --- | --- |
| On the current move | $1$ |
| One move later | $0.95$ |
| Two moves later | $0.95^2=0.9025$ |
| Three moves later | $0.95^3=0.857375$ |

This is not a success probability. The robot's moves are certain in this example;
0.95 is a chosen weight making delayed rewards count less.

From state 2, move right twice:

```text
2 --right / reward 0--> 3 --right / reward 1--> 4 (end)
```

Its return is $0+0.95(1)=0.95$. From state 1, moving right three times gives
$0+0.95(0)+0.95^2(1)=0.9025$. The first reward is not discounted, so three
moves to the goal mean two discount factors, not three.

Taking a detour from state 2, `2 -> 1 -> 2 -> 3 -> 4`, gives rewards
`[0, 0, 0, 1]` and return $0.95^3=0.857375$. Both routes eventually earn one
coin, but the shorter route has higher discounted return.

## 4. What is a state value?

A state value answers: **“Starting here, how much future return should I expect?”**

$$V^\pi(s)=E[G_t\mid S_t=s,\text{ follow policy }\pi].$$

The superscript $\pi$ means “under this policy”; it is not a power.
A policy that always goes left has value zero everywhere in this corridor.
A policy that always goes right has values:

| State | Rewards along the rightward route | Value |
| --- | --- | --- |
| 1 | `[0, 0, 1]` | 0.9025 |
| 2 | `[0, 1]` | 0.95 |
| 3 | `[1]` | 1 |
| 0 or 4 | Already finished; no future rewards | 0 |

**Why is $V(4)=0$ rather than 1?** The reward 1 was already received when
entering 4. Starting at 4 means the episode has ended. Counting another 1
would count the same goal reward twice.

$V^*(s)$ means the best achievable expected return from $s$, over all policies.
The star means “optimal.” The values above are also optimal for this corridor.

## 5. The Bellman idea: split the future into one move plus the rest

The return can be rearranged as:

$$G_t=r_t+\gamma G_{t+1}.$$

Consequently, a move's value is:

$$\text{immediate reward}+\gamma\times\text{value of the next state}.$$

Suppose our current estimates are $V(1)=0.9025$ and $V(3)=1$.
At state 2, compare:

| Candidate action | Next state | Immediate reward | Candidate value |
| --- | --- | --- | --- |
| Left | 1 | 0 | $0+0.95(0.9025)=0.857375$ |
| Right | 3 | 0 | $0+0.95(1)=0.95$ |

Choose right because 0.95 is larger. Notice both immediate rewards are zero;
it is the **future value** that distinguishes the actions.

For a known deterministic transition, value iteration uses:

$$V_{new}(s)=\max_a\left[r(s,a)+\gamma(1-d)V_{old}(s')\right].$$

Here $d=1$ for a terminal transition, and $d=0$ otherwise. Thus $(1-d)$ removes
future continuation after termination. `max` chooses the largest candidate value.
It returns a number; `argmax` would return the action achieving that number.

## 6. Follow value iteration sweep by sweep

We do not begin with the answer. Initialize all estimates to zero:

```python
values = [0.0] * 5
```

A **sweep** updates all nonterminal states. Every update within one sweep uses
the previous sweep's values. This is why the code creates `updated = values.copy()`.

| Sweep | $V(0)$ | $V(1)$ | $V(2)$ | $V(3)$ | $V(4)$ |
| --- | --- | --- | --- | --- | --- |
| Initial | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 1 | 0 |
| 2 | 0 | 0 | 0.95 | 1 | 0 |
| 3 | 0 | 0.9025 | 0.95 | 1 | 0 |
| 4 | 0 | 0.9025 | 0.95 | 1 | 0 |

**Sweep 1:** State 3 discovers the immediate reward: right has candidate 1.
State 2 still sees the *old* $V(3)=0$, so its new value remains zero.

**Sweep 2:** State 2 now sees the previous sweep's $V(3)=1$, so right gives
$0+0.95(1)=0.95$. State 1 still sees the previous $V(2)=0$.

**Sweep 3:** State 1 now sees $V(2)=0.95$, giving $0+0.95(0.95)=0.9025$.

**Sweep 4:** No value changes. The equations and the estimates agree.

```text
Knowledge of reward moves backward:
    sweep 1:                       state 3 knows
    sweep 2:            state 2 knows
    sweep 3: state 1 knows
```

The robot is not moving backward here. These are calculations over a model.

## 7. What exactly converges, and what is solved?

The entries of `values` approach $V^*$: optimal expected discounted returns.
A **fixed point** is a vector unchanged by another Bellman optimality update.
The loop stops when the largest absolute change is below `1e-10`:

```python
change = max(abs(a-b) for a, b in zip(values, updated))
```

This checks numerical stability of the values; it is not a count of successful
robot episodes. In this example the values stabilize exactly after a few sweeps.
More general problems may need many sweeps and only reach an approximation.

Once values are known, compare action candidates in each state to extract a
policy. In states 1, 2, and 3 the best action is right. The problem is solved
when we have a decision rule maximizing expected discounted return, not merely
a list of numbers. The values explain and support that decision rule.

For a finite discounted MDP, the Bellman optimality operator $T$ satisfies

$$\|TV-TU\|_\infty\le\gamma\|V-U\|_\infty.$$

$\|V-U\|_\infty$ means the largest absolute difference between two value vectors.
The inequality says one backup shrinks that difference by at least the factor
$\gamma<1$. This contraction gives a unique fixed point and convergence.
It does not guarantee convergence of arbitrary neural-network RL algorithms.

## 8. What changes when transitions are random?

Suppose an example action gives reward 0 and goes to a state valued at 1 with
probability 0.8, or a terminal state valued at 0 with probability 0.2. Its expected
candidate value is

$$0.8[0+0.95(1)]+0.2[0+0.95(0)]=0.76.$$

We average over possible outcomes **for each action**, then pick the action
with the highest expected candidate. We do not pick whichever random outcome
we would prefer. The general optimality equation is:

$$V^*(s)=\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma V^*(s')].$$

The summation means “add each outcome's probability times its return.” Terminal
state values are zero here. For evaluating a fixed stochastic policy, also average
over its action choices:

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)
[r+\gamma V^\pi(s')].$$

The first equation chooses the best action; the second evaluates the given policy.

## 9. Policy iteration: a different way to obtain the same answer

Value iteration repeatedly updates values using a maximum over actions.
**Policy iteration** alternates two separate jobs:

1. **Policy evaluation:** Hold the policy fixed. Calculate its expected returns.
2. **Policy improvement:** At each state, choose the action giving the largest
   immediate reward plus discounted evaluated continuation.

Repeat until the policy no longer changes.

The script initializes the policy to left everywhere. With the code's consistent
left-first tie rule, its nonterminal policies evolve as follows:

| Evaluated policy at states `[1,2,3]` | Its values at `[1,2,3]` | Next improved policy |
| --- | --- | --- |
| `[left,left,left]` | `[0,0,0]` | `[left,left,right]` |
| `[left,left,right]` | `[0,0,1]` | `[left,right,right]` |
| `[left,right,right]` | `[0,0.95,1]` | `[right,right,right]` |
| `[right,right,right]` | `[0.9025,0.95,1]` | Unchanged |

For example, when state 3 is known to be valuable, improvement changes state 2
to right. Evaluating that new policy makes state 2 valuable; a later improvement
changes state 1. Policy evaluation can itself require many iterative sweeps.

In matrix notation, evaluation solves

$$V^\pi=r_\pi+\gamma P_\pi V^\pi,\qquad
(I-\gamma P_\pi)V^\pi=r_\pi.$$

$P_\pi$ contains transition probabilities under the policy, $r_\pi$ contains
expected immediate rewards, and $I$ is the identity matrix. The script uses
iterative evaluation rather than a matrix solver. See the math primer for
[linear systems and curvature](../00_start/MATH_PRIMER.md).

With exact evaluation, finite states/actions, $\gamma<1$, and consistent tie
handling, policy iteration reaches an optimal policy. The code approximates
policy evaluation using a small numerical tolerance.

## 10. Read the Python and predict its output

| Code | Meaning |
| --- | --- |
| `transition(state, action)` | Ask the known model about a hypothetical move |
| `range(1, 4)` | Update states 1, 2, 3; terminals stay zero |
| `reward + gamma * (not terminated) * values[nxt]` | One move plus discounted continuation |
| `max(candidates)` | Keep the best action's candidate value |
| `updated = values.copy()` | Keep old and new sweeps separate |
| `policy_iteration()` | Alternate fixed-policy evaluation and improvement |

Run from the course folder:

```bash
python3 03_reinforcement_learning/02_mdp_value_iteration.py
```

Expected optimal values: `[0, 0.9025, 0.95, 1, 0]`. Policy iteration also prints
`[1,1,1]`, meaning right in all three nonterminal states.

## 11. Practice before moving on

1. At state 3, why is right worth 1 rather than 0.95? **Its reward is immediate.**
2. Why is the goal state's future value zero? **The episode has already ended.**
3. If $\gamma=0.5$, what are optimal values at states 1–3? **[0.25, 0.5, 1].**
4. If all values start at zero, why does state 2 not improve in sweep 1?
   **It uses the old value of state 3, which is still zero.**
5. Did value iteration need sampled episodes? **No; it used the known model.**
6. What if the model is unknown? **Learn from observed transitions, as in
   [Q-learning and SARSA](03_q_learning_sarsa.md).**
