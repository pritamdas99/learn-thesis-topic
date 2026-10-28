# Monte Carlo: model-based planning and model-free learning

Companion code: [07_monte_carlo.py](07_monte_carlo.py). Requires only Python.
Prerequisites: [MDPs, return, and value iteration](02_mdp_value_iteration.md) and
[Q-learning and SARSA](03_q_learning_sarsa.md). You should be comfortable with
sample averages ([bandits](01_bandits.md)), discounted returns, and epsilon-greedy
action selection. No neural network is needed.

## 1. What problem are we solving now?

Use the same corridor as the earlier lessons:

```text
       0          1          2          3          4
    [END] <--> [room] <--> [START] <--> [room] <--> [GOAL]
     reward 0                                     reward +1
```

Left is action `0`; right is action `1`. Every move gives reward 0 except
entering state 4, which gives 1. Entering 0 or 4 ends the episode. The discount
is $\gamma=0.95$.

Value iteration ([lesson 02](02_mdp_value_iteration.md)) used the full transition
model. Q-learning and SARSA ([lesson 03](03_q_learning_sarsa.md)) learned from one
transition at a time and **bootstrapped**: their target $r+\gamma Q(s',a')$
contained another estimate.

Monte Carlo (MC) methods take a different route: **play complete episodes, add
up what actually happened, and average.** No estimate appears inside a target.
The script shows two uses of that idea:

1. **Model-based rollout planning:** a planner that can *simulate* the corridor
   tries each first action many times in imagination and picks the better one.
2. **Model-free first-visit MC control:** a learner that can only *act* in the
   corridor plays real episodes and averages the returns it observed.

Both should prefer moving right, but they estimate different quantities, so
their numbers differ. Understanding *why* they differ is the main goal.

## 2. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $s, a$ | State and action | `state`, `action` |
| $r_t$ | Reward received for the move made at step $t$ | `reward` |
| $T$ | Number of moves in the episode | `len(trajectory)` |
| $\gamma$ | Discount factor | `gamma=0.95` |
| $G_t$ | Discounted return from step $t$ to the end | `returns[t]` |
| $\pi$ | Policy used to choose actions | random, or epsilon-greedy |
| $Q^\pi(s,a)$ | True expected return of $a$ in $s$, then following $\pi$ | — |
| $\hat Q(s,a)$ | Our estimate (a sample average) | `q[state][action]` |
| $N(s,a)$ | How many returns have been averaged for $(s,a)$ | `counts[state][action]` |
| $\epsilon$ | Probability of a uniformly random action | `epsilon=0.2` |

## 3. What does Monte Carlo estimate?

Monte Carlo replaces an expectation with an average of sampled outcomes. For a
complete episode the return from step $t$ is

$$\boxed{G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}}$$

and the action value of a policy is its expected return:

$$Q^\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].$$

If we observe $N$ independent returns $G^{(1)},\dots,G^{(N)}$ that all started
with $(s,a)$ and then followed $\pi$, their average estimates $Q^\pi(s,a)$:

$$\boxed{\hat Q^\pi(s,a)=\frac1N\sum_{i=1}^N G^{(i)}}.$$

Compare targets:

| Method | Target for $Q(s,a)$ | Contains another estimate? |
| --- | --- | --- |
| Q-learning | $r+\gamma\max_{a'}Q(s',a')$ | Yes (bootstraps) |
| SARSA | $r+\gamma Q(s',a')$ | Yes (bootstraps) |
| Monte Carlo | complete observed $G_t$ | No |

Because MC uses only observed rewards, it avoids bootstrap error. The price:
returns can have **high variance** (one lucky or unlucky episode moves a single
sample a lot), and an update must **wait until the episode ends**. Continuing
tasks with no terminal state need a different treatment; cutting an episode at
an arbitrary step is not a true terminal state.

## 4. Compute one return backward

Suppose an episode from state 2 goes `right, left, right, right`:

```text
2 --R--> 3 --L--> 2 --R--> 3 --R--> 4
rewards:   0         0         0         1
```

It is easiest to compute returns **backward** with $G_t=r_t+\gamma G_{t+1}$,
starting from $G_T=0$:

| $t$ | $(s_t,a_t)$ | $r_t$ | $G_t=r_t+0.95\,G_{t+1}$ |
| --- | --- | --- | --- |
| 3 | (3, R) | 1 | $1+0.95(0)=1$ |
| 2 | (2, R) | 0 | $0+0.95(1)=0.95$ |
| 1 | (3, L) | 0 | $0+0.95(0.95)=0.9025$ |
| 0 | (2, R) | 0 | $0+0.95(0.9025)=0.857375$ |

The code does exactly this in its reversed loop:

```python
returns, total = [0.]*len(trajectory), 0.
for t in reversed(range(len(trajectory))):
    total = trajectory[t][2] + gamma*total
    returns[t] = total
```

## 5. Model-based Monte Carlo: plan by simulating

Here "model-based Monte Carlo" means **rollout planning with a generative
transition model**: a function we may call with any state and action to get a
sampled next state and reward. It is not a single universally named RL
algorithm, and it does not automatically mean Monte Carlo Tree Search (MCTS).

In the script, that model is `model_step(state, action)`, which returns
`(next_state, reward, done)`. For each candidate first action $a$, the planner
simulates $N$ complete episodes in which every *later* action is chosen by a
continuation policy $\pi$, then picks the first action with the larger average:

$$\boxed{\hat Q^\pi(s,a)=\frac1N\sum_{i=1}^N G_0^{(i)},\qquad
a_{\mathrm{plan}}=\arg\max_a\hat Q^\pi(s,a)}.$$

The script uses $s=2$, $N=10000$ (`samples=10000`), and a **uniformly random
continuation** (`rng.randrange(2)`). Its estimates are therefore $Q^{\mathrm{random}}$,
not $Q^*$: "what is my expected return if I take $a$ now and then wander randomly?"

### Trace one simulated rollout

`simulate(2, first_action=0, rng)` might go left, then randomly right, right, right:

| Loop pass | Move | `reward` | `total += discount*reward` | `done`? | `discount` after |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 → 1 (left, the fixed first action) | 0 | 0 | no | 0.95 |
| 2 | 1 → 2 (random right) | 0 | 0 | no | 0.9025 |
| 3 | 2 → 3 (random right) | 0 | 0 | no | 0.857375 |
| 4 | 3 → 4 (random right) | 1 | $0.857375$ | yes | — |

This rollout returns 0.857375. Another rollout that drifts into state 0
returns exactly 0. The planner averages 10000 such numbers per first action.

### What should the average be?

We can check the planner against an exact calculation. Let $V(s)$ be the value
of the random policy (each action with probability 0.5). Then

$$V(1)=0.5(0)+0.5(0.95)V(2),\qquad V(3)=0.5(1)+0.5(0.95)V(2),$$
$$V(2)=0.5(0.95)V(1)+0.5(0.95)V(3).$$

Substituting the first two into the third:

$$V(2)=0.475\,[0.475V(2)+0.475V(2)+0.5]=0.45125\,V(2)+0.2375,$$
$$V(2)=\frac{0.2375}{0.54875}\approx0.43280.$$

Then

$$Q^{\mathrm{random}}(2,L)=0.95V(1)=0.45125V(2)\approx0.19530,$$
$$Q^{\mathrm{random}}(2,R)=0.95V(3)=0.45125V(2)+0.475\approx0.67030.$$

The script prints `[0.19299884270477186, 0.6678522142658854]`. Both are within
about 0.0025 of the exact values. That gap is ordinary sampling noise: the
standard deviation of a single rollout return is roughly 0.34 (left) and 0.40
(right), so with 10000 samples the standard error is about
$0.34/\sqrt{10000}\approx0.0034$ and $0.40/\sqrt{10000}=0.004$. To halve the
noise you need four times as many samples.

The planner chooses `argmax` = right (action 1), and the script asserts
`action == 1`.

### Caveats about the planner

- A better continuation policy, replanning at every step, or a search tree
  produces a different (usually better) planner. Random continuation is the
  simplest choice, not the best.
- The script's simulator is the true environment dynamics. A *learned* model
  can replace it, but then model error is added, and more simulated samples
  cannot remove that error, only the sampling noise.
- The planner only answers the question for state 2; it does not build a table.
- Simulating is "free" in the sense that it spends no real environment
  interactions; it is not free in compute.

## 6. Model-free Monte Carlo control: learn by acting

Now remove the planner's privilege. The learner `mc_control(step, ...)` is given
a `step` function and only ever calls it on the **state it is actually in** with
the action it actually chose. It never asks "what if I had gone left instead?"
That is what "model-free" means here. (The script passes `model_step` as `step`
for convenience; the learner simply does not use it counterfactually.)

Each of the `episodes=20000` iterations does three things:

1. **Play** one episode from state 2, choosing epsilon-greedy actions from the
   current table, and store `(state, action, reward)` triples.
2. **Compute** every return backward (Section 4).
3. **Update** the running average of each $(s,a)$ pair, but only at its
   **first visit** in that episode.

The update is a running mean:

$$\boxed{N(s,a)\leftarrow N(s,a)+1,\qquad
\hat Q(s,a)\leftarrow\hat Q(s,a)+\frac{G_t-\hat Q(s,a)}{N(s,a)}}.$$

This is the same incremental-average trick as the bandit lesson: it gives
exactly the ordinary mean of all returns seen so far for that pair. For example,
with observations 0.5 then 1: first estimate $0+(0.5-0)/1=0.5$, then
$0.5+(1-0.5)/2=0.75$, which equals $(0.5+1)/2$.

## 7. First visit versus every visit

Return to the episode in Section 4. The pair (2, R) occurs at $t=0$ and $t=2$.

| Pair | Visits | First-visit MC uses | Every-visit MC uses |
| --- | --- | --- | --- |
| (2, R) | $t=0$, $t=2$ | 0.857375 only | 0.857375 **and** 0.95 |
| (3, L) | $t=1$ | 0.9025 | 0.9025 |
| (3, R) | $t=3$ | 1 | 1 |

The script implements first visit with a `seen` set: the second (2, R) is
skipped. Every-visit MC is a valid alternative that this file does not implement.

If these were the first updates ($N=0$ everywhere), after this episode the
table would hold $\hat Q(2,R)=0.857375$, $\hat Q(3,L)=0.9025$, $\hat Q(3,R)=1$.

## 8. Epsilon-greedy exploration and why the first success takes a while

The script chooses actions with

```python
greedy = max(range(2), key=lambda a: q[state][a])
action = rng.randrange(2) if rng.random() < epsilon else greedy
```

With probability $1-\epsilon=0.8$ it takes the greedy action; with probability
$\epsilon=0.2$ it picks uniformly between 2 actions. With $m$ actions, the greedy
action therefore has probability $1-\epsilon+\epsilon/m=0.8+0.1=0.9$ and the other
action $\epsilon/m=0.1$.

**Tie-breaking matters.** Python's `max` returns the *first* maximiser, so when
`q[state] == [0.0, 0.0]` the "greedy" action is **left**. At the start every
entry is zero, so the learner walks left 90% of the time. Episodes that go
left from 2 then left from 1 end in state 0 with return 0, and averaging zeros
keeps the table at zero, so left stays "greedy".

With the script's seed (`random.Random(11)`), the table remains all zeros for
the first 103 episodes; every one of them ends in state 0. Episode 104 is the
first to reach the goal: `2 --R--> 3 --R--> 4`, two exploratory right moves in a row.
At that point the first-visit counts were:

| Pair | Count after episode 104 | Returns averaged | New $\hat Q$ |
| --- | --- | --- | --- |
| (3, R) | 1 | just 1 | $1$ |
| (2, R) | 10 | nine 0s, then $0.95$ | $0.95/10=0.095$ |

The value of (2, R) is only 0.095 because nine earlier tries of "right from 2"
ended up failing under the left-leaning behavior of that time. This is a
concrete example of a broader caveat: **the running average mixes returns from
every policy the learner has used**, not just the current one.

Once $\hat Q(3,R)>\hat Q(3,L)$ and $\hat Q(2,R)>\hat Q(2,L)$, the greedy actions flip
to right, and successes become frequent.

## 9. What does the learner converge toward?

The learner keeps $\epsilon=0.2$ forever, so it estimates the values of its own
**exploratory** policy (right with probability 0.9 once right is greedy),
not the optimal greedy values $Q^*$. Solving the Bellman equations for "right
with probability 0.9 in every room" gives the values in the middle column:

| Pair | Values of the 0.9-right policy | Script's learned $\hat Q$ | $Q^*$ (lesson 03) |
| --- | --- | --- | --- |
| (1, L) | 0 | 0.0 | 0 |
| (1, R) | 0.8728 | 0.8707 | 0.9025 |
| (2, L) | 0.7463 | 0.7082 | 0.857375 |
| (2, R) | 0.9379 | 0.9379 | 0.95 |
| (3, L) | 0.8728 | 0.8694 | 0.9025 |
| (3, R) | 1 | 1.0 | 1 |

Most learned numbers are close to the exploratory-policy values. (2, L) is
noticeably lower, consistent with the mixing effect from Section 8: many of its
returns were collected early, when the policy still leaned left. (1, L) and
(3, R) are exact because those moves always end immediately with reward 0 and 1.
The learned numbers are a finite-sample result for one seed, not a proof of
convergence.

Classical convergence of MC control to $Q^*$ needs additional conditions, for
example **GLIE**: *greedy in the limit with infinite exploration* (every pair
keeps being tried, while $\epsilon$ shrinks toward zero). The script does not
decay $\epsilon$, so it does not satisfy GLIE. Its greedy policy is nonetheless
correct: right is preferred in states 1–3.

## 10. Compare all the targets

| Method | Target | Model access | Bootstraps? |
| --- | --- | --- | --- |
| Value iteration | $\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma V(s')]$ | Full model | Yes |
| MC rollout planner | Mean of simulated complete returns | Generative model | No (here) |
| Model-free MC | Complete observed $G_t$ | None | No |
| SARSA | $r+\gamma Q(s',a')$ | None | Yes |
| Q-learning | $r+\gamma\max_{a'}Q(s',a')$ | None | Yes |

Two different axes appear here: *does the method need a model?* and *does its
target use another estimate?* Monte Carlo can appear on either side of the first
axis, but it never bootstraps.

## 11. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `model_step(state, action)` | Generative model: `nxt = state ± 1`, reward `float(nxt == 4)`, done if `nxt in (0, 4)` |
| `simulate(state, first_action, rng, gamma=0.95)` | One rollout: fixed first action, then `rng.randrange(2)` random continuation; returns $G_0$ |
| `total += discount*reward`; `discount *= gamma` | Accumulate $\sum_k\gamma^k r_k$ forward |
| `rollout_plan(state=2, samples=10000)` | Average 10000 rollouts per first action, using `random.Random(7)` |
| `max(range(2), key=lambda a: q[a])` | $\arg\max_a\hat Q^{\mathrm{random}}(2,a)$; first index wins ties |
| `mc_control(step, episodes=20000, gamma=0.95, epsilon=0.2)` | Model-free first-visit MC control, `random.Random(11)` |
| `q, counts = [[0., 0.] ...], [[0, 0] ...]` | $\hat Q$ table and $N(s,a)$ counts, 5 rows × 2 actions |
| `rng.randrange(2) if rng.random() < epsilon else greedy` | Epsilon-greedy behavior |
| `trajectory.append((state, action, reward))` | Store the episode; no update until it ends |
| reversed loop `total = trajectory[t][2] + gamma*total` | Backward return computation $G_t=r_t+\gamma G_{t+1}$ |
| `seen` set | First-visit rule |
| `q[state][action] += (total-q[state][action])/counts[state][action]` | Running mean of returns |

## 12. Run it

From the course folder:

```bash
python3 03_reinforcement_learning/07_monte_carlo.py
```

Output on our machine:

```text
model-based random-continuation Q: [0.19299884270477186, 0.6678522142658854] chosen action: 1
model-free MC Q: [[0.0, 0.8707445185535772], [0.708197832642094, 0.9379273909513608], [0.8694426217271125, 1.0]]
```

The first line is $[\hat Q^{\mathrm{random}}(2,L),\hat Q^{\mathrm{random}}(2,R)]$ and the
planned action. The second line shows `[left, right]` rows for states 1–3 of the
learned table. Both random generators are seeded, so the numbers are reproducible.

The assertions check only that the planner chooses right at state 2 and that
right has the larger learned value in each of states 1–3. They do not check that
any estimate equals $Q^\pi$ or $Q^*$.

## 13. Practice calculations

1. An episode has rewards `[0, 0, 1]` with $\gamma=0.95$. What are $G_0,G_1,G_2$?
   **$G_2=1$, $G_1=0.95$, $G_0=0.9025$.**
2. $\hat Q(s,a)=0.6$ after $N=4$ returns. A fifth return is 1.0. What is the new
   estimate? **$0.6+(1.0-0.6)/5=0.68$.**
3. In episode `2 -R-> 3 -L-> 2 -L-> 1 -L-> 0`, what return does first-visit MC
   assign to (2, R), and what does every-visit MC do with (2, L)? **All rewards
   are zero, so (2, R) gets 0; (2, L) occurs once, so both variants use its
   single return, also 0.**
4. With $\epsilon=0.2$ and two actions, what is the probability of the non-greedy
   action? **$\epsilon/2=0.1$.**
5. Why does the planner's $\hat Q(2,R)\approx0.668$ differ from the learner's
   $\approx0.938$? **Different continuation policies: uniformly random versus
   epsilon-greedy that mostly moves right.**
6. If you want the planner's standard error to fall from about 0.004 to 0.002,
   how many samples do you need? **About four times as many: 40000.**

Exercise: evaluate a fixed random policy with model-free MC (set every action to
`rng.randrange(2)`) and compare its estimates for state 2 against the rollout
planner over five seeds. Then change `samples` and observe how the spread of the
planner's estimates changes.

Reference: [Sutton and Barto, chapters 4–6](http://incompleteideas.net/book/the-book-2nd.html).
Next: [A2C and A3C: advantage actor-critic](08_a2c_a3c.md).
