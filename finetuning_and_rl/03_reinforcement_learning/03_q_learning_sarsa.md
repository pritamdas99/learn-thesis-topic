# Q-learning and SARSA: learn a route by trying actions

Companion code: [03_q_learning_sarsa.py](03_q_learning_sarsa.py).
Prerequisites: [bandit averages](01_bandits.md) and
[MDPs, return, and value iteration](02_mdp_value_iteration.md).
Read this chapter with those two examples in mind; no neural network is needed.

## 1. What problem are we solving now?

Use the same corridor:

```text
       0          1          2          3          4
    [END] <--> [room] <--> [START] <--> [room] <--> [GOAL]
     reward 0                                     reward +1
```

Left is action `0`; right is action `1`. Each move gives reward zero except
entering state 4, which gives 1. Entering 0 or 4 ends the episode.

The goal is to learn which move to choose in each room to obtain a high
**discounted total reward**, not just an immediate reward. Both methods should
learn a greedy decision rule that moves right in all three nonterminal rooms.

Value iteration knew the model and calculated hypothetical moves. These learners
instead choose an action, observe what happens, and update from that experience.
The script still needs environment code to simulate movement, but the learner's
update only uses the observed transition. It does not enumerate a known transition
model to plan. This is the meaning of **model-free** here.

## 2. What information does one experience contain?

An observed transition might be:

```text
Current state: 2
Chosen action: right
Observed reward: 0
Next state: 3
Terminated: False
```

We abbreviate this as $(s,a,r,s',d)$:

| Symbol | Meaning | Example |
| --- | --- | --- |
| $s$ | Current state | 2 |
| $a$ | Selected action | Right |
| $r$ | Reward just received | 0 |
| $s'$ | Next state; prime means “next” | 3 |
| $d$ | 1 if task ended, otherwise 0 | 0 |

A terminal example is $(3,\text{right},1,4,1)$.
The transition is a sample of experience, not a table of every possible outcome.

## 3. What exactly is a Q-value?

A state value $V(s)$ asks, “How valuable is being here?” An action value
$Q(s,a)$ asks, **“How valuable is taking this particular action here, including
what happens afterward?”** The continuation policy must also be specified.

For a fixed policy $\pi$, the true action value is

$$q^\pi(s,a)=E[G_t\mid S_t=s,A_t=a,\text{ then follow }\pi].$$

$G_t=r_t+\gamma r_{t+1}+\cdots$ is discounted return. Uppercase $Q$ below is the
learner's table of estimates. Unlike a bandit's $Q(a)$, it needs a state index,
because “right” can have different consequences in different locations.

The code creates a table with five rows and two columns:

```python
q = [[0.0, 0.0] for _ in range(5)]
# q[state][0] estimates left; q[state][1] estimates right
```

| State | Estimate for left | Estimate for right |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 0 | 0 |
| 4 | 0 | 0 |

Zero initialization means no useful evidence has been incorporated yet. It does
not mean the agent knows that all actions truly have zero future value.
Only the selected state/action entry changes on each update.

## 4. The basic update: move an estimate toward a target

Both algorithms use

$$\boxed{Q_{new}(s,a)=Q_{old}(s,a)+\alpha[y-Q_{old}(s,a)]}.$$

| Part | Interpretation |
| --- | --- |
| $Q_{old}(s,a)$ | Current prediction of future return |
| $y$ | New target, constructed from the observed reward and estimated future |
| $y-Q_{old}(s,a)$ | Prediction error, called the temporal-difference or TD error |
| $\alpha$ | Learning rate: how far to move toward the target |

The code uses $\alpha=0.2$. With old estimate 0.4 and target 0.9:

$$Q_{new}=0.4+0.2(0.9-0.4)=0.4+0.1=0.5.$$

Equivalently, $Q_{new}=(1-\alpha)Q_{old}+\alpha y$: retain 80% of the old
estimate and incorporate 20% of the new target. The target is not necessarily
the true answer; it often contains another imperfect estimate.

This resembles the bandit update $Q\leftarrow Q+(R-Q)/n$. There are two changes:
we use a constant step size 0.2 rather than $1/n$, and the target includes
**future return**, not just the newest immediate reward.

## 5. Q-learning: build the target using the best estimated next action

For a nonterminal transition, Q-learning uses

$$\boxed{y_Q=r+\gamma\max_{a'}Q(s',a')}.$$

The symbol $a'$ means a candidate next action. `max` takes the largest value
in the next state's row. The code uses $\gamma=0.95$.

Suppose we move right from 2 to 3, receiving reward 0, and the next-state row is

```text
Q(3, left)  = 0.2
Q(3, right) = 0.8
```

Then

$$y_Q=0+0.95\max(0.2,0.8)=0.95(0.8)=0.76.$$

If $Q(2,\text{right})=0.4$, its updated value is

$$0.4+0.2(0.76-0.4)=0.472.$$

The prediction increases because the observed move leads to a state with a
promising estimated continuation. We need not receive the final reward on this
particular step to improve the estimate.

## 6. Watch reward information travel backward through actual experience

Start with all entries zero. Consider these selected transitions in chronological
order; the table shows only the entry updated on that transition:

| Observation | Next-state best estimate | Target | Old entry | New entry |
| --- | --- | --- | --- | --- |
| $2\to3$, right, reward 0 | 0 | 0 | $Q(2,R)=0$ | 0 |
| $3\to4$, right, reward 1, terminal | No continuation | 1 | $Q(3,R)=0$ | 0.2 |
| Later: $2\to3$, right, reward 0 | 0.2 | $0.95(0.2)=0.19$ | $Q(2,R)=0$ | 0.038 |
| Later: $3\to4$, right, reward 1, terminal | No continuation | 1 | $Q(3,R)=0.2$ | 0.36 |
| Later: $1\to2$, right, reward 0 | 0.038 | $0.95(0.038)=0.0361$ | $Q(1,R)=0$ | 0.00722 |

The first episode's $2\to3$ update remains zero because the learner has not yet
learned that state 3 is valuable. A later visit can use the newly improved row.
The learner does not automatically revise every earlier transition after finding
the goal. This script has no replay or backward episode-return pass.

For repeated terminal transitions $3\to4$, the estimate follows
`0 -> 0.2 -> 0.36 -> 0.488 -> ...`, approaching 1. Each update closes 20% of
the remaining gap. In this special deterministic terminal case,
$Q_k(3,R)=1-0.8^k$ if it began at zero and received $k$ such updates.

Using an estimate inside another estimate's target is called **bootstrapping**.
Using a reward plus a next-state estimate rather than waiting for the complete
observed return is the central temporal-difference idea.

## 7. SARSA: use the next action actually selected

SARSA uses the same update structure, but its nonterminal target is

$$\boxed{y_{SARSA}=r+\gamma Q(s',a')}.$$

Here $a'$ is the next action **sampled from the agent's behavior rule**, not an
argmax placeholder. SARSA stands for **State, Action, Reward, next State, next Action**.

Use exactly the earlier numbers: $Q(3,L)=0.2$, $Q(3,R)=0.8$, reward 0,
$\gamma=0.95$, current $Q(2,R)=0.4$, and $\alpha=0.2$.

| Next action selected | Q-learning target | SARSA target | SARSA updated $Q(2,R)$ |
| --- | --- | --- | --- |
| Right | 0.76 | $0.95(0.8)=0.76$ | 0.472 |
| Left, due to exploration | 0.76 | $0.95(0.2)=0.19$ | $0.4+0.2(0.19-0.4)=0.358$ |

In the second row, Q-learning still assumes the best estimated continuation;
SARSA accounts for the exploratory left action the agent will actually take.
Neither calculation is mistaken. They are estimating different continuations.
When the sampled next action is greedy, their targets agree for that transition.

The script saves `next_action` and actually executes it next. Choosing a fresh,
different action before the next move would break this sampled SARSA sequence.

## 8. On-policy and off-policy in ordinary language

A **policy** is a rule for choosing actions. The **behavior policy** generates
the agent's experience. The **target policy** is the policy whose return it learns.

| Method | How it acts during training | What its update assumes afterward |
| --- | --- | --- |
| Q-learning | Epsilon-greedy, sometimes random | Greedy continuation: pick the best estimated action |
| SARSA | Epsilon-greedy, sometimes random | Continue with that same exploratory policy |

Q-learning is **off-policy** because behavior and target can differ. SARSA is
**on-policy** because its target uses the same policy that generates experience.
“Off-policy” does not mean the program stops following a policy.

The script uses $\epsilon=0.2$ and two actions. When right uniquely has the
higher estimate, right is chosen with probability $0.8+0.2/2=0.9$ and left with
probability 0.1. Equal-value ties are broken randomly.

With the next-state row `[0.2,0.8]`, SARSA's conditional expected next-value
sample is $0.1(0.2)+0.9(0.8)=0.74$, giving an expected target of $0.95(0.74)=0.703$.
Ordinary SARSA samples either value on each update; it does **not** explicitly
calculate this average. Using that average directly would be Expected SARSA,
a related algorithm that is not implemented in this file.

Continued exploration can delay reward or terminate at the wrong end. SARSA's
values can therefore be lower even when its greedy action choices match Q-learning.

## 9. Why can risky routes expose the difference?

Imagine a separate grid task with a short route beside a cliff and a longer
route away from it. A greedy agent can follow the short route safely, but a random
exploratory action near the cliff can incur a large penalty.

Q-learning's target values greedy future decisions, so it can favor the short
route while the actual exploratory behavior still falls occasionally. SARSA
includes those exploratory consequences in its targets and can prefer the safer
route. This illustrates a tendency in such tasks, not a guarantee that SARSA is
always safer or better. The provided corridor has no cliff and does not run
that experiment.

## 10. Terminal states and collection limits

The complete target uses a terminal indicator $d$:

$$y_Q=r+\gamma(1-d)\max_{a'}Q(s',a'),$$
$$y_{SARSA}=r+\gamma(1-d)Q(s',a').$$

On entering state 4, $r=1$ and $d=1$, so both targets are exactly 1. There is
no future decision after termination. On entering state 0, both targets are 0.

For example, with terminal reward 1, old Q 0.5, and learning rate 0.2:

$$Q_{new}=0.5+0.2(1-0.5)=0.6.$$

The code also stops collecting after 30 steps. This is a practical collection
limit, not an extra terminal state in the underlying corridor. A nonterminal
30th transition still includes its next-state bootstrap. The next collection
starts at 2. If a deadline were an actual task rule affecting optimal behavior,
we would need to model the remaining time as part of the state.

The code calls `choose(nxt)` even on a terminal transition. That selected action
is never executed and its value is multiplied by zero, so it does not affect the
target. It does consume a random-number draw, affecting the later random sequence.

## 11. What converges, and what has been solved?

For Q-learning, the goal is a table approximating **optimal action values** $Q^*$.
At the optimal fixed point in this corridor, the nonterminal rows are:

| State | $Q^*(s,\text{left})$ | $Q^*(s,\text{right})$ | Greedy action |
| --- | --- | --- | --- |
| 1 | 0 | $0.95^2=0.9025$ | Right |
| 2 | $0.95^3=0.857375$ | $0.95$ | Right |
| 3 | $0.95^2=0.9025$ | $1$ | Right |

Why is moving left from 3 worth 0.9025 rather than zero? It leads to state 2;
optimal continuation can turn around: `3 -> 2 -> 3 -> 4`, rewards `[0,0,1]`.
A suboptimal first action need not make eventual success impossible.

For SARSA with fixed exploration, the goal is consistent action values for its
continuing exploratory policy, not necessarily $Q^*$. Since the policy changes
as the table changes, values and behavior improve together during training.

The code runs 500 episodes with constant learning rate 0.2. It does not use a
convergence stopping test. A finite run can produce useful decisions without
establishing mathematical convergence. Classical tabular convergence results
need additional conditions, including sufficient state/action visits and suitable
step sizes; optimal-control SARSA also needs exploration to become greedy under
appropriate conditions. Here persistent exploration remains part of behavior.

**What is solved in the demonstration?** From observed rewards, both learners
find table entries that favor right in states 1–3. The extracted greedy policy
reaches the goal by the shortest route. Training behavior still sometimes explores.

## 12. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `choose(state)` | Sample an epsilon-greedy behavior action |
| `nxt = state + ...` | Simulate the chosen move in the environment |
| `reward = float(nxt == 4)` | Environment returns 1 only on goal arrival |
| `next_action = choose(nxt)` | Select the next behavior action |
| `max(q[nxt])` | Q-learning's greedy next-value estimate |
| `q[nxt][next_action]` | SARSA's sampled next-value estimate |
| `target = reward + 0.95 * ...` | Construct the TD target |
| `q[state][action] += 0.2 * (target-q[state][action])` | Update one visited entry |
| `state, action = nxt, next_action` | Carry the sampled next action into the next move |

Run from the course folder:

```bash
python3 03_reinforcement_learning/03_q_learning_sarsa.py
```

Each printed row contains `[left value, right value]`; the output shows only
states 1–3. The assertion checks that right has the larger value in each row.
It does not check that every number has converged or that SARSA equals $Q^*$.

## 13. Practice calculations

1. A nonterminal transition gives reward 0, next values `[0.3,0.9]`, and
   $\gamma=0.95$. What is Q-learning's target? **$0.95(0.9)=0.855$.**
2. If SARSA samples the first action, what is its target? **$0.95(0.3)=0.285$.**
3. With old value 0.5 and $\alpha=0.2$, what are the two updated values?
   **Q-learning: 0.571. SARSA: 0.457.**
4. If the same transition were terminal with reward 0, what would both targets
   be? **Zero, regardless of the next-state table.**
5. What would $\alpha=1$ do? **Replace the old estimate with the latest target.**
6. Is $\gamma=0.95$ the exploration probability? **No. Gamma discounts future
   rewards; epsilon controls random action selection; alpha controls update size.**

Try changing one of these three parameters at a time and compare several seeds.
Explain both the learned numbers and the greedy actions. Next:
[DQN: replacing a Q-table with a neural network](04_dqn.md).
