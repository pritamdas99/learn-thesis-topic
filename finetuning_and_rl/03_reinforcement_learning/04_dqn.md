# DQN and Double DQN: learning action values with a neural network

Companion code: [04_dqn.py](04_dqn.py). Requires PyTorch.
Prerequisites: [MDPs, return, and value iteration](02_mdp_value_iteration.md),
[Q-learning and SARSA](03_q_learning_sarsa.md), and
[neural network training](../01_foundations/03_neural_network_training.md).
Read this explanation before running the code.

## 1. What problem are we solving now?

The script keeps exactly the same corridor as the Q-learning lesson, so that the
only new ingredient is the neural network:

```text
       0          1          2          3          4
    [END] <--> [room] <--> [START] <--> [room] <--> [GOAL]
     reward 0                                     reward +1
```

Action `0` is left and action `1` is right. Entering state 4 gives reward 1;
every other move gives 0. Entering 0 or 4 ends the episode. The discount is
$\gamma=0.95$, and the goal is again a greedy rule that moves right in states 1–3.

In lesson 03 we stored one number per state/action pair in a table
`q[state][action]`. That works for 5 states. It does not work for an Atari
screen, a robot's joint angles, or any large or continuous observation: there
are too many possible inputs to give each one its own row, and most would never
be visited even once.

**DQN (deep Q-network)** replaces the table with a function $Q_\theta(s,a)$
computed by a neural network with parameters $\theta$. The network reads an
encoding of the state and outputs one estimate per action. Because all states
share the same weights, a training step on one state can also change the
predictions for other states. That **generalization** is the point, but it also
combines three things that are individually fine and jointly risky:
bootstrapped targets, function approximation, and off-policy data. Stability is
no longer as simple as in the tabular case, which is why DQN adds a replay
buffer and a target network.

The corridor is far too easy to *need* a network. It is used so you can check
every number by hand and isolate what neural approximation changes.

## 2. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $s, a, r, s', d$ | One transition: state, action, reward, next state, terminated flag | `(state, action, float(nxt == 4), nxt, terminated)` |
| $x(s)$ | Network input for state $s$: a one-hot vector of length 5 | `features[state]`, e.g. `features[2] = [0,0,1,0,0]` |
| $Q_\theta(s,\cdot)$ | **Online** network: outputs `[Q(s,left), Q(s,right)]` | `net(features[s])` |
| $\theta$ | Online network parameters, trained by Adam | `net.parameters()` |
| $Q_{\bar\theta}(s,\cdot)$ | **Target** network: a lagged copy used to build targets | `target(features[ns])` |
| $\bar\theta$ | Target parameters, copied from $\theta$ every 50 updates | `target.load_state_dict(net.state_dict())` |
| $y$ | Regression target built from $r$ and the next state | `targets` |
| $\gamma$ | Discount factor, 0.95 | `0.95*(1-done)*next_q` |
| $\epsilon$ | Exploration probability | `epsilon = max(0.05, 1-episode/150)` |
| $\mathcal D$ | Replay buffer of past transitions, at most 2000 | `replay = deque(maxlen=2000)` |
| $B$ | Minibatch size, 32 | `rng.sample(list(replay), 32)` |

The bar over $\bar\theta$ means "the frozen copy", not an average.

## 3. What does the network compute?

```python
net = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 2))
```

Input: a one-hot vector of length 5. Hidden layer: 32 ReLU units. Output: two
numbers, the estimated values of left and right. One forward pass gives the
values of **all** actions in a state, which makes $\max_a$ and $\arg\max_a$ cheap.

With one-hot inputs, the first layer effectively looks up one column of its
weight matrix per state, so this network is close to a table. It is still not
exactly a table: the hidden bias, the ReLU, and the whole second layer are
shared by every state. An update for state 2 therefore also nudges the outputs
for states 1, 3, and even the terminal states 0 and 4.

Shapes in a minibatch of $B=32$:

| Expression | Shape | Meaning |
| --- | --- | --- |
| `features[s]` | `[32, 5]` | One-hot rows for the sampled states |
| `net(features[s])` | `[32, 2]` | Both action values for each sampled state |
| `a[:, None]` | `[32, 1]` | The actions that were actually taken |
| `.gather(1, a[:, None])` | `[32, 1]` | Pick, in each row, the column of the taken action |
| `.squeeze(1)` | `[32]` | Drop the extra dimension: `predicted` |
| `r`, `done`, `targets` | `[32]` | Must also be one-dimensional |

Worked `gather` example with a batch of 3:

```text
net rows:  [[0.2, 0.5],    a = [1, 1, 0]   -> gather -> [0.5, 0.7, 0.3]
            [0.1, 0.7],
            [0.3, 0.0]]
```

Only the value of the action actually taken can be compared with the observed
outcome; the other action's output receives no gradient from that row.
If `predicted` had shape `[32,1]` and `targets` had shape `[32]`, subtraction
would silently broadcast to `[32,32]`. Matching one-dimensional shapes avoid
that bug.

## 4. Regression toward a detached Bellman target

Q-learning's update moved a table entry toward $y=r+\gamma\max_{a'}Q(s',a')$.
DQN turns the same idea into a **regression problem**: make the network's
prediction $Q_\theta(s,a)$ close to a label $y$. Classic DQN builds the label
with the separate target network $Q_{\bar\theta}$:

$$\boxed{y=r+\gamma(1-d)\max_{a'}Q_{\bar\theta}(s',a')}$$

| Part | Interpretation |
| --- | --- |
| $r$ | Reward observed on this transition |
| $(1-d)$ | 0 for a terminal transition, so no future value is added |
| $\max_{a'}Q_{\bar\theta}(s',a')$ | Best estimated continuation according to the frozen network |

The targets are computed inside `torch.no_grad()`. That is essential: $y$ is
treated as a fixed label. The update should move the prediction toward the
target, not also move the target toward the prediction through the same
backward pass.

**Worked target example.** Suppose three sampled transitions are

| Transition | $r$ | $d$ | Target-net row $Q_{\bar\theta}(s',\cdot)$ | $y$ |
| --- | --- | --- | --- | --- |
| $2\to3$, right | 0 | 0 | `[0.6, 0.9]` | $0+0.95(0.9)=0.855$ |
| $3\to4$, right | 1 | 1 | `[5.0, 7.0]` | $1+0.95(0)(7.0)=1$ |
| $1\to0$, left | 0 | 1 | `[2.0, 3.0]` | $0$ |

The rows for terminal states 0 and 4 can contain arbitrary numbers. Those states
never appear as the *current* state `s` in replay, so the network is never
trained on them. The mask $(1-d)$ makes their values irrelevant. (In the default
run, an instrumented copy of the script printed values near 0.5 for those rows,
which is meaningless but harmless.)

## 5. The loss: Smooth L1 and gradient clipping

For each sampled transition let $u=Q_\theta(s,a)-y$ be the prediction error.
The script uses `nn.functional.smooth_l1_loss`, whose default threshold is 1:

$$\boxed{\ell(u)=\begin{cases}u^2/2 & |u|<1\\ |u|-1/2 & |u|\ge 1\end{cases}}$$

and averages $\ell$ over the 32 transitions. Its derivative is $u$ near zero and
$\pm1$ for large errors, so one extreme TD target cannot produce an arbitrarily
large gradient. This is the same idea as the Huber loss.

Continue the example with predictions `[0.5, 0.7, 0.3]` from section 3 and
targets `[0.855, 1, 0]`:

| Row | $u=$ prediction $-$ target | $\ell(u)$ | $\partial(\text{mean loss})/\partial\,\text{prediction}=u/3$ |
| --- | --- | --- | --- |
| 1 | $0.5-0.855=-0.355$ | $0.355^2/2=0.0630125$ | $-0.1183$ |
| 2 | $0.7-1=-0.3$ | $0.045$ | $-0.1$ |
| 3 | $0.3-0=0.3$ | $0.045$ | $+0.1$ |

Mean loss $=(0.0630125+0.045+0.045)/3\approx0.0510$. Gradient descent raises
predictions 1 and 2 and lowers prediction 3, each toward its target.

Large-error comparison: prediction 0.5, target 3.5, so $u=-3$. Smooth L1 gives
loss $3-0.5=2.5$ and derivative $-1$. Plain $u^2/2$ would give loss 4.5 and
derivative $-3$. The robust loss limits the push from a single surprising label.

After `loss.backward()`, `clip_grad_norm_(net.parameters(), 10.0)` rescales the
whole parameter gradient if its norm exceeds 10. This is an additional bound on
the size of one optimizer step, applied to parameters rather than to errors.

## 6. Why a target network? Watch the label move

Without a target network, the label is built from the same $\theta$ that is
being updated. Raising $Q_\theta(2,R)$ also changes shared weights, which can
raise $Q_\theta(3,\cdot)$, which raises the next label for $Q_\theta(2,R)$.
The regression keeps chasing a target that runs away from it.

The script keeps a frozen copy:

```python
target = copy.deepcopy(net).requires_grad_(False)
...
if updates % 50 == 0:
    target.load_state_dict(net.state_dict())
```

Between copies, labels for a given transition do not change, so for 50 updates
the online network solves an ordinary, fixed regression problem. Then the target
jumps to the improved network, and the next phase starts. Copying every update
weakens this effect; copying very rarely makes targets stale and slows the
backward spread of reward information you saw in lesson 03.

In the default run, an instrumented copy of the script counted 559 optimizer
updates, so the target was refreshed $\lfloor 559/50\rfloor=11$ times.

## 7. Why replay? Reuse experience and break correlations

Consecutive transitions are strongly correlated: `2 -> 3 -> 2 -> 3 -> ...`
Training only on the newest transition would fit a few states at a time and
then overwrite them. A **replay buffer** stores past transitions and samples a
random minibatch for each update.

In this script:

1. Every environment step appends one tuple to `replay` (capacity 2000; the
   oldest are dropped first when full).
2. Once `len(replay) >= 32`, **every** environment step also performs one
   gradient update on 32 transitions sampled without replacement by
   `rng.sample(list(replay), 32)`.
3. A transition can therefore be reused in many updates.

This reduces within-batch temporal correlation. It is not an unbiased sample of
the current policy's state distribution: old transitions came from older, more
random behavior. Q-learning targets are off-policy (they use $\max_{a'}$, not the
behavior's next action), which is what makes learning from old data legitimate.

Script-specific caveat: in the default run the instrumented copy counted 590
stored transitions in total, so the 2000-transition capacity is never reached
and nothing is ever evicted. On larger tasks the capacity matters.

## 8. Exploration schedule

```python
epsilon = max(0.05, 1-episode/150)
```

| Episode | $\epsilon$ | Behavior |
| --- | --- | --- |
| 0 | 1.0 | Fully random |
| 50 | $1-50/150\approx0.667$ | Mostly random |
| 75 | 0.5 | Half random |
| 100 | $\approx0.333$ | Mostly greedy |
| 142 | $\approx0.0533$ | Almost at the floor |
| 143–199 | 0.05 | Floor: 5% random actions |

The greedy branch uses the **online** network, `net(features[state]).argmax()`.
Each episode starts at state 2 and stops at a terminal state or after 30 steps.
`terminated` masks bootstrap; the 30-step collection limit does not mark a
transition as terminal (as in lesson 03, it is a collection limit, not a task rule).

## 9. Why Double DQN separates selection and evaluation

Maximizing noisy estimates favors actions whose errors happen to be positive.
Tiny example: both actions in $s'$ truly have value 0.5, but each estimate is
independently 0.4 or 0.6 with equal probability. The max is 0.4 only if both
are low (probability 1/4):

$$E[\max]=0.25(0.4)+0.75(0.6)=0.55>0.5.$$

The target is biased upward even though each individual estimate is unbiased.
Through bootstrapping, that bias can be copied into earlier states.

Double DQN **selects** the next action with the online network and **evaluates**
it with the target network:

$$\boxed{a^*=\arg\max_{a'}Q_\theta(s',a'),\qquad y=r+\gamma(1-d)\,Q_{\bar\theta}(s',a^*)}$$

If the evaluator's errors were independent of the selector's, a lucky selection
would no longer also be scored with its own lucky error. The online and target
networks are not truly independent (the target is a lagged copy), so this
**reduces** overestimation; it does not eliminate all estimation error.

Example from the original lesson: online next-state values are `[3,2]` and
target values are `[1,4]`. Ordinary target-network DQN bootstraps from
$\max(1,4)=4$. Double DQN selects action 0 from the online values, then
evaluates it as 1 with the target network.

Redo the first row of section 4 with Double DQN. Suppose the online row for
state 3 is `[0.8, 0.7]` while the target row is still `[0.6, 0.9]`:

| Rule | Next action | Next value | Target $y$ | Error $u$ for prediction 0.5 |
| --- | --- | --- | --- | --- |
| DQN | $\arg\max$ of target row = right | 0.9 | 0.855 | $-0.355$ |
| Double DQN | $\arg\max$ of online row = left | 0.6 | $0.95(0.6)=0.57$ | $-0.07$ |

Terminal rows are unchanged (both still give 1 and 0), so the three-row Smooth L1
loss becomes $(0.00245+0.045+0.045)/3\approx0.0308$ instead of 0.0510.
Neither number is "correct"; the two rules define different regression labels.

## 10. Run either target rule

The default uses Double DQN. Run

```bash
python3 03_reinforcement_learning/04_dqn.py --vanilla-dqn
```

to use the original DQN target $r+\gamma(1-d)\max_a Q_{target}(s',a)$. Both
branches share the same environment, replay buffer, optimizer, and target
network update schedule, making the target-selection difference directly
inspectable. Because the targets differ, the networks and therefore the
sampled trajectories diverge after the first updates, so the two runs do not
see identical data.

## 11. What should the learned values be?

The optimal action values from lesson 03 are the reference:

| State | $Q^*(s,\text{left})$ | $Q^*(s,\text{right})$ | Greedy action |
| --- | --- | --- | --- |
| 1 | 0 | $0.95^2=0.9025$ | Right |
| 2 | $0.95^3=0.857375$ | 0.95 | Right |
| 3 | $0.95^2=0.9025$ | 1 | Right |

The script prints only greedy actions, not values. For checking, a scratch copy
of the script that additionally printed `net(features)` gave, after training,
approximately `[0.0, 0.903]`, `[0.858, 0.95]`, `[0.903, 1.0]` for states 1–3 in
the default (Double DQN) run and `[0.0, 0.902]`, `[0.857, 0.95]`, `[0.903, 1.0]`
with `--vanilla-dqn`. Both are close to $Q^*$ here. This tiny deterministic task
gives little room for overestimation, so it does not demonstrate Double DQN's
advantage; that shows up on noisier tasks with many actions.

## 12. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `net = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 2))` | Online network $Q_\theta$: 5 one-hot inputs, 2 action values |
| `target = copy.deepcopy(net).requires_grad_(False)` | Frozen target network $Q_{\bar\theta}$ |
| `torch.optim.Adam(net.parameters(), lr=0.003)` | Only the online network is optimized |
| `replay = deque(maxlen=2000)` | Replay buffer $\mathcal D$ |
| `features = torch.eye(5)` | One-hot state encodings |
| `epsilon = max(0.05, 1-episode/150)` | Linear exploration decay with floor 0.05 |
| `rng.randrange(2) if rng.random() < epsilon else net(features[state]).argmax().item()` | Epsilon-greedy behavior from the online net |
| `replay.append((state, action, float(nxt == 4), nxt, terminated))` | Store $(s,a,r,s',d)$ |
| `if len(replay) >= 32:` | Start updating once a minibatch is available |
| `zip(*rng.sample(list(replay), 32))` | Random minibatch without replacement |
| `net(features[ns]).argmax(-1, keepdim=True)` | Double DQN: online network **selects** $a^*$ |
| `target(features[ns]).gather(1, next_actions).squeeze(1)` | Double DQN: target network **evaluates** $a^*$ |
| `target(features[ns]).max(-1).values` | Vanilla DQN: target network selects and evaluates |
| `targets = r + 0.95*(1-done)*next_q` | Bellman target $y$, computed under `no_grad` |
| `net(features[s]).gather(1, a[:, None]).squeeze(1)` | Prediction $Q_\theta(s,a)$ for taken actions, shape `[32]` |
| `nn.functional.smooth_l1_loss(predicted, targets)` | Robust regression loss |
| `nn.utils.clip_grad_norm_(net.parameters(), 10.0)` | Bound the gradient norm |
| `if updates % 50 == 0: target.load_state_dict(net.state_dict())` | Refresh target every 50 updates |
| `net(features[1:4]).argmax(-1).tolist()` | Final greedy actions for states 1–3 |

The closing comment mentions **prioritized replay**, an extension not
implemented here: sample large-TD-error transitions more often and use
importance weights to correct the changed sampling distribution.

## 13. Run and investigate

Run from the course folder:

```bash
python3 03_reinforcement_learning/04_dqn.py
python3 03_reinforcement_learning/04_dqn.py --vanilla-dqn
```

Both commands print

```text
greedy actions for states 1,2,3: [1, 1, 1]
```

meaning the greedy policy from the online network moves right in all three
nonterminal states. The script contains **no assertion**: it prints the actions
and you inspect them. It also does not print Q-values, losses, or returns, and
it evaluates only the greedy actions, not the exploratory actions used during
training.

Experiments: remove the target delay (copy every update) or replace the replay
sample with only the newest transition, one change at a time, and compare
several seeds. This easy corridor may still work after removing a stabilizer;
that does not disprove its value on harder tasks.

References: [DQN](https://www.nature.com/articles/nature14236),
[Double DQN](https://arxiv.org/abs/1509.06461).

## 14. Practice calculations

1. A non-terminal transition has $r=0$, target-network next row `[0.4, 0.7]`,
   $\gamma=0.95$. What is the DQN target? **$0.95(0.7)=0.665$.**
2. The online next row is `[0.9, 0.2]`. What is the Double DQN target?
   **Online selects left; target evaluates it: $0.95(0.4)=0.38$.**
3. The transition $3\to4$ has $r=1$, $d=1$, and a target-network row
   `[8, 9]`. What is $y$? **1. The terminal mask removes the next-state value.**
4. Prediction 0.2, target 0.665. What is the Smooth L1 loss?
   **$u=-0.465$, $\ell=0.465^2/2\approx0.1081$.**
5. Prediction 0, target 2.5. Smooth L1 loss and its derivative?
   **$u=-2.5$; $\ell=2.5-0.5=2$; derivative $-1$.**
6. What is $\epsilon$ at episode 120? **$1-120/150=0.2$.**
7. If `net(features[s])` has shape `[32,2]`, what shape does
   `gather(1, a[:, None])` return? **`[32,1]`; `squeeze(1)` makes it `[32]`.**
8. Why does one update on state 2 change the output for state 1?
   **The hidden bias and second layer are shared by all states.**

Next: [policy gradients, REINFORCE, and a learned baseline](05_reinforce_actor_critic.md).
