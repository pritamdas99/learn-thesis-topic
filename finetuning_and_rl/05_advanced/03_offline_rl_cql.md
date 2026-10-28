# Offline RL and the conservative Q-learning penalty

Companion code: [03_offline_rl_cql.py](03_offline_rl_cql.py).
Prerequisites: [Q-learning and SARSA](../03_reinforcement_learning/03_q_learning_sarsa.md),
[DQN and target networks](../03_reinforcement_learning/04_dqn.md), and
[softmax and cross-entropy](../01_foundations/01_python_and_math.md).
Read this explanation before running the code.

## 1. What problem are we solving now?

Every learner so far could **act**: try an action, see the result, and correct
a wrong estimate. Offline RL removes that ability. It learns from a fixed
dataset collected earlier by some other **behavior policy** (a person, an old
controller, a logged system), and it may not collect anything new.

Why is that hard? Actions absent from the dataset may receive arbitrary
neural-network values. A Bellman `max` or an actor can exploit those unsupported
high values, causing errors to reinforce themselves without any new environment
evidence to correct them. Online, the agent would try the overrated action, get
a poor reward, and fix the value. Offline, that correction never arrives.

The script's toy setup: two dataset states, three discrete actions each, and a
Q-table given directly as a trainable tensor:

```text
                action 0   action 1   action 2
state A   Q =     1          5          2        dataset action: 0, Bellman target 1.2
state B   Q =     2          4          1        dataset action: 0, Bellman target 1.8
```

In both states the dataset only shows action 0, yet action 1 has the largest
Q-value. Nothing in the data supports that 5 or that 4. This file is a discrete
**CQL regularizer lab**, not a complete offline RL trainer.

## 2. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $D$ | Fixed offline dataset | two rows here |
| $s$ | Dataset state | row index of `q` |
| $a_D$ | Action recorded in the dataset at $s$ | `actions = torch.tensor([0, 0])` |
| $Q(s,a)$ | Estimated action value | `q[s][a]` |
| $K$ | Number of discrete actions | 3 |
| $y$ | Detached Bellman target | `bellman_targets = [1.2, 1.8]` |
| $\alpha$ | Weight of the conservative penalty | `0.5` |
| $C(Q)$ | Conservative penalty | `conservative_penalty(q, actions)` |

This $\alpha$ is unrelated to SAC's temperature or Q-learning's step size.

## 3. Start with Bellman regression

For dataset action $a_D$ and a detached target $y$:

$$
\boxed{L_{Bellman}=\mathbb E_{(s,a_D)\sim D}[(Q_\theta(s,a_D)-y)^2]}
$$

In a real loop $y=r+\gamma\max_{a'}Q_{\text{target}}(s',a')$ (or a
policy-based variant); the script simply supplies constants.

This directly constrains observed actions. It need not constrain unobserved
ones well. For a state with Q values `[0,5]` but only action 0 observed,
ordinary regression on action 0 says little about whether the value 5 is
justified.

**Worked numbers for the script.** Predictions for the dataset actions are
$Q(A,0)=1$ and $Q(B,0)=2$:

$$L_{Bellman}=\tfrac12[(1-1.2)^2+(2-1.8)^2]=\tfrac12(0.04+0.04)=0.04.$$

Its gradient touches **only** column 0: $\partial L/\partial Q(s,0)=(Q-y)$
(the 2 from squaring cancels the $\tfrac12$ mean), giving $-0.2$ for state A
and $+0.2$ for state B. Columns 1 and 2 get exactly zero. That is the
problem in one line: plain Bellman fitting never touches the suspicious 5 and 4.
And in a real loop that unchecked 5 would feed into other states' targets
through $\max_{a'}$.

## 4. Add a conservative contrast

The simplified discrete penalty is:

$$
\boxed{C(Q)=\mathbb E_{s\sim D}\left[
\log\sum_a e^{Q(s,a)}-Q(s,a_D)\right],
\qquad L=L_{Bellman}+\alpha\, C(Q).}
$$

Log-sum-exp is a smooth approximation to the maximum:

$$
\max_aQ_a\le\log\sum_a e^{Q_a}\le\max_aQ_a+\log K.
$$

So $C$ is roughly "best value anywhere minus value of what the data did". It is
large when some action is rated far above the dataset action.

Subtracting the observed-action value makes the penalty favor dataset-supported
actions relative to other actions. It is not simply subtracting a constant
from every Q value: adding a constant to all action values cancels in $C$.
(Checked numerically: adding 10 to every entry leaves both per-state gaps unchanged.)

## 5. Worked example: compute the penalty row by row

**State A**, Q row `[1, 5, 2]`:

| Quantity | Action 0 | Action 1 | Action 2 | Total |
| --- | --- | --- | --- | --- |
| $e^{Q}$ | 2.7183 | 148.4132 | 7.3891 | 158.5205 |
| softmax $=e^Q/\text{total}$ | 0.01715 | 0.93624 | 0.04661 | 1 |

$\log158.5205=5.0659$. The gap is $5.0659-Q(A,0)=5.0659-1=4.0659$.
Bound check: $5\le5.0659\le5+\log3=6.0986$.

**State B**, Q row `[2, 4, 1]`:

| Quantity | Action 0 | Action 1 | Action 2 | Total |
| --- | --- | --- | --- | --- |
| $e^{Q}$ | 7.3891 | 54.5982 | 2.7183 | 64.7055 |
| softmax | 0.11420 | 0.84379 | 0.04201 | 1 |

$\log64.7055=4.1698$. The gap is $4.1698-2=2.1698$.

**Average over the two states:**

$$C=\tfrac12(4.0659+2.1698)=3.1179.$$

This is the `conservative penalty` the script prints (3.11787). The total loss
is $0.04+0.5(3.1179)=1.5989$ (not printed).

## 6. The derivative reveals the mechanism

For one state:

$$
\boxed{\frac{\partial C_s}{\partial Q_a}
=\operatorname{softmax}(Q)_a-\mathbf1[a=a_D]}
$$

The log-sum-exp part pushes every action down in proportion to its softmax
weight; the $-Q(s,a_D)$ part pushes the dataset action up by 1.

For `[0,5]` and observed action 0, softmax is approximately `[0.0067,0.9933]`.
Gradients are `[-0.9933,+0.9933]`. Gradient descent pushes the observed value
up and the unsupported high value down. Bellman regression simultaneously
anchors observed values toward reward-consistent targets.

### Full gradient of the script's loss

Because $C$ is averaged over 2 states and multiplied by $\alpha=0.5$, each
penalty gradient entry is $0.5\times\tfrac12\times(\text{softmax}-\text{one-hot})=0.25(\text{softmax}-\text{one-hot})$.

| Entry | Bellman part | Penalty part $0.25(p-\mathbf 1)$ | Total = printed `q.grad` |
| --- | --- | --- | --- |
| $Q(A,0)$ | $-0.2$ | $0.25(0.01715-1)=-0.24571$ | **-0.44571** |
| $Q(A,1)$ | 0 | $0.25(0.93624)=+0.23406$ | **+0.23406** |
| $Q(A,2)$ | 0 | $0.25(0.04661)=+0.01165$ | **+0.01165** |
| $Q(B,0)$ | $+0.2$ | $0.25(0.11420-1)=-0.22145$ | **-0.02145** |
| $Q(B,1)$ | 0 | $0.25(0.84379)=+0.21095$ | **+0.21095** |
| $Q(B,2)$ | 0 | $0.25(0.04201)=+0.01050$ | **+0.01050** |

How to read it under gradient descent ($Q\leftarrow Q-\eta\,\text{grad}$):

- **Positive** entries (all unsupported actions) are pushed **down**. The
  biggest push hits the most overrated action, because it has the largest softmax weight.
- State A's dataset action is pushed **up** by both terms: its value 1 is below
  its target 1.2, and the penalty also favors it.
- State B's dataset action shows a tug-of-war: Bellman wants $2\to1.8$
  (down), the penalty wants it up. The net gradient is tiny ($-0.021$).
  CQL does not blindly inflate observed actions; Bellman fitting still matters.

**Hypothetical illustration (not in the script).** One plain gradient step with
step size 1 would give rows `[1.446, 4.766, 1.988]` and `[2.021, 3.789, 0.989]`,
lowering the penalty from 3.118 to 2.695 and the total loss from 1.599 to 1.402.
The script itself calls only `backward()`; it never updates `q`.

Mathematically this discrete penalty equals negative log-softmax of the
observed action when Q values are treated as logits (check: $-\log0.01715=4.0659$,
the state-A gap). Its interpretation here is conservative value regularization,
combined with Bellman fitting, rather than an isolated supervised action classifier.

## 7. What changes when alpha changes?

Rerunning the same loss with different $\alpha$ (computed separately; the script fixes 0.5):

| $\alpha$ | grad $Q(A,0)$ | grad $Q(A,1)$ | grad $Q(B,0)$ | grad $Q(B,1)$ |
| --- | --- | --- | --- | --- |
| 0 | -0.2 | 0 | +0.2 | 0 |
| 0.5 | -0.4457 | +0.2341 | -0.0215 | +0.2109 |
| 2 | -1.1829 | +0.9362 | -0.6858 | +0.8438 |
| 5 | -2.6571 | +2.3406 | -2.0145 | +2.1095 |

With $\alpha=0$ there is no pressure on unsupported actions at all. As
$\alpha$ grows the conservative contribution dominates, and by $\alpha=2$ state
B's dataset action is pushed up even though Bellman wanted it lower. Too much
pessimism can suppress useful actions and prevent improvement beyond the
behavior dataset; too little lets extrapolated values leak into targets.
Choosing $\alpha$ is a real tuning decision, not a solved detail.

## 8. When the dataset contains more than one action

The penalty is an average over dataset rows. If the same state appears twice,
once with action 0 and once with action 1, the averaged gradient is

$$\operatorname{softmax}(Q)-\text{(empirical action frequencies)}.$$

For Q `[0,5]` and data frequencies $(0.5,0.5)$: $[0.0067-0.5,\ 0.9933-0.5]
=[-0.4933,+0.4933]$. Action 1's push-down shrank from 0.9933 to 0.4933 compared with the
single-action case, because the data now partly supports it. The penalty is at
its stationary point when the softmax of Q matches how often the dataset chose
each action, which is why it is said to keep the policy near the data's support.

## 9. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `chosen = q_values.gather(1, dataset_actions[:, None]).squeeze(1)` | Select $Q(s,a_D)$ for each row |
| `torch.logsumexp(q_values, dim=-1)` | Stable $\log\sum_a e^{Q(s,a)}$ per row |
| `(torch.logsumexp(q_values, dim=-1)-chosen).mean()` | $C(Q)$, averaged over states |
| `q = torch.tensor([[1., 5., 2.], [2., 4., 1.]], requires_grad=True)` | Q-table as a trainable tensor (no network) |
| `actions = torch.tensor([0, 0])` | Dataset actions $a_D$ |
| `bellman_targets = torch.tensor([1.2, 1.8])` | Hand-supplied $y$ (would be detached target-net outputs) |
| `bellman = (predicted-bellman_targets).square().mean()` | $L_{Bellman}=0.04$ |
| `loss = bellman+0.5*penalty` | $L=L_{Bellman}+\alpha C$ with $\alpha=0.5$ |
| `loss.backward()` | Fill `q.grad`; no optimizer step follows |

`torch.logsumexp` subtracts the row maximum internally, so it does not overflow
even for very large Q values, unlike a naive `q.exp().sum().log()`.

In `main`, `q` is directly a trainable tensor, targets are hand-supplied
constants, and alpha is 0.5. Backward exposes derivatives; there is no
optimizer loop, replay construction, or policy evaluation.

## 10. Run it

Run from the course folder:

```bash
python3 05_advanced/03_offline_rl_cql.py
```

On the CPU used to verify this lesson the output was (one line, wrapped here):

```text
conservative penalty: 3.1178650856018066 gradient: [[-0.4457131028175354,
0.23405984044075012, 0.011653153225779533], [-0.02145114541053772,
0.21094867587089539, 0.01050251629203558]]
```

These match the hand calculation in sections 5 and 6. The script has no
assertion; check by eye that the unsupported high-Q actions (column 1) have
positive gradients, so descent would lower them. The course's test suite
(`tests/`) separately checks that the penalty pushes down an unsupported
action. No GPU is needed.

## 11. What the script does not show

The CQL paper develops broader objectives (including continuous-action versions
that estimate the log-sum-exp by sampling actions) and theoretical results under
stated assumptions. This tiny regularizer demonstration does not establish a
universal lower bound or guarantee safe deployment. See the
[original CQL paper](https://arxiv.org/abs/2006.04779).

Compare against **behavior cloning**, which simply predicts dataset actions and
avoids Bellman extrapolation but may not improve beyond the behavior policy.
It is a useful baseline; never trust offline Q values alone, and evaluate a
learned policy carefully before deploying it.

## 12. Practice calculations

1. Q row `[0, 0, 0]`, dataset action 0. Penalty for this state?
   **$\log3-0=1.0986$; it is never below 0 and equals $\log K$ when all values tie.**
2. Q row `[3, 0]`, dataset action 0. Penalty and gradient?
   **$\log(e^3+1)-3=0.0486$; softmax $\approx[0.9526,0.0474]$, gradient
   $[-0.0474,+0.0474]$: almost nothing to fix.**
3. Add 100 to every entry of state A. Does its gap change? **No, still 4.0659.**
4. With $\alpha=1$ instead of 0.5, what is the total gradient on $Q(A,1)$?
   **$0.5\times0.93624=0.46812$ (penalty averaged over 2 states, Bellman part 0).**
5. With $\alpha=0$, what is the gradient on $Q(A,1)$? **Zero: Bellman regression
   alone never touches unsupported actions.**
6. Exercise: add more observed examples of the previously unsupported action.
   **Its average penalty gradient becomes softmax minus its new data frequency,
   so it is pushed down less (section 8).**

Next: [distillation and quantization](04_distillation_and_quantization.md).
