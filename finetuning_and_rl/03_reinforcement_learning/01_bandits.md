# Bandits: learn which button gives the most rewards

Companion code: [01_bandits.py](01_bandits.py). This chapter assumes no previous
reinforcement learning knowledge. Read sections 1–6 before the algorithm.

## 1. Imagine three mystery buttons

You can press one button each round. After pressing it, you receive either
one coin or no coin. Your goal is to collect as many coins as possible.

| Action you choose | Possible result | What you initially know |
| --- | --- | --- |
| Button A | 1 coin or 0 coins | Its chance of giving a coin is unknown |
| Button B | 1 coin or 0 coins | Its chance of giving a coin is unknown |
| Button C | 1 coin or 0 coins | Its chance of giving a coin is unknown |

An **agent** is the decision maker: you, or the program choosing buttons.
An **action** is a choice, such as pressing C. A **reward** is the numerical
feedback after an action: here, 1 for a coin and 0 for no coin.
The **environment** is the mechanism producing that reward.

A bandit is this simple repeated decision problem. An “arm” means an available
action, borrowing the name from a slot machine's lever. A three-armed bandit has
three choices. Here the reward probabilities stay fixed, and each pull is sampled
independently. There is no changing location or future state to plan around.
Bandits are a basic RL setting; MDPs later add state transitions and future consequences.

## 2. What does success probability mean?

Here, **success means receiving reward 1 on a particular press**. It does not mean
the agent has learned successfully or chosen the best button.

Imagine the environment secretly uses these probabilities:

| Button | Probability of reward 1 | Probability of reward 0 |
| --- | --- | --- |
| A | 0.2 = 20% | 0.8 = 80% |
| B | 0.5 = 50% | 0.5 = 50% |
| C | 0.8 = 80% | 0.2 = 20% |

The notation $P(R=1\mid A=C)=0.8$ reads:
“Given that we choose button C, the probability of reward 1 is 0.8.”
Here $P$ means probability, $R$ means reward, and the vertical bar means “given.”
In this equation $A$ names the action variable; its value $C$ identifies the button.

An 80% success probability does **not** mean you receive 0.8 of a coin on a press.
You receive either 1 or 0. It also does not guarantee exactly eight successes
in every ten presses. A possible ten-press record for C is:

```text
Press:   1  2  3  4  5  6  7  8  9 10
Reward:  1  1  0  1  1  1  0  1  1  1
```

This particular example has eight coins. Another ten presses might give six,
nine, or even zero. Across many presses, the success fraction tends toward 0.8.
The agent sees the reward record, not the hidden probability table.

## 3. Expected reward: why is it 0.8?

An **expected value** is a probability-weighted average. For one press of C:

$$E[R\mid C]=(1\text{ coin})(0.8)+(0\text{ coins})(0.2)=0.8\text{ coins}.$$

We multiply each possible reward by its probability, then add. Although 0.8 is
not a possible single outcome, it is the long-run average reward per press.

Similarly:

$$E[R\mid A]=1(0.2)+0(0.8)=0.2,$$
$$E[R\mid B]=1(0.5)+0(0.5)=0.5.$$

C has the highest expected reward. It can still fail on a particular press;
A can still succeed. “Best” refers to average performance, not guaranteed success.
For these 0/1 rewards, expected reward equals success probability. That equality
is specific to this reward scheme: if success paid 5 coins, C's expected reward
would be $5(0.8)+0(0.2)=4$ coins, while its success probability would remain 0.8.

## 4. Why is expected total reward 0.8T?

$T$ is the total number of rounds. $0.8T$ means $0.8\times T$.
If we choose C every round, each round contributes expected reward 0.8.

| Number of presses of C | Expected total reward |
| --- | --- |
| 1 | $0.8$ |
| 2 | $0.8+0.8=1.6$ |
| 10 | $10\times0.8=8$ |
| 100 | $100\times0.8=80$ |
| $T$ | $T\times0.8$ |

For example, the actual total over two presses can only be 0, 1, or 2 coins.
The expectation 1.6 is an average over hypothetical repetitions of that experiment.

Using notation, let $R_t$ be the reward at round $t$. The symbol $\sum$ means
“add all these terms”:

$$\sum_{t=1}^T R_t=R_1+R_2+\cdots+R_T.$$

Expected values add, giving:

$$E\left[\sum_{t=1}^T R_t\right]
=\sum_{t=1}^T E[R_t]
=\underbrace{0.8+0.8+\cdots+0.8}_{T\text{ terms}}
=0.8T.$$

This addition rule is called **linearity of expectation**. It does not require
independence. We do assume here that choosing C continues to give mean 0.8.
The result is a benchmark for always choosing the best button. An agent still
learning which button is best will generally earn a smaller expected total.

## 5. What is an action value?

An **action value** answers: “How much reward should I expect, on average,
when I choose this action?” For this bandit, define the true value as

$$q(a)=E[R\mid\text{choose action }a].$$

The lowercase $a$ is a placeholder for any button. Thus $q(A)=0.2$,
$q(B)=0.5$, and $q(C)=0.8$. The environment knows these values, but the agent does not.

We use uppercase $Q(a)$ for the agent's **estimate** of that true value.
The distinction is crucial:

| Quantity | Meaning | Available to the learner? |
| --- | --- | --- |
| $q(C)=0.8$ | Hidden true mean reward of C | No |
| $Q(C)=0.6$, for example | Mean of rewards observed so far from C | Yes |
| Latest reward, such as 0 | Result of one press | Yes |

One reward is an observation. A value estimate summarizes many observations.

## 6. Estimate a value by averaging observed rewards

Suppose the first five times we choose C produce rewards `[1, 0, 1, 1, 0]`.
These are five pulls **of C**; presses of other buttons do not enter C's average.

| Number of C pulls $n$ | New reward | Total coins from C | Estimate $Q_n(C)$ |
| --- | --- | --- | --- |
| 1 | 1 | 1 | $1/1=1.00$ |
| 2 | 0 | 1 | $1/2=0.50$ |
| 3 | 1 | 2 | $2/3\approx0.667$ |
| 4 | 1 | 3 | $3/4=0.75$ |
| 5 | 0 | 3 | $3/5=0.60$ |

The estimate 0.60 does not change the true probability 0.8. It reflects limited,
noisy evidence. More samples generally improve reliability, but an estimate
does not necessarily move closer to the truth after every individual sample.

The sample-average formula is:

$$Q_n(a)=\frac{R_1^{(a)}+R_2^{(a)}+\cdots+R_n^{(a)}}{n}
=\frac1n\sum_{i=1}^n R_i^{(a)}.$$

Here $R_i^{(a)}$ means the reward from the $i$-th time action $a$ was selected;
it is an index label, not a power. $n$ counts observations of that particular
action. An untried action has no sample average yet. Initializing its estimate
to zero is a programming convention, not evidence that it never pays.

## 7. Update the average without storing every reward

After each new reward, the code uses:

$$\boxed{Q_{new}=Q_{old}+\frac1n(R_{new}-Q_{old})}$$

Read this as **old estimate + step size × prediction error**:

| Part | Meaning |
| --- | --- |
| $Q_{old}$ | What we expected before this observation |
| $R_{new}$ | What we actually received |
| $R_{new}-Q_{old}$ | How much the observation differs from our estimate |
| $1/n$ | How strongly this observation changes the estimate |

For C's third observation in the table, $Q_{old}=0.5$, $R_{new}=1$, and $n=3$:

$$Q_{new}=0.5+\frac13(1-0.5)=0.5+0.1667\approx0.6667.$$

For its fifth observation, $Q_{old}=0.75$, $R_{new}=0$, and $n=5$:

$$Q_{new}=0.75+\frac15(0-0.75)=0.75-0.15=0.60.$$

A reward above the estimate moves it upward; one below moves it downward.
Later observations have smaller individual influence because more evidence has
already accumulated. This is ordinary averaging, not a separate approximation.
To derive it, the previous $n-1$ observations have total $(n-1)Q_{old}$:

$$Q_{new}=\frac{(n-1)Q_{old}+R_{new}}n
=Q_{old}+\frac{R_{new}-Q_{old}}n.$$

You can run this tiny example independently:

```python
estimate = 0.0
for count, reward in enumerate([1, 0, 1, 1, 0], start=1):
    estimate += (reward - estimate) / count
    print(count, round(estimate, 3))
# 1 1.0
# 2 0.5
# 3 0.667
# 4 0.75
# 5 0.6
```

## 8. Why must the agent explore?

Imagine the agent tries each button once and observes:

| Button | Observed reward | Current estimate |
| --- | --- | --- |
| A | 1 | 1.0 |
| B | 0 | 0.0 |
| C | 0 | 0.0 |

A happened to get lucky; C happened to get unlucky. Choosing only the highest
current estimate can neglect C, even though C is truly best.

**Exploitation** means choosing the action with the highest current estimate.
**Exploration** means trying actions to collect more information.
An **epsilon-greedy** rule balances them. Epsilon, written $\epsilon$, is a
chosen exploration probability; it is not a button's success probability.

With $\epsilon=0.1$:

```text
Each round:
    10% chance -> explore -> choose A, B, or C uniformly
    90% chance -> exploit -> choose the highest current estimate
Then observe one reward and update only the chosen button's estimate.
```

Suppose C now has the highest estimate. It is chosen through two routes:
exploitation, or exploration that happens to select C. Therefore:

$$P(\text{choose C})=0.9+0.1\times\frac13\approx0.9333.$$

A and B are each chosen with probability $0.1/3\approx0.0333$.
Do not confuse $P(\text{choose C})\approx0.9333$ with
$P(\text{reward 1}\mid C)=0.8$: one describes the agent's choice,
the other describes the environment's response.

When C is the selected greedy action, expected reward per round is:

$$0.0333(0.2)+0.0333(0.5)+0.9333(0.8)\approx0.77.$$

Equivalently: 90% of rounds use C, and exploratory rounds choose uniformly:

$$0.9(0.8)+0.1\left(\frac{0.2+0.5+0.8}{3}\right)=0.72+0.05=0.77.$$

Persistent exploration costs some immediate reward. Before C is identified as
best, the expected reward may be lower still. The actual observed average
fluctuates around its expectation.

## 9. Connect each code line to the story

The companion script uses indices `0, 1, 2` for A, B, C:

```python
true_means = [0.2, 0.5, 0.8]  # environment's hidden probabilities
counts = [0, 0, 0]            # number of presses of each button
estimates = [0.0, 0.0, 0.0]  # learner's current Q estimates
```

`rng.random()` produces a random number in `[0, 1)`. The expression
`rng.random() < 0.1` is true with probability 0.1, triggering exploration.
`rng.randrange(3)` chooses a random button index. Otherwise, `max` selects the
index with the largest estimate. Initial equal estimates are broken in favor
of the first index in this implementation.

The environment generates a reward using:

```python
reward = float(rng.random() < true_means[action])
```

For C, a random number below 0.8 gives reward 1. For example, 0.37 gives 1,
while 0.91 gives 0. The agent's action-selection rule never reads `true_means`.
The simulator needs them to produce feedback; the learner sees only that feedback.

Then the learner updates:

```python
counts[action] += 1  # n must include this new observation
estimates[action] += (reward - estimates[action]) / counts[action]
```

Only the chosen button's count and estimate change. `total_reward / 2000` is
the observed overall reward per round, mixing all three buttons. It is different
from an individual button's estimate.

Run from the course folder:

```bash
python3 03_reinforcement_learning/01_bandits.py
```

Expect C to receive most selections. Its estimate should be near 0.8 after many
pulls, and the overall average near 0.77 once the agent mostly identifies C.
Neither number is guaranteed in a short run.

## 10. Check your understanding

1. A button pays 1 with probability 0.3 and otherwise 0. What is its expected
   reward? **0.3**, though one press gives only 0 or 1.
2. Press it 20 times. What total reward do you expect? **6**, not a guaranteed 6.
3. Observe rewards `[0, 1, 1, 0]`. What is your estimate? **2/4 = 0.5**.
4. The next reward is 1. Update without storing the history:
   **$0.5+(1-0.5)/5=0.6$**.
5. Does that make the true success probability 0.6? **No. It changes our estimate.**
6. Why explore? **Early estimates can rank the buttons incorrectly.**

After these examples make sense, compare epsilon values 0, 0.1, and 0.5 across
several seeds. More exploration gathers information but also spends more rounds
on weaker actions. Next: [MDPs and future consequences](02_mdp_value_iteration.md).
