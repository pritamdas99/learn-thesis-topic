# Reward models: learning scores from pairwise preferences

Companion code: [01_reward_model.py](01_reward_model.py).
Prerequisites: [sigmoid, logs, and derivatives](../01_foundations/01_python_and_math.md),
[cross-entropy and gradient descent](../01_foundations/03_neural_network_training.md),
and [supervised fine-tuning](../02_finetuning/04_huggingface_sft.md).
Read this explanation before running the code. No transformer is needed here.

## 1. What problem are we solving now?

Supervised fine-tuning (SFT) copies demonstrations: “for this prompt, write this
answer.” Often we cannot easily write the single best answer, but a person can
look at **two** answers and say which one is better. People can usually make
that choice more consistently than they can assign a precise absolute quality
score such as “7.3 out of 10.”

A preference dataset therefore stores triples:

```text
prompt x:          "What is 12 * 3?"
chosen   y_w:      "12 * 3 = 36."                         (preferred, the "winner")
rejected y_l:      "Multiplication is repeated addition, so the answer is 38."
```

The goal of a **reward model** $r_\phi(x,y)$ is to output one number per
(prompt, response) so that preferred responses tend to get larger numbers.
Later lessons use that number as the reward an RL algorithm tries to increase.

The scalar is a **learned proxy for preference**, not a direct measurement of
truth or helpfulness. Everything in this lesson is about fitting that proxy.

## 2. What tiny dataset does the script use?

A real reward model reads prompt and response tokens. The script replaces each
response with three hand-written numbers so the mathematics is visible:

| Pair | Chosen `[correct, clear, verbose]` | Rejected `[correct, clear, verbose]` | Chosen minus rejected |
| --- | --- | --- | --- |
| 1 | `[1.0, 1.0, 0.3]` | `[0.0, 1.0, 0.8]` | `[1.0, 0.0, -0.5]` |
| 2 | `[1.0, 0.8, 0.2]` | `[0.2, 0.8, 0.7]` | `[0.8, 0.0, -0.5]` |
| 3 | `[0.8, 1.0, 0.5]` | `[0.1, 0.5, 0.5]` | `[0.7, 0.5, 0.0]` |

These are synthetic labels, not a practical way to measure quality automatically.
Look at the last column before reading further:

- Correctness is higher for the chosen answer in **every** pair.
- Clarity differs only in pair 3.
- Verbosity is **lower** for the chosen answer in pairs 1 and 2 and equal in pair 3.

So in this data, “shorter” is perfectly correlated with “preferred.” Nobody told
the model that brevity matters; the data happens to contain that pattern. Keep
this in mind when we look at the learned weights in section 9.

## 3. Notation

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $x$ | Prompt | not represented in the script |
| $y_w$, $y_l$ | Preferred (winner) and rejected (loser) response | rows of `chosen`, `rejected` |
| $f(y)$ | Feature vector describing a response | `[1., 1., 0.3]` |
| $\phi$ | Trainable reward-model parameters | `model.weight`, shape `[1,3]` |
| $r_\phi(x,y)$ | Scalar reward score | `model(chosen)`, shape `[3,1]` |
| $r_w$, $r_l$ | Scores of winner and loser | `chosen_scores`, `rejected_scores` |
| $\Delta$ | Score difference $r_w-r_l$ | `chosen_scores-rejected_scores` |
| $\sigma(u)$ | Sigmoid $1/(1+e^{-u})$ | inside `logsigmoid` |
| $N$ | Number of pairs | 3 |
| $L_{RM}$ | Reward-model loss averaged over pairs | `preference_loss(...)` |

In the script the reward is linear: $r_\phi(y)=\phi^\top f(y)$, a weighted sum of
the three features. In a real reward model, those quality features are not
provided by an oracle; a network must infer useful representations from prompt
and response tokens.

## 4. How do we turn two scores into a probability?

The **Bradley–Terry model** assumes each response has a positive “strength”
$e^{r}$. The chance the winner beats the loser is its share of the total strength:

$$
P(y_w\succ y_l\mid x)
=\frac{e^{r_w}}{e^{r_w}+e^{r_l}}
=\frac1{1+e^{-(r_w-r_l)}}.
$$

(Divide numerator and denominator by $e^{r_w}$ to get the second form.) Hence

$$\boxed{P(y_w\succ y_l\mid x)=\sigma(r_w-r_l)=\sigma(\Delta)}$$

The symbol $\succ$ reads “is preferred to.” The sigmoid converts any score
difference into a number between zero and one:

| $\Delta=r_w-r_l$ | $\sigma(\Delta)$ | Interpretation |
| --- | --- | --- |
| $-2$ | 0.1192 | Model thinks the labelled winner is probably worse |
| $-1$ | 0.2689 | |
| 0 | 0.5000 | Coin flip: no preference |
| 1 | 0.7311 | |
| 2 | 0.8808 | Fairly confident the winner is better |
| 3 | 0.9526 | |

Only the **difference** of scores appears. That fact matters in section 7.

## 5. The loss: negative log-likelihood of the observed choices

We want large probability for every human choice in the data. Taking the
negative log and averaging over pairs gives

$$\boxed{L_{RM}=-\frac1N\sum_{i=1}^N\log\sigma(r_{w,i}-r_{l,i})}$$

This is ordinary binary [cross-entropy](../01_foundations/03_neural_network_training.md)
where the label is always “the first response won” and the logit is $\Delta$.

| $\Delta$ | $\sigma(\Delta)$ | Loss $-\log\sigma(\Delta)$ |
| --- | --- | --- |
| $-1$ | 0.2689 | 1.3133 |
| 0 | 0.5000 | $\log 2 = 0.6931$ |
| 0.5 | 0.6225 | 0.4741 |
| 1 | 0.7311 | 0.3133 |
| 2 | 0.8808 | 0.1269 |
| 3 | 0.9526 | 0.0486 |

With reward difference 0, preference probability is 0.5 and loss is $\log2$.
With difference 2, probability is about 0.881 and loss about 0.127. The loss
favors assigning the preferred response a larger score, and a wrongly ordered
pair (negative $\Delta$) is penalized heavily.

The code uses `logsigmoid`, which computes $\log\sigma(\Delta)$ stably without
first producing a possibly tiny sigmoid output and then taking its log.

## 6. Worked example: score all three pairs by hand

Pick illustrative weights $\phi=[1,\,1,\,-1]$ (reward correctness and clarity,
penalize verbosity). These are **not** the script's weights; they are chosen
to keep the arithmetic easy.

**Step 1: scores.** For chosen pair 1: $1(1)+1(1)-1(0.3)=1.7$. For rejected
pair 1: $1(0)+1(1)-1(0.8)=0.2$.

| Pair | $r_w$ | $r_l$ | $\Delta$ | $\sigma(\Delta)$ | Loss |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.7 | 0.2 | 1.5 | 0.8176 | 0.2014 |
| 2 | 1.6 | 0.3 | 1.3 | 0.7858 | 0.2410 |
| 3 | 1.3 | 0.1 | 1.2 | 0.7685 | 0.2633 |

**Step 2: average.** $L_{RM}=(0.2014+0.2410+0.2633)/3=0.2352$.

You can also compute $\Delta$ directly from the difference column of section 2:
pair 3 gives $1(0.7)+1(0.5)-1(0)=1.2$. A linear reward only ever sees
$f(y_w)-f(y_l)$.

## 7. Which way do the gradients push?

Let $\Delta=r_w-r_l$. Using $\frac{d}{du}\log\sigma(u)=1-\sigma(u)$:

$$\boxed{\frac{d[-\log\sigma(\Delta)]}{d\Delta}=\sigma(\Delta)-1}$$

This is always negative, so gradient descent **increases** the reward difference.
By the chain rule, the gradient with respect to the winner's score is
$\sigma(\Delta)-1$ and with respect to the loser's score is $1-\sigma(\Delta)$.
At $\Delta=0$, the winner-score gradient is $-0.5$ and loser-score gradient
is $+0.5$: push the winner up and the loser down equally.

| $\Delta$ | $\sigma(\Delta)-1$ |
| --- | --- |
| $-2$ | $-0.8808$ |
| 0 | $-0.5$ |
| 1 | $-0.2689$ |
| 2 | $-0.1192$ |
| 3 | $-0.0474$ |

If the model already predicts the preference confidently, the gradient
magnitude shrinks. Badly ranked pairs get the strongest push.

For the linear model, $\partial\Delta/\partial\phi=f(y_w)-f(y_l)$, so

$$
\nabla_\phi L_{RM}=\frac1N\sum_i[\sigma(\Delta_i)-1]\,[f(y_{w,i})-f(y_{l,i})].
$$

With $\phi=[1,1,-1]$, the factors $\sigma(\Delta_i)-1$ are
$-0.1824,\ -0.2142,\ -0.2315$. The correctness component is

$$
\frac{(-0.1824)(1.0)+(-0.2142)(0.8)+(-0.2315)(0.7)}{3}
=\frac{-0.5158}{3}=-0.1719.
$$

The full gradient is about $[-0.1719,\,-0.0386,\,+0.0661]$. Descent subtracts it,
so the correctness and clarity weights rise and the verbosity weight becomes
more negative, exactly following the patterns in the difference column.

## 8. Absolute reward is not identified by these labels

Add 5 to every score in the worked example: pair 1 becomes $6.7$ versus $5.2$,
still $\Delta=1.5$. Adding the same prompt-dependent constant to both response
scores leaves their difference unchanged, so the probability and loss are
unchanged too:

$$
\sigma\big((r_w+c)-(r_l+c)\big)=\sigma(r_w-r_l).
$$

Therefore pairwise comparisons cannot determine a unique absolute reward zero.
Only score **differences** are identified. This explains why the teaching linear
layer is `nn.Linear(3, 1, bias=False)`: a shared bias would cancel from every
pair anyway. A practical consequence: “this response scored 4.4” means nothing
alone; only comparisons with other responses to the same prompt do.

On perfectly separable data, unregularized logistic preference fitting can
continue increasing score margins without reaching finite optimal weights.
The script's data is separable (every chosen response is more correct), so
longer training keeps growing the weights. In a scratch run continuing the same
setup to 1000 Adam steps (not something the script does), the weights grew to
roughly `[5.19, 5.61, -4.69]` and the loss fell to about 0.0012. A finite number
of steps, regularization, and held-out evaluation matter. Correct pair ranking
alone does not establish well-calibrated reward magnitudes.

## 9. What does the script actually learn?

With `torch.manual_seed(7)` the initial weights are about
`[0.040, -0.348, 0.184]`. Every initial $\Delta$ is slightly negative
(`-0.052, -0.060, -0.146`), so the starting pair accuracy is 0 and the loss is
0.7371, a little above $\log2$.

After 100 Adam steps with learning rate 0.05:

| Quantity | Value |
| --- | --- |
| Learned weights `[correct, clear, verbose]` | `[2.6420, 2.4584, -2.3417]` |
| Pair margins $\Delta$ | `3.813, 3.284, 3.079` |
| Pair probabilities $\sigma(\Delta)$ | `0.978, 0.964, 0.956` |
| Final loss | about 0.0345 |
| Training pair accuracy | 1.0 |

Two observations deserve care:

1. **Clarity got a large weight from a single pair.** Only pair 3 differs in
   clarity, yet the weight reached 2.46. Three pairs are far too few to estimate
   how much clarity really matters.
2. **Verbosity got a large negative weight.** The model learned “shorter is
   better” because the rejected answers in pairs 1 and 2 happen to be longer.
   That is a **shortcut**: a feature correlated with preference in this data,
   not necessarily a cause of it.

Test the shortcut with held-out pairs (computed with the learned weights above;
they are not in the script):

| Held-out chosen | Held-out rejected | $\Delta$ | $\sigma(\Delta)$ | Verdict |
| --- | --- | --- | --- | --- |
| `[1.0, 0.8, 0.9]` long, correct | `[0.0, 0.8, 0.2]` short, wrong | $2.642(1)-2.342(0.7)=1.003$ | 0.732 | Correct ranking, reduced confidence |
| `[0.6, 0.8, 1.0]` long, mostly correct | `[0.1, 0.8, 0.2]` short, wrong | $2.642(0.5)-2.342(0.8)=-0.552$ | 0.365 | **Wrong** ranking |

Perfect training accuracy did not guarantee the intended preference.

## 10. Why this matters for RLHF

The next lessons optimize a policy **against** the reward model. A policy is
very good at finding inputs where a learned score is high for unintended
reasons, such as length, formatting, or flattering phrasing. This is called
**reward hacking** or reward over-optimization. If our toy reward model were used
as a target, a policy could gain score just by becoming terse.

Held-out prompts, disagreement analysis, and adversarial evaluation are
essential before using such a score as an optimization target. Evaluate true
task outcomes separately from learned reward.

Caveat on scope: the script shows the Bradley–Terry loss on three fixed feature
pairs. It does not show a transformer reward head, prompt handling, label noise,
annotator disagreement, ties, or reward-hacking during policy training. Claims
about those settings above are general background, not things the script tests.

## 11. Map the explanation to the script

| Code | Meaning |
| --- | --- |
| `chosen = torch.tensor([[1., 1., 0.3], ...])` | Feature rows $f(y_w)$ for the three winners |
| `rejected = torch.tensor([[0., 1., 0.8], ...])` | Feature rows $f(y_l)$ for the three losers |
| `model = nn.Linear(3, 1, bias=False)` | Linear reward $r_\phi(y)=\phi^\top f(y)$; bias omitted because it cancels |
| `model(chosen)`, `model(rejected)` | Scores $r_w$, $r_l$, each of shape `[3,1]` |
| `chosen_scores-rejected_scores` | Score differences $\Delta_i$ |
| `-nn.functional.logsigmoid(...).mean()` | $L_{RM}=-\frac1N\sum\log\sigma(\Delta_i)$, computed stably |
| `torch.optim.Adam(model.parameters(), lr=0.05)` | Optimizer updating the shared scoring weights |
| `for _ in range(100):` | Fixed number of steps, no convergence test |
| `(model(chosen)>model(rejected)).float().mean()` | Fraction of training pairs ranked correctly |
| `model.weight.detach().tolist()` | Learned $\phi$ = `[correct, clear, verbose]` weights |

Adam updates the same scoring function using both sides of every pair. The
printed pair accuracy measures only these three training comparisons.

## 12. Run and investigate

Run from the course folder (PyTorch must be installed; see the course README):

```bash
python3 04_alignment/01_reward_model.py
```

Output from the provided seed:

```text
training pair accuracy: 1.0
learned feature weights: [[2.6419734954833984, 2.4584295749664307, -2.3417065143585205]]
```

The script has no command-line flags and no assertion; it only prints. Accuracy
1.0 means all three training pairs are ordered correctly. It does not mean the
reward is correct on new responses, and the exact weight values depend on the
seed, learning rate, and step count.

Experiments:

1. Add examples where long **correct** answers are chosen over short incorrect
   ones. Does the verbosity weight move toward zero?
2. Create a held-out pair that breaks a training correlation, such as a short
   correct answer versus a long incorrect one (or the reverse, as in section 9).
   Observe whether the model learned the intended quality or a shortcut.
3. Increase the step count and watch the weights keep growing (section 8).

## 13. Practice calculations

1. Scores $r_w=2.0$, $r_l=0.5$. What is $P(y_w\succ y_l)$ and the loss?
   **$\Delta=1.5$, $\sigma(1.5)=0.8176$, loss $=0.2014$.**
2. What is $\partial L/\partial\Delta$ for that pair? **$0.8176-1=-0.1824$.**
3. Scores $r_w=0.5$, $r_l=1.0$. Probability and loss?
   **$\sigma(-0.5)=0.3775$, loss $0.9741$. The pair is ranked wrongly.**
4. Add 10 to both scores in question 1. What changes? **Nothing; $\Delta$ is still 1.5.**
5. With weights $[1,1,-1]$, what is $\Delta$ for pair 2 of the script?
   **$1(0.8)+1(0)-1(-0.5)=1.3$.**
6. Why does a bias term not help? **It adds the same constant to both scores and cancels in $\Delta$.**

Next: [KL-regularized reward optimization](02_rlhf_kl_penalty.md).
