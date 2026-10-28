# Evaluation and uncertainty: how much should you trust an improvement?

Companion code: [05_evaluation_and_uncertainty.py](05_evaluation_and_uncertainty.py).
Prerequisites: averages from [bandits](../03_reinforcement_learning/01_bandits.md),
sampling and variance from [Monte Carlo](../03_reinforcement_learning/07_monte_carlo.md),
and the idea of comparing several seeds from
[Q-learning and SARSA](../03_reinforcement_learning/03_q_learning_sarsa.md).
The script needs only the Python standard library.
Read this explanation before running the code.

## 1. What problem are we solving now?

You fine-tuned a model and it scores higher than the baseline. Is it actually
better, or did it get lucky on this particular set of test tasks?

A higher score is an observation, not yet a robust conclusion. Two models
evaluated on a small sample can differ by chance. We need three things:

1. A clear **metric** measuring the outcome we care about (task success,
   not just training loss).
2. **Held-out data** that was not used for training or model selection.
3. An **uncertainty estimate** that matches how the examples were sampled.

This script estimates uncertainty in a **paired mean improvement**: both
methods are tested on the same tasks, and we ask how much the average
per-task improvement could vary if we had drawn a different set of tasks.

## 2. The tiny example in the script

Ten evaluation tasks, each scored 1 (solved) or 0 (failed):

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline $a_i$ (`before`) | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| Candidate $b_i$ (`after`) | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 |
| Difference $d_i=b_i-a_i$ | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |

The baseline solves 4/10, the candidate 7/10. Three tasks improve by one; seven
do not change; none get worse. These are synthetic numbers. Ten tasks are far
too few for a serious scientific claim, as the script's own comment says.

| Symbol | Meaning | Example / Code |
| --- | --- | --- |
| $N$ | Number of paired evaluation tasks | 10, `len(differences)` |
| $a_i, b_i$ | Baseline and candidate score on task $i$ | `baseline`, `candidate` |
| $d_i$ | Paired difference $b_i-a_i$ | `differences` |
| $\hat\Delta$ | Estimated mean improvement | `statistics.mean(differences)` |
| $s_d$ | Sample standard deviation of the $d_i$ | not computed by the script |
| $\text{SE}$ | Standard error of $\hat\Delta$ | not computed by the script |
| $R$ | Number of bootstrap replicates | `samples=2000` |
| $\hat\Delta^{*}_r$ | Mean of bootstrap replicate $r$ | entries of `means` |

## 3. The point estimate

$$
\boxed{\hat\Delta=\frac1N\sum_{i=1}^N d_i}
\qquad\Rightarrow\qquad
\hat\Delta=\frac{1+0+0+0+1+0+0+1+0+0}{10}=0.3.
$$

This equals $0.7-0.4$, the difference of the two accuracies. On its own it says
nothing about how stable it is.

## 4. A classical standard error and t-interval, computed by hand

The script does not compute this, but it gives a useful reference. For
independent task differences, the standard error of the mean is

$$
\boxed{\text{SE}=\frac{s_d}{\sqrt N},\qquad
s_d^2=\frac{1}{N-1}\sum_{i=1}^N(d_i-\hat\Delta)^2.}
$$

Step by step:

| Quantity | Arithmetic | Value |
| --- | --- | --- |
| Deviations | three of $1-0.3=0.7$, seven of $0-0.3=-0.3$ | |
| Sum of squares | $3(0.49)+7(0.09)=1.47+0.63$ | 2.10 |
| $s_d^2$ | $2.10/9$ | 0.2333 |
| $s_d$ | $\sqrt{0.2333}$ | 0.4830 |
| SE | $0.4830/\sqrt{10}=0.4830/3.1623$ | **0.1528** |

A 95% t-interval uses $t_{0.975,\,N-1}=t_{0.975,9}\approx2.262$:

$$
0.3\pm2.262(0.1528)=0.3\pm0.3456=[-0.046,\ 0.646].
$$

The t-interval assumes the mean is roughly normally distributed, which is
shaky for ten 0/1 differences, and it extends below 0 even though no task got
worse. The bootstrap below provides another way to approximate the
estimator's sampling variation without assuming a particular distribution for
the task scores.

## 5. Why does pairing matter mathematically?

For two scores on the same task:

$$
\boxed{\operatorname{Var}(b-a)=\operatorname{Var}(b)+\operatorname{Var}(a)
-2\operatorname{Cov}(a,b).}
$$

Task difficulty often causes positive covariance: both methods do well on easy
tasks and poorly on hard ones. Pairing preserves this relationship and can make
comparison less noisy. With the example data (sample variances, divisor $N-1$):

| Quantity | Value |
| --- | --- |
| $\operatorname{Var}(a)$ | 0.2667 |
| $\operatorname{Var}(b)$ | 0.2333 |
| $\operatorname{Cov}(a,b)$ | 0.1333 |
| $\operatorname{Var}(d)=0.2333+0.2667-2(0.1333)$ | 0.2333 |
| Paired SE $=\sqrt{0.2333/10}$ | **0.153** |
| Unpaired SE $=\sqrt{0.2667/10+0.2333/10}$ | **0.224** |

Ignoring the pairing would inflate the standard error by almost half here.
Independently resampling baseline and candidate scores would lose the
correspondence and estimate a different uncertainty structure. As a separate
check (not in the script), resampling the two lists independently with the
same 2000 replicates gave an interval of about `[-0.1, 0.7]`, wider than the
paired result below.

The script resamples the differences directly, which is equivalent to drawing
the same task indices for both methods.

## 6. Bootstrap step by step

Treat the empirical task distribution as an approximation to the population
of tasks. For each of $R=2000$ repetitions:

1. Draw $N$ observed differences **with replacement**.
2. Compute their mean $\hat\Delta^*_r$.
3. Store that mean.

Sort these means and take approximately the 2.5th and 97.5th percentiles as a
**percentile-bootstrap interval**:

$$
\boxed{\text{CI}_{95\%}\approx\big[\hat\Delta^*_{(\lfloor0.025R\rfloor)},\ \hat\Delta^*_{(\lfloor0.975R\rfloor)}\big]}
$$

where the subscript is a 0-based position in the sorted list. With $R=2000$,
that is `means[50]` and `means[1950]`.

"With replacement" means one task can be drawn several times while another is
omitted in a replicate. This simulates sampling variability around the
observed dataset. For example:

| Replicate draws (differences) | Ones drawn | Mean |
| --- | --- | --- |
| 1,1,1,1,1,0,0,0,0,0 | 5 | 0.5 |
| 1,0,0,0,0,0,0,0,0,0 | 1 | 0.1 |
| 0,0,0,0,0,0,0,0,0,0 | 0 | 0.0 |

Because each draw is a 1 with probability $3/10$, the number of ones in a
replicate follows a Binomial(10, 0.3) distribution, so the bootstrap means can
only be 0.0, 0.1, …, 1.0. We can compare the script's actual 2000 replicates
(seed 7) with the exact probabilities:

| Replicate mean | Exact probability | Exact cumulative | Count in script's run |
| --- | --- | --- | --- |
| 0.0 | 0.0282 | 0.0282 | 67 |
| 0.1 | 0.1211 | 0.1493 | 252 |
| 0.2 | 0.2335 | 0.3828 | 467 |
| 0.3 | 0.2668 | 0.6496 | 544 |
| 0.4 | 0.2001 | 0.8497 | 382 |
| 0.5 | 0.1029 | 0.9527 | 194 |
| 0.6 | 0.0368 | 0.9894 | 78 |
| 0.7 | 0.0090 | 0.9984 | 13 |
| 0.8 | 0.0014 | 0.9999 | 3 |

Reading the endpoints: the first 67 sorted values (positions 0–66) are 0, so
position 50 is **0**. The values up to 0.5 fill positions 0–1905, so position
1950 is **0.6**. The interval is `[0, 0.6]`. The exact cumulative column agrees:
2.5% falls inside the 0.0 bucket (cumulative 0.0282) and 97.5% falls inside the
0.6 bucket (from 0.9527 to 0.9894).

The interval is discrete and approximate because the dataset is tiny. The
standard deviation of the 2000 bootstrap means was 0.145, close to the plug-in
value $\sqrt{0.3\cdot0.7/10}=0.145$ and a bit smaller than the t-based SE of
0.153 (the bootstrap uses divisor $N$ rather than $N-1$).

## 7. Interpret the interval carefully

A frequentist 95% confidence procedure aims to cover the population quantity
in about 95% of repeated studies under its assumptions. It is not a statement
that this fixed computed interval contains the fixed truth with 95% posterior
probability. Bootstrap coverage can be poor with small samples, severe skew,
or a mismatched sampling model.

Here the interval's lower end touches 0. That does not prove "no effect", and
it does not prove a reliable effect either. A reasonable summary is: "On 10
tasks, the candidate improved 3 and worsened none; the paired bootstrap
interval for the mean improvement is roughly 0 to 0.6, so more tasks are
needed for a precise estimate."

## 8. What is the unit of resampling?

If examples share users/documents, individual rows are not independent units.
Resample whole users/documents (a **cluster bootstrap**), or use an appropriate
hierarchical procedure. Otherwise the interval will look narrower than it
should.

For RL, multiple evaluation episodes of one trained agent do not replace
multiple independent training seeds: training randomness is another source of
variation. Report the return distribution across held-out evaluation seeds; a
single high-return episode does not demonstrate reliable learning. Match the
resampling design to the claim you want to make.

**Worked seed example (hypothetical numbers, not from the script).** Five
independent training seeds give final mean returns
$[0.82, 0.75, 0.91, 0.68, 0.79]$:

| Step | Arithmetic | Value |
| --- | --- | --- |
| Mean | $3.95/5$ | 0.79 |
| Deviations | $0.03,-0.04,0.12,-0.11,0.00$ | |
| Sum of squares | $0.0009+0.0016+0.0144+0.0121+0$ | 0.0290 |
| $s$ | $\sqrt{0.0290/4}=\sqrt{0.00725}$ | 0.0851 |
| SE | $0.0851/\sqrt5$ | 0.0381 |
| 95% t-interval, $t_{0.975,4}\approx2.776$ | $0.79\pm2.776(0.0381)=0.79\pm0.106$ | $[0.684,\ 0.896]$ |

With only five seeds the $t$ multiplier is large (2.776 rather than about 1.96),
which is the honest price of few independent runs.

## 9. Code mapping and practical limitations

| Code | Meaning |
| --- | --- |
| `if not baseline or len(baseline) != len(candidate):` | Refuse empty or unpaired inputs with `ValueError` |
| `rng = random.Random(seed)` | Private random generator, `seed=7` by default |
| `differences = [b-a for a, b in zip(baseline, candidate)]` | Paired differences $d_i$ |
| `rng.choices(differences, k=len(differences))` | One bootstrap replicate: $N$ draws with replacement |
| `statistics.mean(...)` | Replicate mean $\hat\Delta^*_r$ |
| `means = sorted(... for _ in range(samples))` | All $R=2000$ replicate means, sorted |
| `means[int(0.025*samples)], means[int(0.975*samples)]` | Percentile endpoints, positions 50 and 1950 |
| `return statistics.mean(differences), (...)` | Point estimate $\hat\Delta$ and interval |
| `before = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]` | Baseline scores $a_i$ |
| `after =  [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]` | Candidate scores $b_i$ |

The seed fixes the bootstrap simulation, not uncertainty in the original
experiment. Increasing bootstrap repetitions makes the numerical percentiles
more stable; it does not create more independent evaluation tasks. Increasing
genuine task coverage is different from resampling the same ten tasks.
(Checked separately: `samples=20000` and `seed=1` both still give `(0, 0.6)`.)

## 10. Run and check

Run from the course folder:

```bash
python3 05_advanced/05_evaluation_and_uncertainty.py
```

The script has no command-line flags. Observed output:

```text
mean improvement: 0.3 approximate 95% bootstrap interval: (0, 0.6)
```

Expect mean improvement 0.3 and a wide interval. The lower endpoint prints as
`0` rather than `0.0` because `statistics.mean` of a list of integer zeros
returns the integer 0. There is no assertion in the script; the only built-in
check is the `ValueError` for empty or mismatched score lists.

Exercises:

1. Use identical candidate and baseline scores. Every difference and every
   bootstrap mean must be zero, so the result is `(0, (0, 0))`.
2. Compare a model selected using the same test set: explain why resampling
   cannot undo the resulting selection bias. (The bootstrap only resamples the
   tasks you have; it cannot tell that they were also used to pick the winner.)
3. Following the docstring, compare independent vs. paired bootstrapping on
   strongly correlated tasks and confirm that the paired interval is narrower.

## 11. Practice calculations

1. Differences `[1, 1, 0, -1, 1]`: what is $\hat\Delta$? **$2/5=0.4$.**
2. What are $s_d$ and SE? **Deviations $0.6,0.6,-0.4,-1.4,0.6$; sum of squares
   $0.36\cdot3+0.16+1.96=3.2$; $s_d=\sqrt{3.2/4}=0.894$; SE $=0.894/\sqrt5=0.4$.**
3. With $R=1000$ replicates, which sorted positions does the script's rule
   use? **`int(0.025*1000)=25` and `int(0.975*1000)=975`.**
4. $\operatorname{Var}(a)=0.25$, $\operatorname{Var}(b)=0.25$,
   $\operatorname{Cov}(a,b)=0.2$: what is $\operatorname{Var}(b-a)$?
   **$0.25+0.25-0.4=0.1$; without the covariance term it would be 0.5.**
5. Three seeds give returns `[0.6, 0.8, 0.7]`. Mean and SE?
   **Mean 0.7; $s=\sqrt{(0.01+0.01+0)/2}=0.1$; SE $=0.1/\sqrt3=0.0577$.**
6. Does running 100,000 bootstrap replicates instead of 2000 narrow the
   interval substantially? **No. It only reduces Monte Carlo noise in the
   percentiles; the width is set by the ten real tasks.**

Next: [Activation checkpointing](06_activation_checkpointing.md).
