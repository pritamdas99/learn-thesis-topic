# The mathematical language used in this course

Read this when an equation's notation gets in the way of the idea. You do not
need to memorize every identity before starting. Return to the relevant section
when it first appears in a lesson.

## 1. Indices, sums, and shapes

$x_i$ means entry $i$ of a vector; $x_{ij}$ means row $i$, column $j$ of a matrix.
$\sum_{i=1}^N x_i$ means add entries 1 through $N$. Thus
$\frac1N\sum_i x_i$ is their arithmetic mean. A superscript such as $x^2$
usually means a power, but $V^\pi$ means “value under policy pi,” not exponentiation.

A matrix $X\in\mathbb R^{B\times d}$ has $B$ rows and $d$ columns of real
numbers. If $W\in\mathbb R^{d\times k}$, then $XW$ has shape $[B,k]$:

$$
(XW)_{ij}=\sum_{l=1}^{d}X_{il}W_{lj}.
$$

The transpose $X^T$ swaps rows and columns. PyTorch's `nn.Linear(d,k)` stores
weights as $[k,d]$ and therefore calculates `x @ weight.T + bias` for row batches.
Always write shapes beside an unfamiliar equation before implementing it.

## 2. Functions and derivatives

A function maps inputs to outputs: $f(w)=w^2$ maps 3 to 9. Its derivative
$f'(w)=2w$ describes local sensitivity. At $w=3$, a small increment $h$ changes
the function by approximately $6h$:

$$
f(w+h)\approx f(w)+h f'(w).
$$

For several variables, a partial derivative changes one variable while holding
the others fixed. The gradient stacks all partial derivatives into a vector:

$$
\nabla_\theta L=
[\partial L/\partial\theta_1,\ldots,\partial L/\partial\theta_d]^T.
$$

The chain rule differentiates a composition. If $u=3w$ and $L=u^2$, then
$dL/dw=(dL/du)(du/dw)=2u\cdot3=18w$. Backpropagation repeatedly applies this
rule to a computation graph, adding contributions when a parameter affects
the output through several paths.

Gradient descent is $\theta\leftarrow\theta-\eta\nabla L$, with learning rate
$\eta>0$. Locally it moves against increasing loss. In a nonconvex network,
this does not guarantee finding the global best parameters.

## 3. Probability and expectation

A discrete probability distribution has $p_i\ge0$ and $\sum_i p_i=1$.
$X\sim p$ means sample a random variable from distribution $p$.
The expectation of a function is a probability-weighted average:

$$
\mathbb E_{X\sim p}[f(X)]=\sum_x p(x)f(x).
$$

For rewards 0 and 1 with probabilities 0.2 and 0.8, expected reward is 0.8.
A single sample is still either 0 or 1. An expectation describes an average,
not what must happen on each attempt. A sample mean estimates an expectation;
its reliability depends on the number and dependence of observations.

Conditional probability $p(y\mid x)$ means probability of $y$ given $x$.
An RL policy $\pi(a\mid s)$ is the action distribution conditional on state.
The law of total expectation lets us average first over a next state and then
over actions; this is the basis of Bellman expectation equations.

## 4. Logarithms, exponentials, and likelihood

Natural log and exponential are inverse functions: $\log(e^u)=u$.
Products become sums: $\log(ab)=\log a+\log b$.
A sequence of conditional token probabilities multiplies to a sequence
probability, but their logs add. Working in log space avoids underflow from
multiplying many tiny numbers.

A likelihood treats observed data as fixed and model parameters as variable.
Maximizing its log is equivalent to minimizing negative log-likelihood.
For a correct class with model probability $p$, its loss is $-\log p$.
Probability 1 gives loss 0; probability 0.1 gives about 2.303.

Softmax converts logits $z_i$ into probabilities:

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

Subtracting the same constant from all logits preserves these probabilities.
Subtract the maximum to improve numerical stability. See the
[worked softmax lesson](../01_foundations/01_python_and_math.md).

## 5. Variance, bias, and standardization

Variance measures expected squared deviation from the mean:

$$
\operatorname{Var}(X)=\mathbb E[(X-\mathbb E[X])^2].
$$

Standard deviation is its square root. Standardization subtracts a mean and
divides by a standard deviation, usually adding a tiny denominator constant
for numerical stability. Standardized data is not automatically Gaussian.

For an estimator $\hat q$ of $q$, bias is $\mathbb E[\hat q]-q$.
A method can be consistently wrong (bias) or fluctuate greatly (variance).
Under the usual squared-error decomposition:

$$
\mathbb E[(\hat q-q)^2]=\operatorname{Var}(\hat q)
+(\mathbb E[\hat q]-q)^2.
$$

GAE trades reliance on learned value estimates against noisy observed returns.
These are tendencies, not universal numeric guarantees for every parameter choice.

## 6. Entropy, cross-entropy, and KL divergence

For distributions $p$ and $q$ over the same discrete outcomes:

$$
H(p)=-\sum_i p_i\log p_i,
$$

$$
H(p,q)=-\sum_i p_i\log q_i,
$$

$$
D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}=H(p,q)-H(p).
$$

Entropy measures uncertainty within $p$. Cross-entropy measures the average
negative log-probability that $q$ assigns to outcomes drawn from $p$.
KL compares distributions and is nonnegative when well-defined, but is not a
symmetric distance. If $q_i=0$ where $p_i>0$, KL is infinite.

For a one-hot true label, $H(p)=0$, so cross-entropy becomes the negative log
of the predicted probability of that label. For distillation, the teacher
provides a soft target distribution instead.

A sampled log-ratio $\log(p_i/q_i)$ can be negative. Only its expectation under
$p$ is constrained to be nonnegative. Continuous-action differential entropy
has different properties from discrete entropy and may itself be negative.

## 7. Optimization versus evaluation

$\arg\max_a f(a)$ returns an input action achieving a maximum.
$\max_a f(a)$ returns the maximum value itself. Double DQN deliberately uses
one network for the argmax selection and another to evaluate the selected action.

A training loss is the quantity optimized. A task metric is the quantity you
care about, such as held-out correctness or episode return. They may disagree:
a better reward-model score can reflect a shortcut, and a lower training loss
can reflect memorization.

A `stopgrad` or `detach` operation preserves a value in the forward calculation
while treating it as constant during differentiation. This distinction is
central to target networks, policy-gradient advantages, and frozen teachers.

## Practice before the advanced lessons

- Compute softmax for `[0,0]`: answer `[0.5,0.5]`.
- Differentiate $(2w-1)^2$: answer $4(2w-1)$.
- Compute expected reward for values `[0,2]` with probabilities `[0.75,0.25]`:
  answer 0.5.
- Compute entropy of two equally likely outcomes: answer $\log2\approx0.693$ nats.
- For rewards `[0,1]` and discount 0.9, the initial return is 0.9, not 1.

Continue with the [theory and code index](THEORY_INDEX.md).

## 8. Linear systems and curvature for policy iteration and TRPO

A linear system $Ax=b$ asks for a vector whose matrix product equals $b$.
For $A=\operatorname{diag}(2,4)$ and $b=(6,8)^T$, the solution is $(3,2)^T$.
Numerically solve the system rather than explicitly forming $A^{-1}$.
Policy evaluation solves $(I-\gamma P_\pi)V=r_\pi$; each row couples a state's
value to expected successor values.

A Hessian is the matrix of second partial derivatives,
$H_{ij}=\partial^2 f/(\partial\theta_i\partial\theta_j)$.
For $f(\theta)=\theta_1^2+2\theta_2^2$, it is $\operatorname{diag}(2,4)$.
Taylor's local quadratic approximation is

$$f(\theta+d)\approx f(\theta)+\nabla f(\theta)^Td+\tfrac12d^THd.$$

A symmetric matrix is positive definite if $d^THd>0$ for every nonzero $d$;
positive semidefinite permits zero. KL curvature at equal distributions is
semidefinite. Adding $\eta I$, with $\eta>0$, makes the system easier to solve
and handles redundant softmax logit shifts. TRPO solves a curvature system to
scale steps according to policy change rather than raw parameter distance.
Conjugate gradient approximates such a solve using matrix-vector products,
without storing the full matrix. The small TRPO lab forms the matrix explicitly.
