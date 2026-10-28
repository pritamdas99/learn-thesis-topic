# Python, vectors, softmax, and gradient descent

Companion code: [01_python_and_math.py](01_python_and_math.py). Read this explanation before running the code.

## What you are learning and why

A model is a parameterized calculation. Training chooses its parameters so
predictions match examples. Before a neural network, understand a model with
only two learned numbers: $\hat y=wx+b$. Here $x$ is an input, $w$ is a slope,
and $b$ is an offset. The code learns the hidden rule $y=2x+3$ from five pairs.

Python lists store the examples, a `for` loop repeats updates, and a function
packages a reusable calculation. `zip(a, b)` pairs corresponding list entries;
`sum(...)` adds them. A list comprehension `[f(x) for x in xs]` applies a
calculation to every item and returns the results as a list.

## Vectors and dot products

A scalar is one number; a vector is an ordered list. For vectors of length $d$:

$$
a\cdot b=\sum_{j=1}^{d}a_jb_j.
$$

For `[1, 2]` and `[3, 4]`, this is $1(3)+2(4)=11$. In a model, the dot product
combines input features with their learned weights. A matrix stores several
vectors as rows or columns, allowing many dot products in one operation.

## Softmax: formula, code, and numerical stability

Logits are arbitrary real scores. To obtain a probability distribution:

$$
p_i=\frac{\exp(z_i)}{\sum_j\exp(z_j)}.
$$

Exponential makes each term positive, and dividing by their sum makes the
probabilities add to one. Higher logits produce higher probabilities. These
probabilities are model outputs, not automatically calibrated confidence.

A direct implementation is:

```python
exponentials = [math.exp(score) for score in logits]
denominator = sum(exponentials)
probabilities = [value / denominator for value in exponentials]
```

The actual `softmax` function subtracts $m=\max_j z_j$ first:

$$
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}
=\frac{e^{-m}e^{z_i}}{e^{-m}\sum_j e^{z_j}}
=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

The common factor cancels, so this is the same formula. But the largest
exponential is now $e^0=1$, avoiding overflow from numbers such as $e^{1000}$.
For `[1,2,3]`, exponentiate `[-2,-1,0]` to get approximately
`[0.1353,0.3679,1]`. Divide by 1.5032 to get `[0.0900,0.2447,0.6652]`.
In the original short function, `shifted` contains these **exponentials**, not
just the shifted scores. The final list comprehension normalizes them.

For a correct class $k$, negative log-likelihood is $-\log p_k$. If the correct
class has probability 0.6652, the loss is about 0.4076. Giving it probability
0.01 produces loss about 4.6052. Minimizing this loss encourages more
probability on the correct class.

## Derivatives: which direction reduces error?

Define error $e_i=wx_i+b-y_i$ and mean squared error:

$$
L(w,b)=\frac1N\sum_i e_i^2.
$$

A derivative measures local change. The chain rule gives:

$$
\frac{\partial L}{\partial w}=\frac2N\sum_i e_ix_i,
\qquad
\frac{\partial L}{\partial b}=\frac2N\sum_i e_i.
$$

Why? The derivative of $e_i^2$ is $2e_i$ times the derivative of $e_i$;
$\partial e_i/\partial w=x_i$ and $\partial e_i/\partial b=1$.

For the code's five inputs `[-2,-1,0,1,2]`, targets are `[-1,1,3,5,7]`.
At $w=b=0$, errors are `[1,-1,-3,-5,-7]`. Therefore $\partial L/\partial w=-8$
and $\partial L/\partial b=-6$.

Gradient descent subtracts the gradient:

$$
w\leftarrow w-\eta\frac{\partial L}{\partial w},\qquad
b\leftarrow b-\eta\frac{\partial L}{\partial b}.
$$

With learning rate $\eta=0.1$, the first updated weights are $w=0.8$, $b=0.6$.
The negative gradient means increasing these parameters initially reduces loss.
This follows from the local approximation
$L(\theta+\Delta)\approx L(\theta)+\nabla L^T\Delta$:
choosing $\Delta=-\eta\nabla L$ predicts a decrease of $\eta\|\nabla L\|^2$.
For too large a step, the approximation can fail and loss may grow.

## Connect to the script

| Code | Meaning |
| --- | --- |
| `errors = [...]` | Calculate $\hat y_i-y_i$ |
| `loss = ... / len(xs)` | Average squared error over examples |
| `dw`, `db` | The two derivatives derived above |
| `weight -= learning_rate * dw` | Gradient descent for the slope |
| `bias -= learning_rate * db` | Gradient descent for the offset |

`random.seed(7)` has no effect on this deterministic dataset; it is a
reproducibility habit for later randomized examples. The code reports loss
before its update and weights after its update, so they refer to adjacent
points in the training loop.

## Run, experiment, and check

Run `python 01_foundations/01_python_and_math.py`. Expect the dot product 11,
softmax probabilities summing to one, and weights approaching 2 and 3.
Try $y=-3x+1$; the learned weights should approach -3 and 1. Explain why adding
100 to every logit leaves softmax unchanged. Check the cancellation above.
A frequent error is adding the gradient instead of subtracting it: that
performs ascent on the loss.

Next: [tensors and automatic differentiation](02_tensors_autograd.md).
