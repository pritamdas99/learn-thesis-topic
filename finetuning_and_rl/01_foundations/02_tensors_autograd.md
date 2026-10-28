# Tensors, automatic differentiation, and checking gradients

Companion code: [02_tensors_autograd.py](02_tensors_autograd.py). Read this explanation before running the code.

## The question this example answers

How does PyTorch know which weight changes reduce loss, and how can we check
its answer? We compute the same derivative in three ways: by hand, with
`backward()`, and with small numerical perturbations. Then we examine a shape
mistake that changes the problem without producing an error.

Read [gradient descent](01_python_and_math.md) first. A tensor is an array with
shape, numeric type, and device. `requires_grad=True` enables derivative
tracking for a floating-point tensor. Tracking is different from updating.

## Inputs and matrix multiplication

The arrays are:

$$
X=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
w=\begin{bmatrix}0.2\\0.3\end{bmatrix},\quad
y=\begin{bmatrix}1\\2\end{bmatrix}.
$$

Rows of $X$ are examples. Columns are features. There are two weights and no
bias in this model. Matrix multiplication gives:

$$
\hat y=Xw=\begin{bmatrix}1(0.2)+2(0.3)\\3(0.2)+4(0.3)\end{bmatrix}
=\begin{bmatrix}0.8\\1.8\end{bmatrix}.
$$

The shape rule is `[2,2] @ [2,1] -> [2,1]`. In Python, `@` is matrix
multiplication; `*` would be elementwise multiplication with broadcasting.

## Loss and an explicit derivative

$$
L=\frac12[(0.8-1)^2+(1.8-2)^2]=0.04.
$$

For $N$ examples, the gradient of $L=\|Xw-y\|^2/N$ is:

$$
\nabla_wL=\frac2N X^T(Xw-y).
$$

For this batch, $Xw-y=[-0.2,-0.2]^T$, so:

$$
\frac{\partial L}{\partial w_1}=(-0.2)(1)+(-0.2)(3)=-0.8,
$$

$$
\frac{\partial L}{\partial w_2}=(-0.2)(2)+(-0.2)(4)=-1.2.
$$

The graph is `w -> predictions -> errors -> squared errors -> mean`.
Autograd traverses it backwards and multiplies local derivatives using the
chain rule. `loss.backward()` accumulates the resulting derivatives in
`w.grad`, approximately `[[-0.8],[-1.2]]`. No optimizer is present, so the
weights remain unchanged after this call.

`analytical = w.grad.clone()` saves an independent copy. The name distinguishes
this chain-rule derivative from a finite-difference estimate; PyTorch is doing
automatic differentiation, not symbolic algebra or finite differences.

## Finite differences: measuring a slope without autograd

For a smooth one-variable function:

$$
f'(u)\approx\frac{f(u+\epsilon)-f(u-\epsilon)}{2\epsilon}.
$$

Taylor expansion gives $f(u\pm\epsilon)=f(u)\pm\epsilon f'(u)
+\epsilon^2f''(u)/2\pm\epsilon^3f^{(3)}(u)/6+\cdots$.
Subtracting and dividing cancels the even terms, leaving $f'(u)+O(\epsilon^2)$.
Very small $\epsilon$ can amplify floating-point roundoff; smaller is not
always more accurate.

The script holds $w_2=0.3$ fixed and perturbs only $w_1$ with $\epsilon=0.001$:

| $w_1$ | Predictions | Mean squared error |
| --- | --- | --- |
| 0.201 | [0.801, 1.803] | 0.039205 |
| 0.199 | [0.799, 1.797] | 0.040805 |

The estimated derivative is $(0.039205-0.040805)/0.002=-0.8$.
Restoring the original weight is essential: this is a diagnostic, not a
training update.

| Code | Purpose |
| --- | --- |
| `with torch.no_grad()` | Avoid recording the diagnostic edits/calculations |
| `w[0,0].item()` | Read one weight as a Python number |
| `plus`, `minus` | Evaluate the loss on either side of the original weight |
| `(plus-minus)/(2*eps)` | Central finite-difference estimate |
| `allclose(..., atol=1e-3)` | Compare with absolute tolerance plus default relative tolerance |

## Why broadcasting can silently break a loss

Correct: predictions and targets both have shape `[2,1]`. Each prediction is
compared with its matching target, giving errors `[[-0.2],[-0.2]]`.

`target.squeeze(1)` removes its length-one second dimension, producing shape
`[2]`. PyTorch aligns dimensions from the right. `[2,1]` and `[2]` broadcast to
`[2,2]`, producing all pairwise differences:

$$
\begin{bmatrix}0.8-1&0.8-2\\1.8-1&1.8-2\end{bmatrix}
=\begin{bmatrix}-0.2&-1.2\\0.8&-0.2\end{bmatrix}.
$$

The wrong mean squared error would be $(0.04+1.44+0.64+0.04)/4=0.54$, not 0.04.
Always inspect shapes before computing a regression loss.

## Run and check understanding

Run `python 01_foundations/02_tensors_autograd.py`. Expect numerical derivative
near -0.8, automatic gradient near `[[-0.8],[-1.2]]`, and shape outputs `[2,2]`
and `[2,1]`. Independently perturb the second weight: its estimate should be
near -1.2. Try `eps=0.1` and `eps=1e-8`; this quadratic has no central-difference
truncation error in exact arithmetic, but float32 roundoff can spoil the tiny step.

`manual_seed` has no effect here because the tensors are explicit constants.
`set_num_threads` affects execution scheduling, not the mathematics. Finally,
`if __name__ == '__main__': main()` runs the demonstration only when the file
is executed directly.

Reference: [PyTorch automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).
