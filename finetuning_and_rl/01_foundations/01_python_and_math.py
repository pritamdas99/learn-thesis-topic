"""Python and the mathematics hiding inside training.
Run: python3 01_foundations/01_python_and_math.py
Requires: Python only. Read each comment, then change one number and rerun.

A scalar is one number; a vector is a list of numbers; a matrix is a table.
A parameter is a number learned from data. A hyperparameter (e.g. learning
rate) is chosen by us. Loss measures how wrong predictions are.
Exercise: change the target rule to y = -3*x + 1 and recover its parameters.
"""
import math
import random

# A function packages a reusable calculation. zip pairs corresponding elements.
def dot(a, b):
    assert len(a) == len(b), 'Dot products require matching dimensions.'
    return sum(x * y for x, y in zip(a, b))


def softmax(logits):
    # Subtracting the maximum prevents exp(large_number) from overflowing.
    shifted = [math.exp(x - max(logits)) for x in logits]
    return [x / sum(shifted) for x in shifted]


def main():
    random.seed(7)  # Reproducibility: the same pseudorandom sequence each run.
    print('dot product:', dot([1, 2], [3, 4]))
    probabilities = softmax([1, 2, 3])
    print('probabilities:', probabilities, 'sum:', sum(probabilities))
    # Cross-entropy for a single correct class = -log(probability of that class).
    print('loss when class 2 is correct:', -math.log(probabilities[2]))
    xs = [-2, -1, 0, 1, 2]
    ys = [2*x + 3 for x in xs]
    weight, bias, learning_rate = 0.0, 0.0, 0.1
    for step in range(100):
        errors = [weight*x + bias - y for x, y in zip(xs, ys)]
        loss = sum(e*e for e in errors) / len(xs)
        # Derivatives tell us how loss changes when each parameter changes.
        dw = 2 * sum(e*x for e, x in zip(errors, xs)) / len(xs)
        db = 2 * sum(errors) / len(xs)
        weight -= learning_rate * dw  # Descend: move opposite the gradient.
        bias -= learning_rate * db
        if step % 25 == 0:
            print(f'step={step:3} loss={loss:.6f} w={weight:.3f} b={bias:.3f}')
    assert abs(weight - 2) < 0.01 and abs(bias - 3) < 0.01

if __name__ == '__main__':
    main()
