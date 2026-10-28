"""Tensor shapes, broadcasting, derivatives, and gradient checks.
Requires: torch. A batch has shape [B, features]; a weight matrix [features, out].
Autograd records operations and applies the chain rule in reverse.
Exercise: deliberately omit zero_grad in 03_neural_network_training.py and observe accumulation.
"""
import torch


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    torch.manual_seed(7)
    x = torch.tensor([[1., 2.], [3., 4.]])  # Shape [2 examples, 2 features].
    w = torch.tensor([[0.2], [0.3]], requires_grad=True)
    target = torch.tensor([[1.], [2.]])
    loss = ((x @ w - target)**2).mean()
    loss.backward()  # Populates w.grad; it does NOT update w.
    analytical = w.grad.clone()
    # Finite differences independently check a derivative on this tiny problem.
    eps = 1e-3
    with torch.no_grad():
        original = w[0, 0].item()
        w[0, 0] = original + eps
        plus = ((x @ w - target)**2).mean()
        w[0, 0] = original - eps
        minus = ((x @ w - target)**2).mean()
        w[0, 0] = original
    numerical = (plus - minus) / (2*eps)
    print('gradient:', analytical.tolist(), 'finite difference:', numerical.item())
    assert torch.allclose(analytical[0, 0], numerical, atol=1e-3)
    # Broadcasting [2,1] against [2] produces [2,2], often a SILENT loss bug.
    print('wrong subtraction shape:', (x @ w - target.squeeze(1)).shape)
    print('correct subtraction shape:', (x @ w - target).shape)

if __name__ == '__main__':
    main()
