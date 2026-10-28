"""Trade extra computation for lower activation memory.
Activation checkpointing discards intermediate activations and recomputes them
in backward. It differs from saving model checkpoints to disk.
Exercise: profile peak CUDA memory on a larger network with/without checkpointing.
"""
import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    torch.manual_seed(7)
    ordinary = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 16))
    recomputed = copy.deepcopy(ordinary)
    x = torch.randn(8, 16)
    regular_loss = ordinary(x).square().mean()
    checkpoint_loss = checkpoint(recomputed, x, use_reentrant=False).square().mean()
    regular_loss.backward()
    checkpoint_loss.backward()
    for p, q in zip(ordinary.parameters(), recomputed.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)
    print('recomputed and ordinary gradients match')
    # Real transformers: model.gradient_checkpointing_enable(); usually disable
    # use_cache during training. Savings depend on checkpoint boundaries.
    # Do not put side effects or state-changing operations inside recomputed blocks.

if __name__ == '__main__':
    main()
