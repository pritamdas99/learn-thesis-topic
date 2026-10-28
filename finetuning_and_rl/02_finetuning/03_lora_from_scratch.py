"""Low-rank adaptation (LoRA), trainable counts, and merging.
W_eff = W_frozen + (alpha/r) B A. For W[out,in], A[r,in], B[out,r].
LoRA reduces trainable parameters and optimizer state, not all activation memory.
Exercise: vary rank 1, 2, 4, 8 and compare reconstruction error and parameter count.
"""
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=2, alpha=4):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        # B starts at zero: attaching the adapter initially preserves the base.
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale

    def merged_weight(self):
        return self.base.weight + self.scale * (self.b @ self.a)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    base = nn.Linear(8, 4, bias=False)
    adapter = LoRALinear(base)
    x = torch.randn(128, 8)
    original = base.weight.detach().clone()
    with torch.no_grad():
        target_weight = original + torch.randn(4, 2) @ torch.randn(2, 8) * 0.1
        y = x @ target_weight.T
    optimizer = torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad], lr=0.03)
    for _ in range(200):
        loss = ((adapter(x)-y)**2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert torch.equal(base.weight, original), 'Frozen weights must not change.'
    assert torch.allclose(adapter(x), x @ adapter.merged_weight().T, atol=1e-5)
    print('trainable parameters:', sum(p.numel() for p in adapter.parameters() if p.requires_grad))
    print('final mean squared error:', loss.item())
    # Save the adapter AND base-model identity. Adapter weights alone are not
    # a standalone model. Quantized base merging needs special handling.

if __name__ == '__main__':
    main()
