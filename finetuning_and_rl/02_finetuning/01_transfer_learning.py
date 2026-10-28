"""Pretrain, replace the head, freeze, then gradually unfreeze.
Fine-tuning starts from weights learned on a previous task. Feature extraction
trains only the new head; full fine-tuning updates the backbone too.
Synthetic source/target tasks keep this runnable without downloading anything.
Exercise: compare head-only and full tuning on a small target training set.
"""
import torch
from torch import nn


def fit(model, x, y, steps, lr):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        loss = nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    x = torch.randn(400, 2)
    source_labels = (x[:, 0] > 0).long()
    target_labels = (x[:, 0] + 0.5*x[:, 1] > 0).long()
    backbone = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
    model = nn.Sequential(backbone, nn.Linear(16, 2))
    fit(model, x[:300], source_labels[:300], 100, 0.03)
    model[1] = nn.Linear(16, 2)  # New task, new head.
    for p in backbone.parameters():
        p.requires_grad_(False)
    fit(model, x[:300], target_labels[:300], 60, 0.02)
    for p in backbone.parameters():
        p.requires_grad_(True)
    # Recreate the optimizer so newly unfrozen parameters are included.
    fit(model, x[:300], target_labels[:300], 60, 0.002)
    with torch.no_grad():
        accuracy = (model(x[300:]).argmax(-1) == target_labels[300:]).float().mean()
    print('target test accuracy:', accuracy.item())
    # Real transfer: use a lower backbone LR, monitor source-task forgetting,
    # and handle BatchNorm running statistics when freezing vision backbones.

if __name__ == '__main__':
    main()
