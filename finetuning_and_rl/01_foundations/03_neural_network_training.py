"""A complete supervised training loop with unseen validation data.
An epoch visits the training set once. A minibatch is part of that set.
ReLU supplies nonlinearity; without it, stacked linear layers remain linear.
Train data updates weights; validation selects settings; test data is held
until the final comparison. Never tune repeatedly on the test set.
Exercise: increase width to 256 and shrink training data to see overfitting.
"""
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    x = torch.randn(600, 2)
    y = ((x[:, 0] * x[:, 1]) > 0).long()  # XOR-like classification.
    train = DataLoader(TensorDataset(x[:400], y[:400]), batch_size=32, shuffle=True)
    model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    best_loss, best_state, stale = float('inf'), None, 0
    for epoch in range(60):
        model.train()  # Activates dropout / training behavior if present.
        for inputs, labels in train:
            optimizer.zero_grad(set_to_none=True)
            # CrossEntropyLoss expects raw logits, NOT already-softmaxed values.
            loss = nn.functional.cross_entropy(model(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()  # eval() does not disable gradients; no_grad does.
        with torch.no_grad():
            val = nn.functional.cross_entropy(model(x[400:500]), y[400:500]).item()
        if val < best_loss:
            best_loss, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch % 10 == 0:
            print(f'epoch={epoch} validation_loss={val:.4f}')
        if stale >= 8:  # Early stopping avoids training past the best validation point.
            break
    model.load_state_dict(best_state)
    with torch.no_grad():
        accuracy = (model(x[500:]).argmax(-1) == y[500:]).float().mean().item()
    print(f'held-out test accuracy: {accuracy:.1%}')

if __name__ == '__main__':
    main()
