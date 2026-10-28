"""Learn a reward score from chosen/rejected response pairs.
Bradley-Terry: P(chosen wins) = sigmoid(r_chosen - r_rejected).
Only score differences are identified. The absolute reward zero is arbitrary.
Real reward models score prompt+response tokens; here features expose the math.
Exercise: add a spurious length feature and watch shortcut learning appear.
"""
import torch
from torch import nn


def preference_loss(chosen_scores, rejected_scores):
    return -nn.functional.logsigmoid(chosen_scores-rejected_scores).mean()


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    torch.manual_seed(7)
    # Features mean [correctness, clarity, verbosity]; these are synthetic labels,
    # not a practical way to measure response quality automatically.
    chosen = torch.tensor([[1., 1., 0.3], [1., 0.8, 0.2], [0.8, 1., 0.5]])
    rejected = torch.tensor([[0., 1., 0.8], [0.2, 0.8, 0.7], [0.1, 0.5, 0.5]])
    model = nn.Linear(3, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(100):
        loss = preference_loss(model(chosen), model(rejected))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print('training pair accuracy:', (model(chosen)>model(rejected)).float().mean().item())
    print('learned feature weights:', model.weight.detach().tolist())
    # Held-out prompts, disagreement analysis, and adversarial evaluation are
    # essential before using such a score as an optimization target.

if __name__ == '__main__':
    main()
