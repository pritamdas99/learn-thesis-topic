"""TRPO equation lab: exact expectations for a three-action bandit.
Dense Hessian solve replaces conjugate gradient, appropriate only at toy scale.
Includes a KL constraint and backtracking; no environment/critic trainer.
"""
import torch


def trpo_step(theta, rewards, delta=0.01, damping=1e-3):
    old_logs = theta.detach().log_softmax(-1)
    old_probs = old_logs.exp()
    advantages = (rewards-(old_probs*rewards).sum()).detach()

    def surrogate(x):
        ratios = (x.log_softmax(-1)-old_logs).exp()
        return (old_probs*ratios*advantages).sum()

    def kl(x):
        return (old_probs*(old_logs-x.log_softmax(-1))).sum()

    gradient = torch.autograd.functional.jacobian(surrogate, theta)
    fisher = torch.autograd.functional.hessian(kl, theta)
    metric = fisher+damping*torch.eye(theta.numel(), dtype=theta.dtype)
    direction = torch.linalg.solve(metric, gradient)
    curvature = direction@metric@direction
    if curvature <= 1e-16:
        return theta.detach().clone(), 0., False
    step = (2*delta/curvature).sqrt()*direction
    before = surrogate(theta).detach()
    for j in range(15):
        candidate = theta.detach()+(0.5**j)*step
        candidate_kl = kl(candidate).item()
        if candidate_kl <= delta and surrogate(candidate) > before:
            return candidate, candidate_kl, True
    return theta.detach().clone(), 0., False


if __name__ == '__main__':
    theta = torch.zeros(3, dtype=torch.float64)
    rewards = torch.tensor([0., 0.3, 1.], dtype=torch.float64)
    for _ in range(15):
        theta, kl, accepted = trpo_step(theta, rewards)
        assert kl <= 0.01+1e-10
    print('TRPO action probabilities:', theta.softmax(-1).tolist())
    assert theta.softmax(-1)[2] > 0.8
