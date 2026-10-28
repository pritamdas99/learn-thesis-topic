"""Group Relative Policy Optimization, with one-token responses.
Sample several responses for the SAME prompt; normalize their rewards within
that group; use those relative advantages in a PPO-like objective. No critic.
This teaching variant uses population standard deviation, clipped ratios and
an exact categorical KL term. Production GRPO variants differ in normalization,
length weighting, KL estimator and clipping; this is not a TRL default replica.
Exercise: give every response the same reward and explain the zero advantage.
"""
import torch
from torch import nn
from torch.distributions import Categorical


def group_advantages(rewards):
    return (rewards-rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True, unbiased=False)+1e-6)


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    torch.manual_seed(7)
    # Each row is a prompt. Each action is a one-token possible answer.
    logits = nn.Parameter(torch.zeros(2, 4))
    reference_logs = torch.zeros(2, 4).log_softmax(-1)
    correct_answers = torch.tensor([1, 3])
    optimizer = torch.optim.Adam([logits], lr=0.04)
    for iteration in range(100):
        with torch.no_grad():
            old_logs_all = logits.log_softmax(-1)
            actions = Categorical(logits=logits).sample((8,)).T  # [prompts, group].
            old_logs = old_logs_all.gather(1, actions)
            rewards = (actions == correct_answers[:, None]).float()
            advantages = group_advantages(rewards)
        for _ in range(2):
            logs_all = logits.log_softmax(-1)
            new_logs = logs_all.gather(1, actions)
            ratio = (new_logs-old_logs).exp()
            surrogate = torch.minimum(ratio*advantages, ratio.clamp(0.8, 1.2)*advantages)
            kl = (logs_all.exp()*(logs_all-reference_logs)).sum(-1).mean()
            loss = -surrogate.mean()+0.02*kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    print('answer probabilities:', logits.softmax(-1).detach().tolist())
    assert torch.equal(group_advantages(torch.ones(2, 8)), torch.zeros(2, 8))
    # Verifiable rewards still require tests for shortcuts and incorrect verifiers.

if __name__ == '__main__':
    main()
