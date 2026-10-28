"""Soft Actor-Critic: continuous actions and entropy regularization.
ADVANCED EQUATION LAB: one full actor/critic/temperature/target update on fake
replay data. This is NOT a complete environment-training implementation.
SAC is off-policy; it uses replay and twin critics to reduce overestimation.
Reparameterization a=tanh(mu+sigma*noise) lets gradients flow through sampling.
Exercise: replace fake replay with real transitions from a continuous environment.
"""
import copy
import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(3, 4)  # Two action means and two log standard deviations.

    def sample(self, states):
        mean, log_std = self.net(states).chunk(2, dim=-1)
        distribution = Normal(mean, log_std.clamp(-5, 2).exp())
        raw = distribution.rsample()  # sample() would break pathwise gradients.
        actions = raw.tanh()
        # Change-of-variables correction for tanh; actions lie in [-1,1].
        # The softplus identity is stable even when tanh saturates near +/-1.
        correction = 2*(torch.log(torch.tensor(2.0))-raw-nn.functional.softplus(-2*raw))
        log_prob = (distribution.log_prob(raw)-correction).sum(-1, keepdim=True)
        return actions, log_prob


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    torch.manual_seed(7)
    actor = Actor()
    q1 = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1))
    q2 = copy.deepcopy(q1)
    target1, target2 = copy.deepcopy(q1).requires_grad_(False), copy.deepcopy(q2).requires_grad_(False)
    critics = list(q1.parameters())+list(q2.parameters())
    q_optimizer = torch.optim.Adam(critics, lr=0.001)
    policy_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    log_alpha = nn.Parameter(torch.tensor(-1.0))
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=0.001)
    states, next_states = torch.randn(32, 3), torch.randn(32, 3)
    actions, rewards = torch.rand(32, 2)*2-1, torch.randn(32, 1)
    terminated = torch.zeros(32, 1)
    with torch.no_grad():
        next_actions, next_logs = actor.sample(next_states)
        inputs = torch.cat([next_states, next_actions], -1)
        next_q = torch.minimum(target1(inputs), target2(inputs))-log_alpha.exp()*next_logs
        target = rewards + 0.99*(1-terminated)*next_q
    inputs = torch.cat([states, actions], -1)
    critic_loss = (q1(inputs)-target).square().mean()+(q2(inputs)-target).square().mean()
    q_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    q_optimizer.step()
    for p in critics:
        p.requires_grad_(False)  # Still differentiate Q with respect to ACTIONS.
    sampled, log_prob = actor.sample(states)
    inputs = torch.cat([states, sampled], -1)
    actor_loss = (log_alpha.exp().detach()*log_prob-torch.minimum(q1(inputs), q2(inputs))).mean()
    policy_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    policy_optimizer.step()
    for p in critics:
        p.requires_grad_(True)
    target_entropy = -2.0  # Common heuristic: negative action dimensionality.
    alpha_loss = -(log_alpha*(log_prob.detach()+target_entropy)).mean()
    alpha_optimizer.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_optimizer.step()
    with torch.no_grad():
        for online, target_net in ((q1, target1), (q2, target2)):
            for p, target_p in zip(online.parameters(), target_net.parameters()):
                target_p.lerp_(p, 0.005)  # Polyak averaging: slow target movement.
    print('critic loss:', critic_loss.item(), 'actor loss:', actor_loss.item())

if __name__ == '__main__':
    main()
