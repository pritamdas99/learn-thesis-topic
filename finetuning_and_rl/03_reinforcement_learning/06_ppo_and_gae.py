"""Proximal Policy Optimization and generalized advantage estimation.
PPO reuses a fresh on-policy rollout for a FEW epochs, then collects new data.
ratio = exp(log_pi_new - log_pi_old). Clipping limits the surrogate incentive
for large policy changes; it is not a hard KL constraint or safety guarantee.
This is a complete tiny chain trainer; no Gymnasium dependency is required.
Exercise: vary clip=0.1/0.2/0.4 and log approximate KL and success rate.
"""
import torch
from torch import nn
from torch.distributions import Categorical


def gae(rewards, values, next_values, terminated, episode_end, gamma=0.95, lam=0.95):
    # terminated masks value bootstrap; episode_end (terminated OR truncated)
    # prevents the trace from crossing into a reset episode.
    deltas = rewards + gamma*(1-terminated)*next_values-values
    advantages = torch.zeros_like(rewards)
    carry = 0.0
    for t in reversed(range(len(rewards))):
        carry = deltas[t] + gamma*lam*(1-episode_end[t])*carry
        advantages[t] = carry
    return advantages, advantages+values


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    actor, critic = nn.Linear(5, 2), nn.Linear(5, 1)
    parameters = list(actor.parameters())+list(critic.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.02)
    features = torch.eye(5)
    state, age = 2, 0
    for iteration in range(35):
        states, actions, old_logs, rewards, values, next_values, terminals, ends = [], [], [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(64):
                distribution = Categorical(logits=actor(features[state]))
                action = distribution.sample()
                nxt = state + (1 if action.item() else -1)
                age += 1
                terminal, truncated = nxt in (0, 4), age >= 20
                states.append(state)
                actions.append(action.item())
                old_logs.append(distribution.log_prob(action))
                values.append(critic(features[state]).squeeze())
                # Use the final observation, BEFORE resetting, for bootstrap.
                next_values.append(critic(features[nxt]).squeeze())
                rewards.append(float(nxt == 4))
                terminals.append(float(terminal))
                ends.append(float(terminal or truncated))
                state = nxt
                if terminal or truncated:
                    state, age = 2, 0
        states, actions = torch.tensor(states), torch.tensor(actions)
        old_logs, values, next_values = map(torch.stack, (old_logs, values, next_values))
        advantages, returns = gae(torch.tensor(rewards), values, next_values,
                                  torch.tensor(terminals), torch.tensor(ends))
        advantages = (advantages-advantages.mean()) / (advantages.std(unbiased=False)+1e-8)
        # Targets, advantages and old log-probabilities stay fixed across epochs.
        for _ in range(4):
            distribution = Categorical(logits=actor(features[states]))
            new_logs = distribution.log_prob(actions)
            ratio = (new_logs-old_logs).exp()
            unclipped = ratio*advantages
            clipped = ratio.clamp(0.8, 1.2)*advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (critic(features[states]).squeeze(-1)-returns).square().mean()
            loss = policy_loss + 0.5*value_loss - 0.01*distribution.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
        if iteration % 10 == 0:
            print('iteration:', iteration, 'successful terminal transitions:', int(sum(rewards)))
    with torch.no_grad():
        print('right probabilities:', actor(features[1:4]).softmax(-1)[:, 1].tolist())

if __name__ == '__main__':
    main()
