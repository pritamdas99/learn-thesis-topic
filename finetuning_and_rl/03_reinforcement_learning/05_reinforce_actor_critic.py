"""Policy gradients, REINFORCE, and a learned value baseline.
Policy pi(a|s) directly produces action probabilities. REINFORCE weights
log pi(a|s) by observed return. Subtracting V(s) reduces variance without
changing the expected policy gradient (when treated as a baseline).
This uses Monte Carlo returns with a learned critic, not TD-based A2C.
Exercise: replace advantage with raw returns and compare learning variance.
"""
import torch
from torch import nn
from torch.distributions import Categorical


def discounted_returns(rewards, gamma=0.95):
    result, total = [], 0.0
    for reward in reversed(rewards):
        total = reward + gamma*total
        result.insert(0, total)
    return torch.tensor(result)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    actor, critic = nn.Linear(5, 2), nn.Linear(5, 1)
    optimizer = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()), lr=0.03)
    features = torch.eye(5)
    for episode in range(250):
        state, logs, values, rewards, entropies = 2, [], [], [], []
        # Here the 30-step horizon is PART OF THE DEFINED finite task: no
        # bootstrap at its end. A continuing-task time limit needs bootstrap.
        for _ in range(30):
            distribution = Categorical(logits=actor(features[state]))
            action = distribution.sample()
            logs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            values.append(critic(features[state]).squeeze(-1))
            state += 1 if action.item() else -1
            rewards.append(float(state == 4))
            if state in (0, 4):
                break
        returns = discounted_returns(rewards)
        values = torch.stack(values)
        advantage = returns-values
        # detach prevents the actor objective from modifying the baseline.
        actor_loss = -(torch.stack(logs)*advantage.detach()).mean()
        critic_loss = advantage.square().mean()
        entropy = torch.stack(entropies).mean()
        loss = actor_loss + 0.5*critic_loss - 0.01*entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        print('probability of right:', actor(features[1:4]).softmax(-1)[:, 1].tolist())

if __name__ == '__main__':
    main()
