"""Deep Q-networks: replay, target networks, and Double DQN.
A neural net replaces the Q table. Replay breaks temporal correlations.
A delayed target network reduces moving-target instability. Double DQN uses
the online net to SELECT an action and the target net to EVALUATE it.
Exercise: remove replay/target networks one at a time and compare seed variance.
"""
import argparse
import copy
import random
from collections import deque
import torch
from torch import nn


def main(double=True):
    rng = random.Random(7)
    torch.manual_seed(7)
    torch.set_num_threads(1)
    net = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 2))
    target = copy.deepcopy(net).requires_grad_(False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
    replay = deque(maxlen=2000)
    features = torch.eye(5)
    updates = 0
    for episode in range(200):
        state = 2
        epsilon = max(0.05, 1-episode/150)
        for step in range(30):
            with torch.no_grad():
                action = rng.randrange(2) if rng.random() < epsilon else net(features[state]).argmax().item()
            nxt = state + (1 if action else -1)
            terminated = nxt in (0, 4)
            replay.append((state, action, float(nxt == 4), nxt, terminated))
            if len(replay) >= 32:
                s, a, r, ns, done = zip(*rng.sample(list(replay), 32))
                s, a, ns = map(lambda z: torch.tensor(z, dtype=torch.long), (s, a, ns))
                r, done = torch.tensor(r), torch.tensor(done, dtype=torch.float32)
                with torch.no_grad():
                    if double:
                        next_actions = net(features[ns]).argmax(-1, keepdim=True)
                        next_q = target(features[ns]).gather(1, next_actions).squeeze(1)
                    else:
                        next_q = target(features[ns]).max(-1).values
                    targets = r + 0.95*(1-done)*next_q
                predicted = net(features[s]).gather(1, a[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(predicted, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()
                updates += 1
                if updates % 50 == 0:
                    target.load_state_dict(net.state_dict())
            state = nxt
            if terminated:
                break
    with torch.no_grad():
        print('greedy actions for states 1,2,3:', net(features[1:4]).argmax(-1).tolist())
    # Prioritized replay is an extension: sample large TD errors more often,
    # but use importance weights to correct the changed sampling distribution.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vanilla-dqn', action='store_true', help='Use original DQN target instead of Double DQN')
    args = parser.parse_args()
    main(double=not args.vanilla_dqn)
