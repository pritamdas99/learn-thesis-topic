"""Small CPU A2C/A3C trainers. A3C uses concurrent threads and locked SGD writes.
Local rollout computation is asynchronous; no batch barrier between workers.
"""
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import nn


def make_model():
    return nn.Linear(5, 3)  # two actor logits and one value


def rollout_loss(model, state, generator, steps=5, gamma=0.95):
    logs, values, rewards, entropies = [], [], [], []
    terminal = False
    for _ in range(steps):
        out = model(torch.eye(5)[state])
        probs = out[:2].softmax(-1)
        action = torch.multinomial(probs, 1, generator=generator).item()
        logs.append(probs[action].log())
        entropies.append(-(probs*probs.log()).sum())
        values.append(out[2])
        state += 1 if action else -1
        rewards.append(float(state == 4))
        terminal = state in (0, 4)
        if terminal:
            break
    with torch.no_grad():
        total = torch.tensor(0.) if terminal else model(torch.eye(5)[state])[2]
        targets = []
        for reward in reversed(rewards):
            total = reward + gamma*total
            targets.insert(0, total)
        targets = torch.stack(targets)
    advantage = targets-torch.stack(values)
    loss = -(torch.stack(logs)*advantage.detach()).mean()
    loss += 0.5*advantage.square().mean()-0.01*torch.stack(entropies).mean()
    return loss, 2 if terminal else state


def train_a2c(updates=250, workers=4):
    torch.manual_seed(7)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    generators = [torch.Generator().manual_seed(20+i) for i in range(workers)]
    states = [2]*workers
    for _ in range(updates):
        losses = []
        for i in range(workers):
            loss, states[i] = rollout_loss(model, states[i], generators[i])
            losses.append(loss)
        optimizer.zero_grad()
        torch.stack(losses).mean().backward()
        optimizer.step()  # one synchronized update for all workers
    return model


def train_a3c(updates_per_worker=250, workers=4):
    torch.manual_seed(7)
    global_model = make_model()
    optimizer = torch.optim.SGD(global_model.parameters(), lr=0.03)
    lock = threading.Lock()

    def worker(index):
        with lock:
            local = copy.deepcopy(global_model)
        generator = torch.Generator().manual_seed(40+index)
        state = 2
        for _ in range(updates_per_worker):
            with lock:
                local.load_state_dict(global_model.state_dict())
            local.zero_grad()
            loss, state = rollout_loss(local, state, generator)
            loss.backward()  # other workers can update global parameters here
            with lock:
                optimizer.zero_grad()
                for shared, private in zip(global_model.parameters(), local.parameters()):
                    shared.grad = private.grad.detach().clone()
                optimizer.step()  # stale local gradient, no synchronization barrier

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers)))
    return global_model


if __name__ == '__main__':
    torch.set_num_threads(1)
    for name, train in [('A2C', train_a2c), ('A3C', train_a3c)]:
        model = train()
        with torch.no_grad():
            p = model(torch.eye(5)[1:4])[:, :2].softmax(-1)[:, 1]
        print(name, 'right probabilities:', p.tolist())
        assert torch.isfinite(p).all()
