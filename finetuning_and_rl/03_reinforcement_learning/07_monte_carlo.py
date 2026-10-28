"""Model-based rollout planning and model-free first-visit Monte Carlo control.
Only standard library required. Both use complete episodes, no bootstrapping.
"""
import random


def model_step(state, action):
    nxt = state + (1 if action else -1)
    return nxt, float(nxt == 4), nxt in (0, 4)


def simulate(state, first_action, rng, gamma=0.95):
    # Simulator/model access: fixed random continuation, not optimal control.
    total, discount, action = 0., 1., first_action
    while True:
        state, reward, done = model_step(state, action)
        total += discount*reward
        if done:
            return total
        discount *= gamma
        action = rng.randrange(2)


def rollout_plan(state=2, samples=10000):
    rng = random.Random(7)
    q = [sum(simulate(state, a, rng) for _ in range(samples))/samples
         for a in (0, 1)]
    return q, max(range(2), key=lambda a: q[a])


def mc_control(step, episodes=20000, gamma=0.95, epsilon=0.2):
    # Learner only calls step: it never reads transitions or simulator internals.
    rng = random.Random(11)
    q, counts = [[0., 0.] for _ in range(5)], [[0, 0] for _ in range(5)]
    for _ in range(episodes):
        state, trajectory = 2, []
        while state not in (0, 4):
            greedy = max(range(2), key=lambda a: q[state][a])
            action = rng.randrange(2) if rng.random() < epsilon else greedy
            nxt, reward, done = step(state, action)
            trajectory.append((state, action, reward))
            state = nxt
            if done:
                break
        returns, total = [0.]*len(trajectory), 0.
        for t in reversed(range(len(trajectory))):
            total = trajectory[t][2] + gamma*total
            returns[t] = total
        seen = set()
        for (state, action, _), total in zip(trajectory, returns):
            if (state, action) in seen:
                continue
            seen.add((state, action))
            counts[state][action] += 1
            q[state][action] += (total-q[state][action])/counts[state][action]
    return q


if __name__ == '__main__':
    q, action = rollout_plan()
    print('model-based random-continuation Q:', q, 'chosen action:', action)
    learned = mc_control(model_step)
    print('model-free MC Q:', learned[1:4])
    assert action == 1
    assert all(row[1] > row[0] for row in learned[1:4])
