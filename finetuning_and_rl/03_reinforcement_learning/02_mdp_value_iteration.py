"""Markov decision processes and the Bellman equation.
An MDP specifies states, actions, transition probabilities, rewards, discount.
V*(s) = max_a E[r + gamma V*(s')]. Dynamic programming assumes a known model.
This five-state chain terminates at either edge; reaching the right earns +1.
Exercise: add a step penalty or change gamma and explain the resulting values.
"""


def transition(state, action):
    next_state = state + (1 if action == 1 else -1)
    terminated = next_state in (0, 4)
    reward = float(next_state == 4)
    return next_state, reward, terminated


def value_iteration(gamma=0.95):
    values = [0.0]*5
    for _ in range(100):
        updated = values.copy()
        for state in range(1, 4):
            candidates = []
            for action in (0, 1):
                nxt, reward, terminated = transition(state, action)
                candidates.append(reward + gamma * (not terminated) * values[nxt])
            updated[state] = max(candidates)
        change = max(abs(a-b) for a, b in zip(values, updated))
        values = updated
        if change < 1e-10:
            break
    return values


def policy_iteration(gamma=0.95):
    """Evaluate a fixed policy to convergence, then greedily improve it."""
    policy = [0]*5
    values = [0.0]*5
    for _ in range(100):
        for _ in range(10000):
            updated = values.copy()
            for state in range(1, 4):
                nxt, reward, done = transition(state, policy[state])
                updated[state] = reward + gamma*(not done)*values[nxt]
            change = max(abs(a-b) for a, b in zip(values, updated))
            values = updated
            if change < 1e-12:
                break
        stable = True
        for state in range(1, 4):
            candidates = []
            for action in (0, 1):
                nxt, reward, done = transition(state, action)
                candidates.append(reward + gamma*(not done)*values[nxt])
            best = max(range(2), key=lambda a: candidates[a])
            stable &= best == policy[state]
            policy[state] = best
        if stable:
            return values, policy
    raise RuntimeError('Policy iteration did not stabilize')


if __name__ == '__main__':
    v = value_iteration()
    print('optimal state values:', v)
    assert abs(v[1] - 0.95**2) < 1e-9
    # Optimal policy: move right from every nonterminal state.

    pv, policy = policy_iteration()
    print('policy iteration:', pv, policy[1:4])
    assert max(abs(a-b) for a,b in zip(v,pv)) < 1e-9
