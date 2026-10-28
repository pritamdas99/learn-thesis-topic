"""Model-free tabular Q-learning and SARSA.
Q(s,a) estimates discounted future return after action a in state s.
Q-learning bootstraps with max_a Q(s',a): off-policy (greedy target).
SARSA bootstraps with the action actually sampled: on-policy.
A terminal transition has no future value. A time limit is different: it
usually still bootstraps because the underlying task did not truly terminate.
Exercise: add a cliff penalty to see how on/off-policy targets differ.
"""
import random


def train(method='q_learning', seed=7):
    rng = random.Random(seed)
    q = [[0.0, 0.0] for _ in range(5)]
    def choose(state):
        if rng.random() < 0.2:
            return rng.randrange(2)
        # Randomize ties to avoid systematically favoring action zero.
        best = max(q[state])
        return rng.choice([a for a in (0, 1) if q[state][a] == best])
    for _ in range(500):
        state, action = 2, choose(2)
        for step in range(30):
            nxt = state + (1 if action else -1)
            terminated = nxt in (0, 4)
            reward = float(nxt == 4)
            next_action = choose(nxt)
            bootstrap = max(q[nxt]) if method == 'q_learning' else q[nxt][next_action]
            target = reward + 0.95 * (not terminated) * bootstrap
            q[state][action] += 0.2 * (target-q[state][action])
            if terminated:
                break
            state, action = nxt, next_action
            # Ending at 30 steps truncates collection, not the Bellman target.
    return q


if __name__ == '__main__':
    for method in ('q_learning', 'sarsa'):
        q = train(method)
        print(method, 'Q values:', q[1:4])
        assert all(row[1] > row[0] for row in q[1:4])
