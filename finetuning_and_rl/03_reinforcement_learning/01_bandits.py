"""Rewards, exploration, and incremental value estimates.
A bandit has actions but no changing state. RL adds state transitions and
long-term consequences. Epsilon-greedy explores randomly with probability eps.
Exercise: compare eps=0, 0.1, 0.5 across ten seeds, not just one lucky run.
"""
import random


def main():
    rng = random.Random(7)
    true_means = [0.2, 0.5, 0.8]
    counts, estimates = [0]*3, [0.0]*3
    total_reward = 0
    for step in range(2000):
        action = rng.randrange(3) if rng.random() < 0.1 else max(range(3), key=estimates.__getitem__)
        reward = float(rng.random() < true_means[action])
        counts[action] += 1
        estimates[action] += (reward-estimates[action]) / counts[action]
        total_reward += reward
    print('action counts:', counts, 'estimated rewards:', estimates)
    print('average reward:', total_reward/2000)
    # The environment knows true_means; the learning agent only sees samples.

if __name__ == '__main__':
    main()
