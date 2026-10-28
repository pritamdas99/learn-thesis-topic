"""Evaluate outcomes, not just loss; quantify uncertainty.
Use held-out tasks and multiple seeds. Compare methods under equal data/compute.
This paired bootstrap uses the same sampled task IDs for both methods.
Exercise: compare independent vs paired bootstrapping on strongly correlated tasks.
"""
import random
import statistics


def paired_bootstrap(baseline, candidate, samples=2000, seed=7):
    if not baseline or len(baseline) != len(candidate):
        raise ValueError('Need nonempty paired scores on the same evaluation tasks.')
    rng = random.Random(seed)
    differences = [b-a for a, b in zip(baseline, candidate)]
    means = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return statistics.mean(differences), (means[int(0.025*samples)], means[int(0.975*samples)])


if __name__ == '__main__':
    before = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    after =  [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    delta, interval = paired_bootstrap(before, after)
    print('mean improvement:', delta, 'approximate 95% bootstrap interval:', interval)
    # Ten synthetic tasks are far too few for a serious scientific claim.
    # If examples share documents/users, resample clusters, not individual rows.
    # RL: report return distribution across held-out evaluation seeds; a single
    # high-return episode does not demonstrate reliable learning.
