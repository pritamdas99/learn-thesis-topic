"""Offline RL and a simplified discrete Conservative Q-Learning loss.
Offline RL learns from a FIXED dataset; it cannot explore to correct mistakes.
Bootstrapping on unsupported actions can produce overestimated Q values.
CQL adds alpha*(logsumexp_a Q(s,a)-Q(s,a_dataset)) to a Bellman loss.
This equation lab shows the regularizer, not a full CQL benchmark trainer.
Exercise: increase alpha and inspect gradients on actions absent from the data.
"""
import torch


def conservative_penalty(q_values, dataset_actions):
    chosen = q_values.gather(1, dataset_actions[:, None]).squeeze(1)
    return (torch.logsumexp(q_values, dim=-1)-chosen).mean()


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    q = torch.tensor([[1., 5., 2.], [2., 4., 1.]], requires_grad=True)
    actions = torch.tensor([0, 0])
    bellman_targets = torch.tensor([1.2, 1.8])  # In a real loop: detached target-net outputs.
    predicted = q.gather(1, actions[:, None]).squeeze(1)
    bellman = (predicted-bellman_targets).square().mean()
    penalty = conservative_penalty(q, actions)
    loss = bellman+0.5*penalty
    loss.backward()
    print('conservative penalty:', penalty.item(), 'gradient:', q.grad.tolist())
    # Positive gradients on unsupported high-Q actions push their values DOWN
    # under gradient descent. Too much pessimism can also prevent improvement.
    # Behavior cloning is a useful baseline; never trust offline Q values alone.

if __name__ == '__main__':
    main()
