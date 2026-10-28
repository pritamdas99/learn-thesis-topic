"""The bridge from supervised tuning to reward optimization.
Typical RLHF: SFT -> preference reward model -> policy optimization (e.g. PPO).
The frozen reference is usually the SFT policy; it anchors behavior.
Maximize E_pi[reward] - beta KL(pi || reference). This exact three-action
bandit illustrates the objective, NOT a complete autoregressive RLHF trainer.
Exercise: sweep beta and report both expected reward AND KL divergence.
"""
import torch


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    reference = torch.tensor([0.6, 0.3, 0.1])
    reward = torch.tensor([0.0, 0.5, 1.0])
    logits = torch.nn.Parameter(reference.log().clone())
    optimizer = torch.optim.Adam([logits], lr=0.05)
    beta = 0.2
    for _ in range(150):
        log_probs = logits.log_softmax(-1)
        probs = log_probs.exp()
        kl = (probs*(log_probs-reference.log())).sum()
        objective = (probs*reward).sum()-beta*kl
        optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        optimizer.step()
    with torch.no_grad():
        p = logits.softmax(-1)
        exact_optimum = (reference.log()+reward/beta).softmax(-1)
        print('learned policy:', p.tolist(), 'analytic optimum:', exact_optimum.tolist())
    # For sampled text, sum log_pi(token)-log_ref(token) over response tokens.
    # A single sampled log ratio can be negative even though expected KL >= 0.
    # Reward hacking: higher learned reward need not imply higher human quality.

if __name__ == '__main__':
    main()
