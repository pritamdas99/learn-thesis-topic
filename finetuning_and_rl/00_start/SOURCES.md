# Primary sources for deeper study

These links are references, not runtime dependencies. Online documentation can
change; match it to the package version you installed. Implementations here are
original teaching examples and deliberately smaller than production trainers.

## Foundations and fine-tuning

- [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html): gradients and computational graphs.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): transformer architecture.
- [LoRA paper](https://arxiv.org/abs/2106.09685) and [PEFT LoRA guide](https://huggingface.co/docs/peft/en/conceptual_guides/lora): low-rank adaptation.
- [PEFT quicktour](https://huggingface.co/docs/peft/en/quicktour): adapters and pretrained model integration.
- [QLoRA paper](https://arxiv.org/abs/2305.14314) and [bitsandbytes integration](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes): quantized base-model tuning.

## Reinforcement learning

- [Sutton and Barto book, author's site](http://incompleteideas.net/book/the-book-2nd.html): MDPs, dynamic programming, Monte Carlo, TD, policy gradients.
- [Gymnasium basic usage](https://gymnasium.farama.org/introduction/basic_usage/): reset/step API and episode boundaries.
- [DQN paper](https://www.nature.com/articles/nature14236): deep Q-learning.
- [Double DQN paper](https://arxiv.org/abs/1509.06461): reducing overestimation.
- [GAE paper](https://arxiv.org/abs/1506.02438): advantage estimation.
- [PPO paper](https://arxiv.org/abs/1707.06347): clipped policy optimization.
- [SAC documentation](https://spinningup.openai.com/en/latest/algorithms/sac.html): entropy-regularized continuous control.
- [CQL paper](https://arxiv.org/abs/2006.04779): conservative offline Q-learning.

## Alignment

- [InstructGPT paper](https://arxiv.org/abs/2203.02155): instruction tuning and human feedback.
- [DPO paper](https://arxiv.org/abs/2305.18290) and [TRL DPO documentation](https://huggingface.co/docs/trl/en/dpo_trainer): direct preference optimization.
- [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) and [TRL GRPO documentation](https://huggingface.co/docs/trl/en/grpo_trainer): group-relative policy optimization and implementation variants.

Suggested paper-reading order: problem -> objective equation -> assumptions ->
algorithm -> experiments -> limitations. Reproduce one equation before reading
all implementation details. The advanced roadmap names further topics without
claiming that this folder implements them all.

## Additional requested methods

- [A3C](https://arxiv.org/abs/1602.01783): asynchronous advantage actor-critic.
- [TRPO](https://arxiv.org/abs/1502.05477): constrained policy updates.
- [SDPO, v1](https://arxiv.org/html/2601.20802v1): feedback-conditioned self-teacher, reverse-KL objective.
- [OPSD, v1](https://arxiv.org/html/2601.18734v1): privileged reference context and on-policy distillation.
- [Self-Distilled RLVR / RLSD, v1](https://arxiv.org/html/2604.03128v1): verifier advantages with teacher-weighted token credit.

These versioned research references were checked on 2026-10-22. The companion
objective labs state their divergence choices and do not reproduce full trainers.
