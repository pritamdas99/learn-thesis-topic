# Theory and code: complete reading index

Read the explanation first, work through its numerical example, then open the matching code. Each row pairs files with exactly the same filename stem.

If the notation is unfamiliar, start with the [math primer](MATH_PRIMER.md). For the overall progression, see the [learning map](LEARNING_MAP.md).

## 01 foundations

| Topic | Mathematical explanation | Runnable code |
| --- | --- | --- |
| Python, vectors, softmax, and gradient descent | [Read theory](../01_foundations/01_python_and_math.md) | [Open Python](../01_foundations/01_python_and_math.py) |
| Tensors, automatic differentiation, and checking gradients | [Read theory](../01_foundations/02_tensors_autograd.md) | [Open Python](../01_foundations/02_tensors_autograd.py) |
| Neural networks and the complete training loop | [Read theory](../01_foundations/03_neural_network_training.md) | [Open Python](../01_foundations/03_neural_network_training.py) |
| Tokens, embeddings, and causal attention | [Read theory](../01_foundations/04_tokens_attention.md) | [Open Python](../01_foundations/04_tokens_attention.py) |
| A tiny transformer language model from start to sampling | [Read theory](../01_foundations/05_tiny_transformer.md) | [Open Python](../01_foundations/05_tiny_transformer.py) |

## 02 finetuning

| Topic | Mathematical explanation | Runnable code |
| --- | --- | --- |
| Transfer learning: reuse features, then adapt them | [Read theory](../02_finetuning/01_transfer_learning.md) | [Open Python](../02_finetuning/01_transfer_learning.py) |
| Instruction tuning, response-only loss, and token masks | [Read theory](../02_finetuning/02_instruction_data_and_masks.md) | [Open Python](../02_finetuning/02_instruction_data_and_masks.py) |
| LoRA: adapting a model through low-rank weight updates | [Read theory](../02_finetuning/03_lora_from_scratch.md) | [Open Python](../02_finetuning/03_lora_from_scratch.py) |
| Fine-tuning a pretrained language model: objective, data, and updates | [Read theory](../02_finetuning/04_huggingface_sft.md) | [Open Python](../02_finetuning/04_huggingface_sft.py) |
| Reloading models and generating tokens | [Read theory](../02_finetuning/05_reload_and_generate.md) | [Open Python](../02_finetuning/05_reload_and_generate.py) |

## 03 reinforcement learning

| Topic | Mathematical explanation | Runnable code |
| --- | --- | --- |
| Bandits: learning which action is rewarding | [Read theory](../03_reinforcement_learning/01_bandits.md) | [Open Python](../03_reinforcement_learning/01_bandits.py) |
| MDPs, Bellman equations, value iteration, and policy iteration | [Read theory](../03_reinforcement_learning/02_mdp_value_iteration.md) | [Open Python](../03_reinforcement_learning/02_mdp_value_iteration.py) |
| Q-learning and SARSA: Bellman learning from experience | [Read theory](../03_reinforcement_learning/03_q_learning_sarsa.md) | [Open Python](../03_reinforcement_learning/03_q_learning_sarsa.py) |
| DQN and Double DQN: learning action values with a neural network | [Read theory](../03_reinforcement_learning/04_dqn.md) | [Open Python](../03_reinforcement_learning/04_dqn.py) |
| Policy gradients, REINFORCE, and a learned baseline | [Read theory](../03_reinforcement_learning/05_reinforce_actor_critic.md) | [Open Python](../03_reinforcement_learning/05_reinforce_actor_critic.py) |
| PPO and GAE: from rewards to a stable policy update | [Read theory](../03_reinforcement_learning/06_ppo_and_gae.md) | [Open Python](../03_reinforcement_learning/06_ppo_and_gae.py) |
| Model-based rollout planning and model-free Monte Carlo control | [Read theory](../03_reinforcement_learning/07_monte_carlo.md) | [Open Python](../03_reinforcement_learning/07_monte_carlo.py) |
| A2C and concurrent A3C | [Read theory](../03_reinforcement_learning/08_a2c_a3c.md) | [Open Python](../03_reinforcement_learning/08_a2c_a3c.py) |
| TRPO constrained update | [Read theory](../03_reinforcement_learning/09_trpo.md) | [Open Python](../03_reinforcement_learning/09_trpo.py) |

## 04 alignment

| Topic | Mathematical explanation | Runnable code |
| --- | --- | --- |
| Reward models: learning scores from pairwise preferences | [Read theory](../04_alignment/01_reward_model.md) | [Open Python](../04_alignment/01_reward_model.py) |
| RLHF and KL regularization: improve reward without unrestricted drift | [Read theory](../04_alignment/02_rlhf_kl_penalty.md) | [Open Python](../04_alignment/02_rlhf_kl_penalty.py) |
| DPO: derive preference optimization from a regularized reward objective | [Read theory](../04_alignment/03_dpo.md) | [Open Python](../04_alignment/03_dpo.py) |
| GRPO: compare groups of answers instead of learning a value critic | [Read theory](../04_alignment/04_grpo.md) | [Open Python](../04_alignment/04_grpo.py) |
| SDPO, RLSD, and OPSD objectives | [Read theory](../04_alignment/05_sdpo_rlsd_opsd.md) | [Open Python](../04_alignment/05_sdpo_rlsd_opsd.py) |

## 05 advanced

| Topic | Mathematical explanation | Runnable code |
| --- | --- | --- |
| Efficient training: accumulation, precision, clipping, schedules, and checkpoints | [Read theory](../05_advanced/01_training_efficiency.md) | [Open Python](../05_advanced/01_training_efficiency.py) |
| Soft Actor-Critic: continuous actions, entropy, and twin value estimates | [Read theory](../05_advanced/02_sac_update.md) | [Open Python](../05_advanced/02_sac_update.py) |
| Offline RL and the conservative Q-learning penalty | [Read theory](../05_advanced/03_offline_rl_cql.md) | [Open Python](../05_advanced/03_offline_rl_cql.py) |
| Distillation and quantization: two different ways to make models smaller | [Read theory](../05_advanced/04_distillation_and_quantization.md) | [Open Python](../05_advanced/04_distillation_and_quantization.py) |
| Evaluation and uncertainty: how much should you trust an improvement? | [Read theory](../05_advanced/05_evaluation_and_uncertainty.md) | [Open Python](../05_advanced/05_evaluation_and_uncertainty.py) |
| Activation checkpointing: recompute values to save training memory | [Read theory](../05_advanced/06_activation_checkpointing.md) | [Open Python](../05_advanced/06_activation_checkpointing.py) |
| Distributed data parallel training: split data and average gradients | [Read theory](../05_advanced/07_distributed_training.md) | [Open Python](../05_advanced/07_distributed_training.py) |

Every explanation includes the problem, key notation, mathematical reasoning, a worked calculation, the connection to its implementation, and exercises or interpretation checks. Some advanced Python files are equation labs; their companion chapters identify the additional work needed for a full trainer.

See the [requested RL syllabus](RL_COVERAGE.md) for prerequisites and topic-by-topic coverage.
