# Practical tricks, with the reason behind each one

## Before training

- Overfit 8–32 examples first. Failure usually means the data, loss, mask, or
  optimizer is wrong; a larger GPU rarely fixes it.
- Print decoded inputs and supervised labels. Check EOS, padding, truncation,
  template boundaries, and whether any response tokens survive.
- Split by source/document/user before chunking. Overlapping windows from the
  same document in both splits can create misleading validation scores.
- Record model revision, tokenizer, template, dataset hash, seed, library
  versions and configuration. A seed alone does not guarantee reproducibility.
- Start with a simple baseline and a held-out evaluation set. For RL include a
  random policy; for offline RL include behavior cloning.

## When loss behaves badly

| Symptom | Likely checks |
| --- | --- |
| Loss is NaN | Empty supervised masks, invalid log values, excessive LR, FP16 overflow |
| Loss never changes | Frozen parameters, missing backward/step, detached computation |
| Loss falls but generation is bad | Leakage, wrong template, tiny/poor data, wrong task metric |
| CUDA out of memory | Sequence length, activation size, batch size, optimizer state |
| Training resumes differently | Missing RNG/sampler/optimizer/scheduler state |
| LoRA has no effect | Target module names, adapter loading, actual trainable parameters |
| RL reward suddenly collapses | Update size, stale rollouts, incorrect advantage/bootstrap masks |
| High reward but poor behavior | Reward hacking, verifier bugs, evaluation mismatch |

## Fine-tuning techniques

Use smaller learning rates when modifying pretrained weights. Warmup avoids
large early updates; clipping limits large gradients. Neither replaces correct
loss scaling. Select learning rate on validation, not a fixed universal rule.

Try data quality and coverage before more epochs. Monitor forgetting on the
original capabilities as well as target performance. Increase LoRA rank only
when the adaptation capacity appears limiting; compare under a fixed budget.

Use gradient accumulation for larger effective batches, mixed precision for
appropriate hardware, and activation checkpointing for long sequences.
A 4-bit base still has floating-point activations and adapter optimizer state.
Sequence length can dominate memory even when weights are heavily compressed.

## RL techniques

Separate training exploration from evaluation. Evaluate without updates and
with a defined action-selection rule. Report multiple training seeds and
several evaluation episodes per trained agent.

Keep rewards and observations at manageable scales. Fit normalization using
training experience and freeze it during evaluation. Reward normalization and
reward clipping change optimization; reward clipping can change the task.

For Gymnasium, step returns `(observation, reward, terminated, truncated, info)`.
Reset after either end flag. Bootstrap across a time-limit truncation when the
underlying task continues, using the final observation before reset; stop the
GAE trace at both kinds of episode boundary. See the [official API guide](https://gymnasium.farama.org/introduction/basic_usage/).

For PPO, log entropy, value loss, approximate KL, clip fraction, and return.
For DQN/SAC, inspect target values, replay diversity, and update-to-data ratio.
Small surrogate loss alone does not prove the policy is improving.

## Honest experiments

Change one factor at a time. Compare at equal data/compute budgets and keep a
final test set untouched. Report failures and confidence intervals along with
averages. For repeated tasks from the same source, resample at the source level.
