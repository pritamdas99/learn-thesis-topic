# Capstones and thesis experiments

Each project should produce runnable code, a configuration file, a data/split
manifest, an evaluation table, and a short report of failures and limitations.
Start with the smallest version that answers your question.

## 1. Full fine-tuning versus LoRA

Question: when does a higher adapter rank improve held-out task performance?
Build a small domain-specific instruction dataset with non-overlapping source
splits. Compare the unchanged base, full tuning, and LoRA ranks 2/8/16.
Keep training data, token budget, and evaluation prompts fixed. Report task
accuracy, validation loss, trainable parameters, time, peak memory, and source
capability retention. Run multiple seeds. Do not infer quality from loss alone.

Stretch: repeat with QLoRA and separate numerical compression effects from
changes in hardware, batch size, and optimizer.

## 2. DQN versus PPO

Question: how do sample efficiency and stability differ on a small discrete task?
First use the chain environment. Then implement a Gymnasium adapter and move
to a task such as CartPole. Preserve termination/truncation semantics. Compare
at equal environment steps, log learning curves, and evaluate multiple seeds.

Ablations: DQN without replay, DQN without a target network, PPO without entropy,
and PPO with different rollout/update ratios. Include a random-action baseline.

## 3. DPO under preference noise

Question: how robust is preference optimization when pair labels are corrupted?
Begin with the categorical DPO lesson. Generate train/validation/test preference
pairs from a known utility. Flip 0/10/30 percent of TRAINING labels only. Sweep
beta using validation data. Measure held-out preference accuracy, KL to the
reference, and whether chosen and rejected probabilities move as expected.

Stretch: use a small language model and examine response-length confounding.

## 4. GRPO reward design

Question: what happens when a reward is easy to game?
Expand the one-token task into generated arithmetic answers. Compare exact
correctness rewards with a flawed format/length-based reward. Keep a separate
correctness evaluator. Measure true accuracy, reward, diversity, response length,
and the fraction of groups with zero reward variance.

Stretch: compare group sizes at equal TOTAL generated-token budgets.

## 5. Offline RL with missing action coverage

Question: how does dataset coverage change the usefulness of pessimism?
Collect fixed chain transitions with different behavior policies. Compare
behavior cloning, ordinary offline Q-learning, and the discrete CQL penalty.
Use held-out environment evaluation without adding those transitions to training.
Plot return against coverage and conservative penalty coefficient.

## A report template

- Research question and testable hypothesis.
- Baselines and why they are appropriate.
- Data provenance, split policy, and leakage checks.
- Exact configuration, versions, hardware, seeds and compute budget.
- Evaluation protocol with uncertainty and paired comparisons where applicable.
- Results, representative failures, and alternative explanations.
- Limitations and the next experiment that could disprove your interpretation.

These are starting experiments, not claims of novel thesis contributions.
Check the literature before describing any result as new.
