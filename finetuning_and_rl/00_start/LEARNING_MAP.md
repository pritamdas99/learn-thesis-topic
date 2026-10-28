# A practical learning map

Every lesson now has a matching `.md` explanation and `.py` implementation.
Use the [paired index](THEORY_INDEX.md) to read theory before code.
For notation and mathematical prerequisites, see the [math primer](MATH_PRIMER.md).

Prerequisites: basic computer use and willingness to read Python. The first
lesson introduces lists, functions, loops, dot products, softmax and derivatives;
if those are completely new, pause and practice each with your own numbers.
You do not need to know deep learning before starting.

## Stage 1: Learn what training actually does

Read `01_foundations/01` through `03`. Understand a prediction, loss, derivative,
parameter update, minibatch, and held-out evaluation. Derive the two gradients
in linear regression by hand. Explain why removing all nonlinearities makes a
multilayer network equivalent to one linear map.

Checkpoint: fit a different synthetic function and explain training versus
validation performance without using the word “magic.”

## Stage 2: Understand language models

Read foundations `04` and `05`. Trace text -> token IDs -> embeddings -> causal
attention -> logits -> next-token loss. Track [batch, sequence, vocabulary]
shapes. Distinguish pretraining from fine-tuning and inference.

Checkpoint: explain why a causal model cannot see tomorrow's target token,
why generation repeatedly calls the model, and what temperature changes.

## Stage 3: Fine-tune responsibly and efficiently

Read all files in `02_finetuning`. Start with a source/target task, then learn
response-only supervision and LoRA algebra. Use the external model workflow
only after understanding the offline versions. Read the saved adapter config
and identify which original base weights are needed for reload.

Checkpoint: explain the difference between the attention mask and label mask;
show that frozen parameters do not change; compare full tuning and LoRA at the
same evaluation budget. QLoRA means a quantized frozen base plus trainable adapters.

## Stage 4: Learn RL from small environments

Read `03_reinforcement_learning` in order. Distinguish reward (one step) from
return (discounted future sum). Compute a Bellman backup by hand. Move from a
table to a neural Q function, then to an explicitly stochastic policy.

Checkpoint: explain on-policy versus off-policy, why replay helps DQN but stale
replay is problematic for ordinary PPO, and why termination differs from a
collector's time limit. Explain why GAE uses two different masks.

## Stage 5: Connect LLM fine-tuning to preference optimization

Read `04_alignment`. SFT imitates demonstrations. A reward model learns pairwise
preferences. RLHF optimizes generated behavior against rewards and a reference
constraint. DPO learns directly from fixed preference pairs. GRPO compares
multiple sampled answers to the same prompt without a learned value critic.

Checkpoint: trace which tensors receive gradients. Explain the separate roles
of the reference policy and the old rollout policy. They are NOT interchangeable.
The alignment examples use tiny feature/categorical problems to expose the
objectives; they do not implement production LLM alignment pipelines.

## Stage 6: Study advanced objectives and systems

Read `05_advanced`. Reproduce each equation before extending it. SAC and CQL are
explicit equation labs; extend them into full agents only after understanding
replay collection, target updates, evaluation, and environment action bounds.
Read the roadmap below for subjects that require separate larger projects.

Checkpoint: distinguish memory for parameters, gradients, optimizer states, and
activations. Explain how mixed precision, LoRA, checkpointing, DDP and FSDP
address different parts of the problem.

## Further advanced topics (reading/research extensions, not implemented here)

| Topic | Central idea | Useful next experiment |
| --- | --- | --- |
| TRPO | Constrain policy updates using a KL-based trust region | Compare measured KL with PPO clipping |
| TD3 | Twin critics, delayed actor updates, target action noise | Compare with SAC on bounded continuous actions |
| Distributional RL | Learn a return distribution rather than just its mean | Implement categorical return projection |
| Prioritized replay | Sample important transitions, correct sampling bias | Compare TD-error sampling with uniform replay |
| n-step returns | Trade bootstrap bias against return variance | Sweep horizon at fixed environment steps |
| Model-based RL | Learn dynamics and plan imagined futures | Measure compounding model error |
| IQL / offline RL | Learn without querying unsupported actions | Compare offline policy with behavior cloning |
| Contextual bandits | Choose actions conditional on features | Add observations to the bandit lesson |
| Multi-agent RL | Other learning agents make the environment nonstationary | Start with cooperative matrix games |
| Hierarchical RL | Learn temporally extended skills/options | Separate navigation subgoals from control |
| RNN policies / POMDPs | Infer hidden state from history | Hide velocity in a control environment |
| Curriculum / sparse rewards | Improve learning signal and task sequence | Compare shaping with unmodified task return |
| Prefix/prompt tuning, adapters, DoRA | Alternative parameter-efficient updates | Compare parameter and activation budgets |
| FSDP / ZeRO | Shard weights, gradients, and/or optimizer states | Profile peak memory across ranks |
| Sequence packing / attention kernels | Reduce padding and attention overhead | Ensure packed documents have correct boundaries |
| Long-context tuning / RoPE | Adapt positional behavior and context usage | Evaluate retrieval across position and length |
| Continual learning | Adapt while limiting catastrophic forgetting | Measure both source and target tasks |
| RLAIF, process rewards, verifiable rewards | Change feedback source or granularity | Audit verifier errors and length shortcuts |
| Preference variants (IPO, KTO, ORPO) | Change preference assumptions/objectives | Derive one objective and test label noise |
| Tool-using agents | Actions may call tools over multiple turns | Define tool costs, observations, and termination |

Choose one capstone in `06_projects/PROJECTS.md`. A narrow, controlled study is
more useful than running many methods without understanding their comparisons.

## Requested RL sequence

Use the [RL syllabus guide](RL_COVERAGE.md) for MDP/RL setup, policy iteration, Monte Carlo, A2C/A3C, TRPO, PPO/GRPO, and SDPO/RLSD/OPSD with prerequisite links and code.
