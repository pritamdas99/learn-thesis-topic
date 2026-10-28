# Fine-tuning and reinforcement learning — theory, mathematics, and code

For the requested RL syllabus, start with the [RL coverage and prerequisites guide](00_start/RL_COVERAGE.md).

Start with the [theory and code index](00_start/THEORY_INDEX.md), then follow
[the learning map](00_start/LEARNING_MAP.md). Every one of the 31 lesson scripts
has a matching Markdown chapter beside it: for example,
[`06_ppo_and_gae.md`](03_reinforcement_learning/06_ppo_and_gae.md) explains the
mathematics behind [`06_ppo_and_gae.py`](03_reinforcement_learning/06_ppo_and_gae.py).

Read the chapter first for intuition, definitions, derivations, worked numerical
examples, and a map from equations to code. Then run and modify the Python file.
Start with the [math primer](00_start/MATH_PRIMER.md) if derivatives, expectations,
entropy, or KL divergence are unfamiliar.

This is a broad beginner-to-advanced foundation, not an exhaustive treatment of
every research method. Complete trainers and isolated equation labs are clearly
identified. Small toy successes are not evidence of real-world model quality.

## Setup

Use Python 3.11 or newer. Run these commands in this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 01_foundations/01_python_and_math.py
python 01_foundations/02_tensors_autograd.py
```

On Windows, activate with `.venv\Scripts\activate` instead. For a CPU-only
PyTorch installation on Linux/Windows, you can use
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
For CUDA, follow the [PyTorch installation selector](https://pytorch.org/get-started/locally/).
No GPU or account is needed for the core lessons. The first Python/math lesson,
bandits, value iteration, Q-learning/SARSA, and bootstrap evaluation need only Python.

## Reading order

| Folder | What you learn |
| --- | --- |
| [00_start](00_start/LEARNING_MAP.md) | Study order, terminology, experiments, troubleshooting |
| [01_foundations](01_foundations/01_python_and_math.md) | Python/math, tensors, autograd, neural networks, attention, tiny transformer |
| [02_finetuning](02_finetuning/01_transfer_learning.md) | Transfer learning, SFT labels, LoRA, real Hugging Face training, reload/inference |
| [03_reinforcement_learning](03_reinforcement_learning/01_bandits.md) | Bandits, MDPs, value/policy iteration, Monte Carlo, SARSA/Q-learning, DQN, policy gradients, A2C/A3C, TRPO, PPO/GAE |
| [04_alignment](04_alignment/01_reward_model.md) | Preference reward models, RLHF objective, DPO, GRPO, SDPO/RLSD/OPSD |
| [05_advanced](05_advanced/01_training_efficiency.md) | Efficient training, SAC, offline CQL, distillation, quantization, uncertainty, checkpointing, DDP |
| [06_projects](06_projects/PROJECTS.md) | Capstone experiments and thesis directions |

Read matching `.md` and `.py` files together in numerical order within each folder. The real-model scripts are
optional: you can continue into RL without downloading any pretrained model.

## Run and check the offline lessons

```bash
python run_lessons.py --list
python run_lessons.py --stdlib-only
python run_lessons.py
python -m unittest discover -s tests -v
```

The runner excludes external model downloads and the multi-process DDP lesson.
It runs each lesson separately with a timeout and reports failures. To run DDP:

```bash
torchrun --standalone --nproc_per_node=2 05_advanced/07_distributed_training.py
```

## Fine-tune an actual pretrained model

```bash
python -m pip install -r requirements-huggingface.txt
python 02_finetuning/04_huggingface_sft.py --model sshleifer/tiny-gpt2 --steps 5
python 02_finetuning/05_reload_and_generate.py
# A larger pretrained base, still small by LLM standards:
python 02_finetuning/04_huggingface_sft.py --model distilgpt2 --method lora --steps 20
# Full parameter fine-tuning:
python 02_finetuning/04_huggingface_sft.py --method full --output outputs/full
```

The tiny GPT-2 option checks the pipeline; expect poor generated text. Even
DistilGPT-2 with eight teaching examples will not become a useful assistant.
First runs require internet and disk space for model downloads. CPU training of
the larger model is slower and requires more RAM. Outputs are saved locally,
never uploaded. Select a new `--output` directory to preserve previous runs.

For the optional NVIDIA GPU QLoRA branch, install `bitsandbytes`, then add
`--qlora --method lora`. That branch is hardware-dependent. Larger model choices
need a compatible tokenizer/template and enough GPU memory. QLoRA support is
based on the [Transformers quantization documentation](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes).

## How to study a file

1. Read the matching Markdown chapter and derive its worked example by hand.
2. Open the Python file and connect each main operation to its equation.
3. Predict the tensor shapes and output before running it.
4. Run once, then change one hyperparameter and record the result.
5. Complete the exercise in the comments. Explain why the result changed.
6. Reimplement the central equation without looking, then compare.

Keep an experiment journal: hypothesis, seed, configuration, data split,
metrics, failure cases, and conclusion. Consult [tricks and debugging](00_start/TRICKS_AND_DEBUGGING.md),
[the glossary](00_start/GLOSSARY.md), and [primary sources](00_start/SOURCES.md).
See [verification details](VERIFICATION.md) for what was actually executed.
