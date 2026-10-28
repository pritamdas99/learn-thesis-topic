# Requested RL syllabus: prerequisites, mathematics, and code

Start here to study the requested topics. Paths are relative to this file;
run commands from the `finetuning_and_rl` folder. Core code is offline and CPU-sized.

## Prerequisite sequence

1. Python loops, functions, lists, classes, random sampling, and numerical debugging:
   [Python and math](../01_foundations/01_python_and_math.md).
2. Sums, vectors/matrices, conditional probability, expectations, variance,
   derivatives/chain rule, softmax, logarithms, entropy and KL:
   [math primer](MATH_PRIMER.md). Practice each short calculation there.
3. Tensors, gradients, gradient descent and loss functions:
   [autograd](../01_foundations/02_tensors_autograd.md) and
   [neural network training](../01_foundations/03_neural_network_training.md).
4. For deep RL: function approximation and optimizer behavior. For TRPO add
   Hessians and solving linear systems; its chapter introduces local curvature.
5. For GRPO and self-distillation: [token/attention basics](../01_foundations/04_tokens_attention.md),
   [causal language modeling](../01_foundations/05_tiny_transformer.md), and
   [response masking](../02_finetuning/02_instruction_data_and_masks.md).

## RL setup and the difference between an MDP and RL

An MDP is a mathematical problem specification
$\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma,\rho_0)$: states, actions,
transition law, expected reward, discount, and initial-state distribution.
The Markov property is
$P(S_{t+1},r_t\mid S_0,A_0,\ldots,S_t,A_t)=P(S_{t+1},r_t\mid S_t,A_t)$.
RL is a learning framework: an agent uses experience to improve decisions,
frequently without knowing $P$ or $R$. An MDP may be solved by planning with a
known model; RL may learn a model or learn values/policies directly. A partially
observed agent generally needs history or belief state rather than the raw observation.

$$S_0\sim\rho_0,\quad A_t\sim\pi_\theta(\cdot\mid S_t),\quad
(S_{t+1},r_t)\sim p(\cdot,\cdot\mid S_t,A_t),$$
$$J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].$$

Here $r_t$ denotes the reward received after action $A_t$; some books call it
$R_{t+1}$. Define $V^\pi(s)=E[G_t\mid S_t=s]$,
$Q^\pi(s,a)=E[G_t\mid S_t=s,A_t=a]$, and $A^\pi=Q^\pi-V^\pi$.
The agent/environment loop is:

```python
state = env.reset()
while True:
    action = agent.act(state)       # exploration is part of behavior policy
    nxt, reward, terminated, truncated = env.step(action)
    agent.observe(state, action, reward, nxt, terminated, truncated)
    if terminated or truncated:
        state = env.reset()
    else:
        state = nxt
```

This is interface pseudocode; the runnable lessons define their own tiny
environments. A real Gymnasium `reset()` returns `(observation, info)` and
`step()` includes an `info` dictionary. A true terminal suppresses bootstrap;
an external continuing-task time limit bootstraps from the final observation
while stopping the trace at reset. Lesson 06 implements both masks explicitly.

Model-based means using a transition model for prediction or planning;
model-free means learning without that model. On-policy means targeting the
behavior policy; off-policy allows a different target policy. Bootstrapping
means using a learned estimate inside a target. These are separate distinctions.

## Coverage for item 2

| Requested topic | Theory and prerequisite | Runnable code |
| --- | --- | --- |
| MDP, RL setup, MDP versus RL | This guide, then [Bellman equations](../03_reinforcement_learning/02_mdp_value_iteration.md) | [corridor model](../03_reinforcement_learning/02_mdp_value_iteration.py) |
| Value iteration and policy iteration | [evaluation, improvement, convergence](../03_reinforcement_learning/02_mdp_value_iteration.md); probability and discounted return | [both algorithms](../03_reinforcement_learning/02_mdp_value_iteration.py) |
| Model-based Monte Carlo | [simulation and rollout planning](../03_reinforcement_learning/07_monte_carlo.md); sample means | [rollout planner](../03_reinforcement_learning/07_monte_carlo.py) |
| Model-free Monte Carlo | [first-visit MC control](../03_reinforcement_learning/07_monte_carlo.md); epsilon-greedy exploration | [MC control](../03_reinforcement_learning/07_monte_carlo.py) |
| SARSA and Q-learning | [on/off-policy TD updates](../03_reinforcement_learning/03_q_learning_sarsa.md); Bellman targets | [both tabular learners](../03_reinforcement_learning/03_q_learning_sarsa.py) |
| Deep Q-learning (DQN) | [replay, target networks, DQN versus Double DQN](../03_reinforcement_learning/04_dqn.md); neural nets | [DQN / Double DQN trainer](../03_reinforcement_learning/04_dqn.py) |

SARSA uses $Q\leftarrow Q+\alpha[r+\gamma Q(s',a')-Q]$ with sampled next action.
Q-learning uses $Q\leftarrow Q+\alpha[r+\gamma\max_b Q(s',b)-Q]$.
Terminal targets are $r$. DQN regresses $Q_\theta(s,a)$ toward
$y=r+\gamma(1-d)\max_bQ_{target}(s',b)$ with replay and frozen targets.
Double DQN selects $b^*=\arg\max_bQ_\theta(s',b)$ then evaluates
$Q_{target}(s',b^*)$. The default uses Double DQN; pass `--vanilla-dqn` for original DQN.

## Coverage for item 3

| Requested topic | Theory | Runnable code and scope |
| --- | --- | --- |
| Policy gradient / REINFORCE | [likelihood-ratio derivation and baseline](../03_reinforcement_learning/05_reinforce_actor_critic.md) | [complete MC-return trainer](../03_reinforcement_learning/05_reinforce_actor_critic.py) |
| A2C and A3C | [n-step advantage and synchronization](../03_reinforcement_learning/08_a2c_a3c.md) | [synchronous and concurrent trainers](../03_reinforcement_learning/08_a2c_a3c.py) |
| TRPO | [KL constraint and curvature](../03_reinforcement_learning/09_trpo.md) | [exact-bandit update lab](../03_reinforcement_learning/09_trpo.py) |
| PPO | [clipped surrogate and GAE](../03_reinforcement_learning/06_ppo_and_gae.md) | [complete toy trainer](../03_reinforcement_learning/06_ppo_and_gae.py) |
| GRPO | [group rewards and critic-free advantages](../04_alignment/04_grpo.md) | [one-token response trainer](../04_alignment/04_grpo.py) |

For a discounted episodic objective, the likelihood-ratio estimator is
$\nabla J=E[\sum_t\gamma^t\nabla\log\pi_\theta(A_t\mid S_t)(G_t-b(S_t))]$.
An action-independent baseline cancels because
$\sum_a\pi(a\mid s)\nabla\log\pi(a\mid s)=\nabla\sum_a\pi(a\mid s)=0$.
Many practical implementations, including the small teaching trainer, average
unweighted per-step terms; distinguish that convention from the exact discounted
start-state objective when deriving an estimator. PPO maximizes
$E[\min(\rho\hat A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)\hat A)]$.
GRPO constructs $\hat A_i=(r_i-\bar r)/(\sigma_r+\varepsilon)$ within each prompt's
group, then uses a related surrogate. Zero reward variance gives zero reward advantage.

## Coverage for item 4

[SDPO, RLSD, and OPSD chapter](../04_alignment/05_sdpo_rlsd_opsd.md) provides
paper-specific definitions, equations, a worked calculation, gradient handling,
and links to primary sources. [Companion code](../04_alignment/05_sdpo_rlsd_opsd.py)
implements masked reverse KL, forward KL/JSD, and RLSD token credit plus a
policy surrogate. These are objective labs, not complete LLM training pipelines.
Read GRPO and the KL sections before this chapter.

## Suggested execution order

```bash
python3 03_reinforcement_learning/02_mdp_value_iteration.py
python3 03_reinforcement_learning/07_monte_carlo.py
python3 03_reinforcement_learning/03_q_learning_sarsa.py
python3 03_reinforcement_learning/04_dqn.py
python3 03_reinforcement_learning/05_reinforce_actor_critic.py
python3 03_reinforcement_learning/08_a2c_a3c.py
python3 03_reinforcement_learning/09_trpo.py
python3 03_reinforcement_learning/06_ppo_and_gae.py
python3 04_alignment/04_grpo.py
python3 04_alignment/05_sdpo_rlsd_opsd.py
```

Mastery check: derive each target, identify which distribution generates data,
mark every detached quantity, explain terminal handling, and predict one update
by hand. Compare multiple seeds and held-out metrics before claiming improvement.
