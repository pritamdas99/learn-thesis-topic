"""Numerical checks of important concepts, beyond whether scripts merely run.
Run: python -m unittest discover -s tests -v
"""
import importlib.util
from pathlib import Path
import unittest
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]


def lesson(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LearningInvariants(unittest.TestCase):
    def test_causal_attention_cannot_see_future(self):
        module = lesson('01_foundations/04_tokens_attention.py')
        x = torch.randn(4, 8)
        before, _ = module.causal_attention(x)
        x[3] += 100
        after, _ = module.causal_attention(x)
        torch.testing.assert_close(before[:3], after[:3])

    def test_transformer_causality(self):
        module = lesson('01_foundations/05_tiny_transformer.py')
        model = module.TinyLM(10).eval()
        with torch.no_grad():
            before = model(torch.tensor([[1, 2, 3, 4]]))
            after = model(torch.tensor([[1, 2, 3, 9]]))
        torch.testing.assert_close(before[:, :3], after[:, :3])

    def test_prompt_and_padding_do_not_contribute_to_loss(self):
        module = lesson('02_finetuning/02_instruction_data_and_masks.py')
        ids, labels, mask = module.make_batch([('hello', 'a'), ('a longer prompt', 'bc')])
        logits = torch.randn(*ids.shape, 258, requires_grad=True)
        module.response_loss(logits, labels).backward()
        ignored = labels[:, 1:].eq(-100)
        self.assertTrue(torch.all(logits.grad[:, :-1][ignored] == 0))
        self.assertTrue(torch.all(labels[~mask] == -100))
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_lora_initialization_and_merge(self):
        module = lesson('02_finetuning/03_lora_from_scratch.py')
        base = torch.nn.Linear(8, 4)
        adapter = module.LoRALinear(base)
        x = torch.randn(3, 8)
        torch.testing.assert_close(adapter(x), base(x))
        with torch.no_grad():
            adapter.b.normal_()
        torch.testing.assert_close(adapter(x), x@adapter.merged_weight().T+base.bias)
        self.assertFalse(base.weight.requires_grad)

    def test_bellman_values(self):
        module = lesson('03_reinforcement_learning/02_mdp_value_iteration.py')
        self.assertAlmostEqual(module.value_iteration()[2], 0.95)

    def test_policy_iteration_matches_bellman_solution(self):
        module = lesson('03_reinforcement_learning/02_mdp_value_iteration.py')
        values, policy = module.policy_iteration()
        self.assertEqual(policy[1:4], [1, 1, 1])
        for actual, expected in zip(values, module.value_iteration()):
            self.assertAlmostEqual(actual, expected)

    def test_trpo_accepts_improving_step_inside_actual_kl_bound(self):
        module = lesson('03_reinforcement_learning/09_trpo.py')
        theta = torch.zeros(3, dtype=torch.float64)
        rewards = torch.tensor([0., 0.3, 1.], dtype=torch.float64)
        updated, kl, accepted = module.trpo_step(theta, rewards)
        self.assertTrue(accepted)
        self.assertLessEqual(kl, 0.01)
        self.assertGreater((updated.softmax(-1)*rewards).sum().item(),
                           (theta.softmax(-1)*rewards).sum().item())

    def test_distillation_masks_and_teacher_stop_gradient(self):
        module = lesson('04_alignment/05_sdpo_rlsd_opsd.py')
        for kind in ('sdpo', 'opsd', 'jsd'):
            student = torch.zeros(1, 2, 3, requires_grad=True)
            teacher = torch.tensor([[[0., 1., 2.], [2., 1., 0.]]], requires_grad=True)
            loss = module.distillation_loss(student, teacher, torch.tensor([[1., 0.]]), kind)
            loss.backward()
            self.assertIsNone(teacher.grad)
            self.assertEqual(student.grad[0, 1].abs().sum().item(), 0.)
            self.assertGreater(student.grad[0, 0].abs().sum().item(), 0.)
            same = module.distillation_loss(student.detach(), student.detach(),
                                             torch.ones(1, 2), kind)
            self.assertAlmostEqual(same.item(), 0., places=6)

    def test_rlsd_preserves_sign_and_recovers_uniform_credit(self):
        module = lesson('04_alignment/05_sdpo_rlsd_opsd.py')
        sequence = torch.tensor([1., -1., 0.])
        teacher = torch.full((3, 2), 0.4).log().requires_grad_()
        old = torch.full((3, 2), 0.2).log().requires_grad_()
        advantages = module.rlsd_advantages(sequence, teacher, old)
        torch.testing.assert_close(advantages, torch.tensor([[1.1, 1.1], [-0.9, -0.9], [0., 0.]]))
        self.assertFalse(advantages.requires_grad)
        uniform = module.rlsd_advantages(sequence, teacher, old, mixing=0.)
        torch.testing.assert_close(uniform, sequence[:, None].expand(-1, 2))

    def test_returns(self):
        module = lesson('03_reinforcement_learning/05_reinforce_actor_critic.py')
        torch.testing.assert_close(module.discounted_returns([0., 0., 1.], 0.5),
                                   torch.tensor([0.25, 0.5, 1.]))

    def test_gae_terminal_does_not_bootstrap(self):
        module = lesson('03_reinforcement_learning/06_ppo_and_gae.py')
        advantages, returns = module.gae(torch.tensor([1.]), torch.tensor([2.]),
                                         torch.tensor([100.]), torch.ones(1), torch.ones(1))
        torch.testing.assert_close(returns, torch.tensor([1.]))

    def test_gae_truncation_bootstraps_but_stops_trace(self):
        module = lesson('03_reinforcement_learning/06_ppo_and_gae.py')
        adv, _ = module.gae(torch.tensor([1., 100.]), torch.zeros(2),
                             torch.tensor([2., 0.]), torch.tensor([0., 1.]),
                             torch.ones(2), gamma=0.5)
        self.assertAlmostEqual(adv[0].item(), 2.0)  # 1+0.5*2, no next-episode reward.

    def test_dpo_prefers_winner(self):
        module = lesson('04_alignment/03_dpo.py')
        winner = torch.tensor(0., requires_grad=True)
        loser = torch.tensor(0., requires_grad=True)
        loss = module.dpo_loss(winner, loser, torch.tensor(0.), torch.tensor(0.))
        loss.backward()
        self.assertLess(winner.grad.item(), 0)
        self.assertGreater(loser.grad.item(), 0)

    def test_sequence_logprob_excludes_prompt_and_padding(self):
        module = lesson('04_alignment/03_dpo.py')
        logs = torch.zeros(1, 4, 2)
        tokens = torch.tensor([[0, 1, 1, 0]])
        mask = torch.tensor([[0., 0., 1., 0.]])
        result = module.sequence_logprob(logs, tokens, mask)
        torch.testing.assert_close(result, torch.tensor([-0.69314718]))

    def test_equal_grpo_rewards_have_zero_advantage(self):
        module = lesson('04_alignment/04_grpo.py')
        torch.testing.assert_close(module.group_advantages(torch.ones(2, 8)), torch.zeros(2, 8))

    def test_cql_pushes_down_unsupported_actions(self):
        module = lesson('05_advanced/03_offline_rl_cql.py')
        q = torch.tensor([[0., 5.]], requires_grad=True)
        module.conservative_penalty(q, torch.tensor([0])).backward()
        self.assertGreater(q.grad[0, 1].item(), 0)
        self.assertLess(q.grad[0, 0].item(), 0)

    def test_eos_is_not_masked_when_used_as_pad_id(self):
        module = lesson('02_finetuning/04_huggingface_sft.py')
        batch = module.collate([{'input_ids': [1, 9], 'labels': [-100, 9]},
                                {'input_ids': [2, 3, 9], 'labels': [-100, 3, 9]}], pad_id=9)
        self.assertEqual(batch['attention_mask'][0].tolist(), [1, 1, 0])
        self.assertEqual(batch['labels'][0].tolist(), [-100, 9, -100])

if __name__ == '__main__':
    unittest.main()
