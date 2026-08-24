from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.nn.modules.head import TextPromptSegment


class SemanticRolePoolingTest(unittest.TestCase):
    def setUp(self):
        self.head = TextPromptSegment(nc=2, hidden=8, embed_dim=4, upsample=1, text_dim=6, ch=(8, 8, 8))

    def test_zero_initialization_matches_shared_token_pooling(self):
        tokens = torch.randn(2, 5, 6)
        valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

        shared = self.head._pool_text_tokens(tokens, text_token_mask=valid)
        roles = self.head._pool_text_roles(tokens, text_token_mask=valid)

        self.assertEqual(tuple(roles.shape), (2, 3, 6))
        for role in range(3):
            torch.testing.assert_close(roles[:, role], shared)

    def test_role_scorers_can_select_different_tokens(self):
        tokens = torch.tensor([[[2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        self.head.role_token_score.weight.data.zero_()
        self.head.role_token_score.weight.data[0, 0] = 5.0
        self.head.role_token_score.weight.data[1, 0] = -5.0

        roles = self.head._pool_text_roles(tokens)

        self.assertGreater(float(roles[0, 0, 0]), 1.9)
        self.assertLess(float(roles[0, 1, 0]), -1.9)
        torch.testing.assert_close(roles[0, 2], tokens.mean(1)[0])

    def test_forward_routes_gradients_through_all_roles(self):
        self.head.position_gate_weight.data.fill_(0.1)
        features = [torch.randn(2, 8, 8, 8), torch.randn(2, 8, 4, 4), torch.randn(2, 8, 2, 2)]
        tokens = torch.randn(2, 5, 6)
        valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

        logits = self.head(features, text_embedding=tokens, text_token_mask=valid)
        logits.square().mean().backward()

        self.assertEqual(tuple(logits.shape), (2, 1, 8, 8))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsNotNone(self.head.role_token_score.weight.grad)
        self.assertIsNotNone(self.head.role_valid_token_bias.grad)
        self.assertIsNotNone(self.head.position_text_proj.weight.grad)
        self.assertIsNotNone(self.head.position_gate_weight.grad)
        role_grad = self.head.role_token_score.weight.grad.abs().sum(dim=1)
        self.assertTrue(torch.all(role_grad > 0))


if __name__ == "__main__":
    unittest.main()
