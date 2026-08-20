from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.nn.modules.head import TextPromptSegment


class TextTokenPoolingTest(unittest.TestCase):
    def setUp(self):
        self.head = TextPromptSegment(nc=2, hidden=8, embed_dim=8, upsample=1, text_dim=4, ch=(8, 8, 8))

    def test_zero_initialization_matches_original_mean_pooling(self):
        tokens = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0], [100.0, 100.0, 100.0, 100.0]]]
        )
        valid = torch.tensor([[True, True, False]])

        pooled = self.head._pool_text_tokens(tokens, text_token_mask=valid)

        torch.testing.assert_close(pooled, tokens.mean(1))

    def test_valid_token_bias_can_suppress_padding(self):
        tokens = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]]])
        valid = torch.tensor([[True, True, False]])
        self.head.valid_token_bias.data.fill_(20.0)

        pooled = self.head._pool_text_tokens(tokens, text_token_mask=valid)

        torch.testing.assert_close(pooled, tokens[:, :2].mean(1), rtol=1e-4, atol=1e-4)

    def test_forward_backpropagates_into_token_pooling(self):
        features = [torch.randn(2, 8, 8, 8), torch.randn(2, 8, 4, 4), torch.randn(2, 8, 2, 2)]
        tokens = torch.randn(2, 5, 4)
        valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

        logits = self.head(
            features,
            text_embedding=tokens,
            text_token_mask=valid,
        )
        logits.mean().backward()

        self.assertEqual(tuple(logits.shape), (2, 1, 8, 8))
        self.assertIsNotNone(self.head.text_token_score.weight.grad)
        self.assertIsNotNone(self.head.valid_token_bias.grad)
        self.assertTrue(torch.isfinite(self.head.text_token_score.weight.grad).all())
        self.assertTrue(torch.isfinite(self.head.valid_token_bias.grad).all())
        self.assertGreater(float(self.head.valid_token_bias.grad.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
