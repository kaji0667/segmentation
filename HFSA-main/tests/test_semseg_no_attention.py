from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.nn.modules.head import TextPromptSegment


class NoSpatialAttentionHeadTest(unittest.TestCase):
    def setUp(self):
        self.head = TextPromptSegment(nc=2, hidden=8, embed_dim=4, upsample=1, text_dim=6, ch=(8, 8, 8))

    def test_attention_parameters_are_removed(self):
        self.assertFalse(hasattr(self.head, "query_proj"))
        self.assertFalse(hasattr(self.head, "key_proj"))
        self.assertFalse(hasattr(self.head, "attention_gate_weight"))
        self.assertFalse(hasattr(self.head, "attention_logit_scale"))

    def test_decoder_uses_visual_value_and_similarity_channels_only(self):
        first_decoder_conv = self.head.mask_decoder[0].conv
        self.assertEqual(first_decoder_conv.in_channels, self.head.embed_dim * 2 + 1)

    def test_forward_shape_and_gradients(self):
        features = [torch.randn(2, 8, 8, 8), torch.randn(2, 8, 4, 4), torch.randn(2, 8, 2, 2)]
        tokens = torch.randn(2, 5, 6)
        valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

        logits = self.head(features, text_embedding=tokens, text_token_mask=valid)
        logits.square().mean().backward()

        self.assertEqual(tuple(logits.shape), (2, 1, 8, 8))
        self.assertIsNotNone(self.head.similarity_gate_weight.grad)
        self.assertGreater(float(self.head.similarity_gate_weight.grad.abs()), 0.0)
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
