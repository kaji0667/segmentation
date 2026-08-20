from pathlib import Path
import math
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.nn.modules.head import TextPromptSegment


class SpatialAttentionMapTest(unittest.TestCase):
    def setUp(self):
        self.head = TextPromptSegment(nc=2, hidden=8, embed_dim=2, upsample=1, text_dim=4, ch=(8, 8, 8))

    @staticmethod
    def _focused_key_query():
        key = torch.tensor(
            [[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]],
            dtype=torch.float32,
        )
        query = torch.tensor([[[1.0], [0.0]]], dtype=torch.float32)
        return key, query

    def test_uniform_attention_maps_to_zero(self):
        key = torch.zeros(1, 2, 3, 3)
        query = torch.tensor([[[1.0], [0.0]]])

        attention = self.head._build_spatial_attention_map(key, query)

        torch.testing.assert_close(attention, torch.zeros_like(attention), atol=1e-7, rtol=0.0)

    def test_attention_is_bounded_and_highlights_matching_pixel(self):
        key, query = self._focused_key_query()

        attention = self.head._build_spatial_attention_map(key, query)

        self.assertEqual(tuple(attention.shape), (1, 1, 2, 2))
        self.assertGreater(float(attention[0, 0, 0, 0]), 0.0)
        self.assertLess(float(attention[0, 0, 0, 1]), 0.0)
        self.assertLessEqual(float(attention.max()), 1.0)
        self.assertGreaterEqual(float(attention.min()), -1.0)

    def test_higher_temperature_increases_attention_contrast(self):
        key, query = self._focused_key_query()
        base = self.head._build_spatial_attention_map(key, query)
        self.head.attention_logit_scale.data.fill_(math.log(8.0))

        sharper = self.head._build_spatial_attention_map(key, query)

        self.assertGreater(float(sharper.max() - sharper.min()), float(base.max() - base.min()))

    def test_forward_backpropagates_into_attention_temperature(self):
        features = [torch.randn(2, 8, 8, 8), torch.randn(2, 8, 4, 4), torch.randn(2, 8, 2, 2)]
        tokens = torch.randn(2, 5, 4)
        valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

        logits = self.head(features, text_embedding=tokens, text_token_mask=valid)
        logits.square().mean().backward()

        self.assertEqual(tuple(logits.shape), (2, 1, 8, 8))
        self.assertIsNotNone(self.head.attention_logit_scale.grad)
        self.assertTrue(torch.isfinite(self.head.attention_logit_scale.grad))
        self.assertGreater(float(self.head.attention_logit_scale.grad.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
