from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.nn.modules.head import TextPromptSegment


class _ConstantDecoder(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, decoder_input):
        return decoder_input.new_full((decoder_input.shape[0], 1, *decoder_input.shape[-2:]), self.value)


class TargetBackgroundDecoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.head = TextPromptSegment(nc=2, hidden=8, embed_dim=4, upsample=1, text_dim=6, ch=(8, 8, 8))
        self.features = [
            torch.randn(2, 8, 8, 8),
            torch.randn(2, 8, 4, 4),
            torch.randn(2, 8, 2, 2),
        ]
        self.tokens = torch.randn(2, 5, 6)

    def test_decoders_are_parameter_independent(self):
        target_parameters = {id(parameter) for parameter in self.head.target_decoder.parameters()}
        background_parameters = {id(parameter) for parameter in self.head.background_decoder.parameters()}

        self.assertTrue(target_parameters)
        self.assertTrue(background_parameters)
        self.assertTrue(target_parameters.isdisjoint(background_parameters))

    def test_forward_shape_and_both_decoder_gradients(self):
        logits = self.head(self.features, text_embedding=self.tokens)
        logits.square().mean().backward()

        self.assertEqual(tuple(logits.shape), (2, 1, 8, 8))
        for decoder in (self.head.target_decoder, self.head.background_decoder):
            gradients = [parameter.grad for parameter in decoder.parameters()]
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_target_logits_are_subtracted_from_background_logits_with_correct_sign(self):
        self.head.target_decoder = _ConstantDecoder(2.0)
        self.head.background_decoder = _ConstantDecoder(0.0)
        target_positive = self.head(self.features, text_embedding=self.tokens)

        self.head.target_decoder = _ConstantDecoder(0.0)
        self.head.background_decoder = _ConstantDecoder(2.0)
        background_positive = self.head(self.features, text_embedding=self.tokens)

        torch.testing.assert_close(target_positive - background_positive, torch.full_like(target_positive, 4.0))


if __name__ == "__main__":
    unittest.main()
