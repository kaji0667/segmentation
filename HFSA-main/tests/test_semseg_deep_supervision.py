import unittest
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.nn.modules.head import TextPromptSegment
from ultralytics.utils.loss import SemanticSegmentationLoss


class _LossConfig(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.loss_pos_weight_max = 10.0
        self.loss_small_target_weight = 1.5
        self.loss_small_target_area = 0.0025
        self.loss_tversky_fp_weight = 0.5
        self.loss_fp_weight = 0.0
        self.loss_aux_p3_weight = 0.2
        self.loss_aux_p4_weight = 0.1


class DeepSupervisionTest(unittest.TestCase):
    def test_training_auxiliary_outputs_and_eval_main_output(self):
        head = TextPromptSegment(
            nc=20,
            hidden=16,
            embed_dim=8,
            upsample=8,
            text_dim=12,
            ch=(16, 32, 64),
        )
        head.deep_supervision_enabled = True
        features = (
            torch.randn(2, 16, 8, 8),
            torch.randn(2, 32, 4, 4),
            torch.randn(2, 64, 2, 2),
        )
        text = torch.randn(2, 12)

        head.train()
        outputs = head(features, text_embedding=text)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(tuple(outputs[0].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(outputs[1].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(outputs[2].shape), (2, 1, 4, 4))

        masks = torch.zeros(2, 64, 64, dtype=torch.long)
        masks[0, 4:8, 4:8] = 1
        masks[1, 20:44, 16:48] = 1
        criterion = SemanticSegmentationLoss(_LossConfig())
        loss, _ = criterion(outputs, {"mask": masks})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(head.aux_mask_decoder[0][-1].weight.grad)
        self.assertIsNotNone(head.aux_mask_decoder[1][-1].weight.grad)

        head.eval()
        with torch.no_grad():
            prediction = head(features, text_embedding=text)
        self.assertIsInstance(prediction, torch.Tensor)
        self.assertEqual(tuple(prediction.shape), (2, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
