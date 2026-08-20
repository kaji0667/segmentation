from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.rrsisd_refseg_dataset import RRSISDRefSegDataset, _directional_axes


class AxisAwareAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        self.mask = np.arange(6, dtype=np.uint8).reshape(2, 3)

    def _dataset(self, policy: str = "axis-aware") -> RRSISDRefSegDataset:
        return RRSISDRefSegDataset(
            [],
            augment=True,
            hflip_prob=1.0,
            vflip_prob=1.0,
            color_jitter=0.0,
            directional_flip_policy=policy,
        )

    def test_direction_words_are_classified_by_axis(self):
        self.assertEqual(_directional_axes("the ship on the left"), (True, False))
        self.assertEqual(_directional_axes("the vehicle above the bridge"), (False, True))
        self.assertEqual(_directional_axes("the field below and to the right"), (True, True))
        self.assertEqual(_directional_axes("the object in the center"), (False, False))

    def test_horizontal_word_blocks_only_horizontal_flip(self):
        image, mask = self._dataset()._augment(self.image, self.mask, "the ship on the left")
        np.testing.assert_array_equal(image, np.flip(self.image, axis=0))
        np.testing.assert_array_equal(mask, np.flip(self.mask, axis=0))

    def test_above_blocks_only_vertical_flip(self):
        image, mask = self._dataset()._augment(self.image, self.mask, "the vehicle above the bridge")
        np.testing.assert_array_equal(image, np.flip(self.image, axis=1))
        np.testing.assert_array_equal(mask, np.flip(self.mask, axis=1))

    def test_legacy_policy_still_blocks_both_axes(self):
        image, mask = self._dataset("legacy")._augment(self.image, self.mask, "the ship on the left")
        np.testing.assert_array_equal(image, self.image)
        np.testing.assert_array_equal(mask, self.mask)

    def test_mask_size_is_aligned_to_image_size(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        mask = np.zeros((5, 4), dtype=np.uint8)
        mask[1:4, 1:3] = 1

        aligned = self._dataset()._align_mask_to_image(image, mask)

        self.assertEqual(aligned.shape, (3, 4))
        self.assertTrue(set(np.unique(aligned)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
