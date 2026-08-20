from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.rrsisd_refseg_dataset import _load_text_embeddings
from train_semseg import predict_with_optional_text


class UnusedTextInputCleanupTest(unittest.TestCase):
    def test_legacy_role_masks_are_ignored_when_loading_cache(self):
        payload = {
            "ids": ["sample-1"],
            "embeddings": torch.randn(1, 5, 4),
            "token_masks": torch.tensor([[True, True, True, False, False]]),
            "object_token_masks": torch.ones(1, 5, dtype=torch.bool),
            "spatial_token_masks": torch.ones(1, 5, dtype=torch.bool),
            "context_token_masks": torch.ones(1, 5, dtype=torch.bool),
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "cache.pt"
            torch.save(payload, cache_path)
            loaded = _load_text_embeddings(cache_path)["sample-1"]

        self.assertEqual(set(loaded), {"text_embedding", "text_token_mask"})

    def test_prediction_routes_only_the_active_text_inputs(self):
        class DummyModel:
            def __init__(self):
                self.kwargs = None

            def __call__(self, image, **kwargs):
                self.kwargs = kwargs
                return image

        model = DummyModel()
        batch = {
            "img": torch.randn(1, 3, 8, 8),
            "class_idx": torch.tensor([1]),
            "text_embedding": torch.randn(1, 5, 4),
            "text_token_mask": torch.ones(1, 5, dtype=torch.bool),
            "text_object_mask": torch.ones(1, 5, dtype=torch.bool),
            "text_spatial_mask": torch.ones(1, 5, dtype=torch.bool),
            "text_context_mask": torch.ones(1, 5, dtype=torch.bool),
        }

        predict_with_optional_text(model, batch, text_queries=True)

        self.assertEqual(set(model.kwargs), {"class_idx", "text_embedding", "text_token_mask"})


if __name__ == "__main__":
    unittest.main()
