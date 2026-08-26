# HFSA task packages

`tasks/` contains task-specific data handling, applications, training, evaluation, inference, metrics, and checkpoint policy. Shared model components remain in `ultralytics/`.

The current integrated task packages are:

- `refseg/`: text-guided referring segmentation.
- `classification/`: single-label remote-sensing scene classification.
- `counting/`: text-guided detection followed by NMS box counting.

Each task has a thin root-level `train_<task>.py` and `test_<task>.py` entry plus matching Shell scripts. The existing target-detection `train.py`, `val.py`, and `text_encoder/` pipeline is intentionally outside this reorganization.

`refseg/checkpoint.py` is task-specific because referring segmentation uses a custom PyTorch training loop and must explicitly persist optimizer, scheduler, early-stopping, validation-threshold, RNG, and data-generator state. Classification stores its smaller Head checkpoint from its trainer, while counting delegates checkpoint management to Ultralytics detection training.
