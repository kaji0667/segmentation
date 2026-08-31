# HFSA task packages

`tasks/` contains task-specific data handling, applications, training, evaluation, inference, metrics, and checkpoint policy. Shared model components remain in `ultralytics/`.

The current integrated task packages are:

- `refseg/`: text-guided referring segmentation.
- `classification/`: single-label remote-sensing scene classification.
- `counting/`: text-guided detection followed by NMS box counting.

Single-sample inference remains task-specific. `refseg/inference.py` accepts an
image plus a referring expression and returns a binary mask, probability map,
overlay and metadata. Scene classification accepts only an image, while object
counting accepts an image plus a target-class prompt. A future outer router
must preserve these different contracts after the user manually selects a task.

`routing/` now provides the model-agnostic catalog used by the local web shell.
It serializes user-facing task metadata, validates each task's distinct inputs,
and supports late Adapter registration without importing or loading any model
when the page starts. Concrete classification, counting and RefSeg Adapters are
intentionally left for subsequent single-task interface work.

Each task has a thin root-level `train_<task>.py` and `test_<task>.py` entry plus matching Shell scripts. The existing target-detection `train.py`, `val.py`, and `text_encoder/` pipeline is intentionally outside this reorganization.

`refseg/checkpoint.py` is task-specific because referring segmentation uses a custom PyTorch training loop and must explicitly persist optimizer, scheduler, early-stopping, validation-threshold, RNG, and data-generator state. Classification stores its smaller Head checkpoint from its trainer, while counting delegates checkpoint management to Ultralytics detection training.
