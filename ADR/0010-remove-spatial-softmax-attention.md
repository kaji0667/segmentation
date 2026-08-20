# ADR-0010: Remove Spatial-Softmax Attention for a Controlled Ablation

## Status

Accepted as a controlled experiment. Full metric outcome is pending.

## Context

The learnable token-pooling run is the best same-protocol checkpoint, while the calibrated query/key heatmap under ADR-0009 increased precision but reduced recall, oIoU, official mIoU, class-macro mIoU, and high-IoU success rates. Both the original and calibrated attention maps normalize across all spatial positions, so target pixels compete for fixed probability mass. This is a poor semantic match for a dense mask that may contain many equally valid foreground pixels.

The project also retained `text_object_mask`, `text_spatial_mask`, and `text_context_mask` cache fields and forwarding arguments from a rejected role-aware proposal. Source audit confirmed that the active head never consumes them. Their word-to-token heuristic is not aligned to OpenCLIP BPE positions and should not remain in the active data path.

## Decision

- Keep learnable token pooling and its active `text_token_mask` input.
- Remove `query_proj`, `key_proj`, `attention_gate_weight`, `attention_logit_scale`, spatial attention-map construction, and the attention decoder channel.
- Keep `value_proj` because it remains an active visual decoder branch independent of query/key attention.
- Build the spatial gate from the learned visual gate plus the weighted independent pixel-text similarity map.
- Concatenate gated visual, gated value, and similarity into the decoder.
- Remove the unused object/spatial/context mask arguments from dataset loading, training routing, model prediction, and the segmentation head.
- Stop writing heuristic role-mask fields into newly generated OpenCLIP caches. Continue reading legacy caches by ignoring their extra fields.

## Alternatives

- Feed a constant-zero attention channel. Rejected because it would preserve a dead decoder input and dead parameters rather than deliver the requested cleanup.
- Replace spatial softmax immediately with an independent sigmoid/tanh map. Deferred until the no-attention ablation establishes whether a second text-localization branch is necessary at all.
- Remove `value_proj`. Rejected because it is still used by the decoder and removing it would combine two ablations.

## Impact

- Reduces trainable parameters by 116,097 relative to the calibrated-attention model.
- Changes the decoder input from 258 to 257 channels for `embed_dim=128`.
- Old segmentation-head checkpoints cannot strict-load into this candidate; training continues from the same pretrained YOLO backbone/neck initialization and skips the task head as before.
- Existing token caches remain usable without regeneration.
- Requires the same seed-42, split, augmentation, loss, threshold, checkpoint, and test protocol as `runs/semseg/tpool`.

## Verification

- Python compilation passed for the head, model routing, training entry, dataset loader, and tests.
- Eleven directed tests passed, covering checkpoint selection, token pooling, no-attention structure/gradients, legacy-cache compatibility, and active text-input routing.
- A 2-train/2-val/2-test CUDA smoke completed under `runs/semseg/noattn_smoke2` and strict checkpoint loading passed.

## Related

- `ADR/0008-learnable-token-pooling.md`
- `ADR/0009-calibrate-spatial-attention-heatmap.md`
- `HFSA-main/ultralytics/nn/modules/head.py`
- `HFSA-main/ultralytics/nn/tasks.py`
- `HFSA-main/train_semseg.py`
- `HFSA-main/dataset/rrsisd_refseg_dataset.py`
