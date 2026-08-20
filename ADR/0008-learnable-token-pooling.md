# ADR-0008: Learnable Token Pooling for Text-Guided Segmentation

## Status

Accepted. The full seed-42 run improved the same-protocol mIoU-selection baseline and token pooling is retained as the active text aggregation path.

## Context

`TextPromptSegment` receives cached OpenCLIP token features with shape `[B, 77, 768]`, but previously reduced them with an unconditional `mean(1)`. RRSIS-D validation expressions contain only 6.69 valid tokens on average, so padding-position features participate in the same average as words. The model therefore has no trainable mechanism to emphasize useful contextual tokens or reduce padding influence.

Recent referring-segmentation work motivates finer text use instead of a single fixed sentence average: CLIP-Adapter (arXiv:2110.04544), Global-Local Context Features (arXiv:2303.17811), RMSIN (arXiv:2312.12470), and RSRefSeg (arXiv:2501.06809).

An initial role-aware proposal was rejected after independent review because the existing object/spatial masks approximate word positions rather than true OpenCLIP BPE spans and can include reference objects in same-class relation expressions.

## Decision

Replace fixed token mean pooling inside `TextPromptSegment` with one zero-initialized learnable attention pool:

```python
scores = token_score(text_tokens) + valid_token_bias * valid_mask
weights = softmax(scores, dim=1)
text_vector = sum(text_tokens * weights)
```

- `token_score` is a bias-free `Linear(768, 1)` initialized to zero.
- `valid_token_bias` is one scalar initialized to zero.
- At initialization, all token logits are zero, so the output is mathematically equivalent to the previous `tokens.mean(1)` baseline.
- Do not use the heuristic object/spatial masks in this experiment.
- Do not change OpenCLIP, backbone, neck, decoder, loss, spatial-gate weights, checkpoint selection, or evaluation protocol.

## Alternatives

- Hard masked mean pooling: rejected because it changes the initial text vector and weakens attribution.
- Object/spatial role residuals: rejected because current masks are not aligned to true BPE spans and can mark distractor objects.
- Token-level cross-attention decoder: rejected as a larger architectural change.
- Residual MLP adapter: deferred because learned token pooling directly addresses aggregation with fewer parameters.

## Impact

- Adds 769 trainable parameters.
- Keeps the 2-D `[B, D]` text-vector path unchanged.
- Keeps the old behavior at initialization up to floating-point reduction order.
- Old segmentation checkpoints loaded strictly into the new architecture will report two missing parameter entries; current training initialization skips the segmentation head and uses non-strict loading.
- Requires a full seed-42 comparison on official metrics and per-category mIoU.

## Related

- `HFSA-main/ultralytics/nn/modules/head.py`
- `HFSA-main/tests/test_semseg_text_token_pooling.py`
- `ADR/0002-learnable-spatial-gate-weights.md`
- `ADR/0006-restore-no-deep-supervision-baseline.md`
- `ADR/0007-select-and-save-raw-best-miou-checkpoint.md`

## Outcome

- Test `oIoU=0.683420`, `mIoU=0.519818`, and class-macro mIoU `0.545010`.
- Compared with `base_miou`, oIoU improved by `0.011223`, official mIoU by `0.010626`, and class-macro mIoU by `0.008753`.
- Precision, Recall, F1, and all reported Pr@ thresholds also improved under the same selection protocol.
- The next experiment keeps token pooling and calibrates the spatial attention heatmap under ADR-0009.
