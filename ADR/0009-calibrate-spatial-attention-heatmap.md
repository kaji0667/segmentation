# ADR-0009: Calibrate the Spatial Attention Heatmap

## Status

Rejected as the mainline after the full seed-42 controlled experiment.

## Context

The active `TextPromptSegment` normalizes both spatial keys and the text query, computes their cosine similarity, divides it again by `sqrt(embed_dim)`, and applies a spatial softmax. With `embed_dim=128`, the logits are bounded to approximately `[-0.088, 0.088]`. At the P3 resolution of `64 x 64`, the resulting 4,096-position probability map is close to uniform and has a mean magnitude of only `1/4096`.

This makes the attention channel poorly scaled for both the spatial gate and the mask decoder. The completed learnable-token-pooling checkpoint also learned a negative attention-gate weight, consistent with the current map not providing a useful positive localization signal.

## Decision

- Keep normalized key/query cosine similarity, but remove the additional `1/sqrt(embed_dim)` compression.
- Add one learnable logit-scale scalar initialized to zero, so the initial positive temperature is `exp(0)=1`.
- Apply spatial softmax to the temperature-scaled cosine logits.
- Convert probabilities to relative density with `probability * num_positions - 1`, making a uniform map exactly zero and removing the `1/HW` magnitude dependence.
- Apply `tanh` to bound the heatmap to `[-1, 1]` before it enters the existing spatial gate and mask decoder.
- Keep token pooling, P3/P4/P5 fusion, FiLM, similarity path, decoder, loss, backbone, neck, OpenCLIP, and evaluation unchanged.

## Alternatives

- Only remove `1/sqrt(embed_dim)`: rejected because the softmax output would still have a `1/HW` scale that is too small for a dense mask feature.
- Feed raw cosine similarity directly: rejected because the existing query/key branch should retain a normalized competition mechanism distinct from the direct pixel-text similarity branch.
- Use an unbounded `HW * softmax` map: rejected because a sharply focused distribution could produce values close to `HW` and saturate the gate.
- Replace the head with token-pixel cross-attention: deferred because it is a substantially larger architectural change.

## Impact

- Adds one trainable scalar parameter.
- Produces a signed, bounded, resolution-independent localization feature.
- Uniform or uninformative attention contributes exactly zero instead of a small positive constant everywhere.
- Old strict checkpoints do not contain `attention_logit_scale`; the controlled experiment requires fresh training from the same pretrained backbone/neck initialization.
- Requires comparison against `runs/semseg/tpool` using oIoU, official mIoU, Pr@0.5-0.9, precision/recall/F1, positive-area rates, per-category mIoU, learned temperature, and fixed previews.

## Related

- `HFSA-main/ultralytics/nn/modules/head.py`
- `HFSA-main/tests/test_semseg_spatial_attention.py`
- `ADR/0008-learnable-token-pooling.md`
- `DEVELOPMENT_LOG.md`

## Outcome

- The full run selected epoch 29 and evaluated all 3,481 test samples at the frozen validation threshold `0.70`.
- Test result: `oIoU=0.678791`, `mIoU=0.514869`, class-macro mIoU `0.538315`, precision `0.796597`, recall `0.821107`, and `Pr@0.9=0.128699`.
- Against `runs/semseg/tpool`, oIoU regressed by `0.004630`, mIoU by `0.004949`, class-macro mIoU by `0.006695`, recall by `0.015892`, and `Pr@0.9` by `0.008905`; precision improved by `0.008254`.
- The map reduced predicted foreground area but made masks less complete. Global spatial-softmax competition is not retained as the active architecture.
- ADR-0010 evaluates the cleaner alternative of removing the attention branch entirely.
