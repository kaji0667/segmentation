# ADR-0011: Axis-Aware Directional Flips and Mask-Size Reconciliation

## Status

Accepted for controlled evaluation. Full legacy-versus-axis-aware metric outcome is pending.

## Context

The former RRSIS-D augmentation rule blocked both horizontal and vertical flips whenever an expression contained any directional or positional word. This is unnecessarily restrictive for single-axis language: `left` makes a horizontal flip invalid but does not invalidate a vertical flip, while `above` makes a vertical flip invalid but does not invalidate a horizontal flip. The former vocabulary also omitted `above` and `below`.

A full metadata audit also found 14 official samples whose decoded JPEG height differs from the `800 x 800` RLE mask size. Silently resizing the image and mask independently to the final square size hides the source mismatch and leaves the reconciliation policy implicit.

## Decision

- Classify direction words by horizontal and vertical axes.
- Under the default `axis-aware` policy, block only the flip on the constrained axis.
- Treat `above` and `below` as vertical constraints and diagonal compass words as constraints on both axes.
- Retain a selectable `legacy` policy for strict comparison with the former behavior.
- Consume one horizontal and one vertical random draw for every sample under both policies, then gate application by the policy. This keeps later color-jitter draws aligned between controlled runs.
- Preserve all 14 size-mismatched samples and nearest-neighbor resize the decoded binary mask to the actual JPEG size before the common model-input resize.
- Compare the two augmentation policies with identical no-attention architecture, seed, data split, loss, sampler, optimizer, scheduler, checkpoint selection, and frozen-threshold test evaluation.

## Alternatives

- Continue blocking both flips for every directional expression: rejected because it discards valid augmentation on the unconstrained axis.
- Rewrite direction words after flipping: rejected because relation-bearing expressions can contain multiple targets and cannot be safely rewritten with simple token replacement.
- Drop the 14 mismatched samples: rejected because it changes the official split and disproportionately removes small or spatially described targets.
- Crop or pad every mismatched mask: rejected because the dataset does not identify whether missing or extra JPEG rows are anchored at the top, bottom, or both; nearest-neighbor coordinate rescaling makes the least unsupported geometric assumption.

## Impact

- No changes to backbone, neck, OpenCLIP, segmentation head, loss, or evaluation.
- The active augmentation default changes from global directional blocking to axis-aware blocking.
- Existing commands can reproduce the former behavior with `--augment-direction-policy legacy` or `AUGMENT_DIRECTION_POLICY=legacy`.
- Full controlled runs are required before claiming a metric improvement.

## Related

- `HFSA-main/dataset/rrsisd_refseg_dataset.py`
- `HFSA-main/train_semseg.py`
- `HFSA-main/run_semseg_preset.sh`
- `HFSA-main/tests/test_rrsisd_axis_aware_augmentation.py`
- `runs/semseg/noattn_aug_legacy`
- `runs/semseg/noattn_aug_axis`
