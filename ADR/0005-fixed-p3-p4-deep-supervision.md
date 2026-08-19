# ADR-0005: Fixed P3/P4 Text-Conditioned Deep Supervision

## Status

Superseded by ADR-0006. The completed P3/P4 and P3-only runs did not improve the official test result, so this design is no longer part of the active model.

## Context

The current FPN-style `TextPromptSegment` fuses P3/P4/P5 and applies the binary mask loss only to the final output. The latest official test result has strong aggregate oIoU but weaker sample mIoU and weak performance on classes such as vehicle, harbor, and bridge. A low-risk experiment is needed to provide more direct training signals to the finer P3/P4 features without changing the backbone, neck, OpenCLIP encoder, final spatial gate, or inference output.

## Decision

- Add text-conditioned auxiliary binary mask heads to P3 and P4 only.
- Use fixed loss weights: final `1.0`, P3 `0.20`, P4 `0.10`.
- Do not add P5 supervision in the first experiment because its 1/32 resolution can erase small targets.
- Downsample auxiliary targets with foreground-preserving adaptive max pooling.
- Reuse the current dynamic positive-weight BCE plus Tversky loss for every output.
- Apply small-target sample multiplication only to the final loss, avoiding duplicate amplification in the auxiliary branches.
- Enable the branches only when either auxiliary loss weight is positive.
- During validation and inference, return only the final mask logits.

The controlled preset is `run_semseg_preset.sh ds`, with default output `runs/semseg/ds_p3p4`.

After the first completed P3/P4 run, a follow-up controlled preset `ds_p3` keeps P3 at `0.20`, disables P4, and lowers `min_delta` from `0.001` to `0.0002`. This isolates whether the coarse P4 auxiliary target caused regressions on thin or boundary-sensitive classes while allowing smaller validation improvements to update `best.pt`.

## Alternatives

- Supervise P3/P4/P5 equally. Rejected for the first run because coarse P5 labels can harm small targets and make attribution harder.
- Learn the auxiliary weights. Rejected because the requested first experiment uses fixed weights and a learned weighting mechanism would add another variable.
- Replace the decoder. Rejected because previous decoder replacements regressed substantially.
- Add image-only auxiliary masks. Rejected because every output must remain conditioned on the referring expression.

## Impact

- Adds training-only mask predictions and a small number of head parameters.
- Adds no auxiliary branch computation during validation or inference.
- Preserves the two existing learnable spatial-gate weights.
- Requires a seed-42 controlled run against `rrsisd_learnable_gate_official_seed42` before any improvement claim.

## Outcome

- `ds_p3p4` test: `oIoU=0.685367`, `mIoU=0.508498`, below the no-deep-supervision baseline `oIoU=0.691823`, `mIoU=0.509016`.
- `ds_p3` test: `oIoU=0.671189`, `mIoU=0.493307`, a clear regression.
- P3-only supervision disproved the hypothesis that the coarse P4 auxiliary target was the main cause of the regression.
- The source code, loss parameters, presets, and auxiliary-head test were removed. Experiment directories remain under `runs/semseg/` as evidence.

## Related

- `HFSA-main/ultralytics/nn/modules/head.py`
- `HFSA-main/ultralytics/utils/loss.py`
- `HFSA-main/train_semseg.py`
- `HFSA-main/run_semseg_preset.sh`
- `ADR/0002-learnable-spatial-gate-weights.md`
- `ADR/0004-standardize-rrsisd-evaluation-metrics.md`
