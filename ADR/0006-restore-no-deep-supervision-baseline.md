# ADR-0006: Restore the No-Deep-Supervision Baseline

## Status

Accepted

## Context

ADR-0005 introduced fixed text-conditioned auxiliary mask losses on P3/P4, followed by a P3-only ablation. The no-deep-supervision learnable-gate baseline achieved test `oIoU=0.691823` and `mIoU=0.509016`. P3/P4 deep supervision reached `oIoU=0.685367`, while P3-only deep supervision fell to `oIoU=0.671189` and `mIoU=0.493307`. The regressions affected F1, Pr@0.7-0.9, and most categories.

## Decision

- Restore the exact no-deep-supervision semantic segmentation implementation from commit `5533406` for the head, loss, training arguments, and preset script.
- Keep the two learnable spatial-gate weights.
- Keep the official RRSIS-D evaluation protocol and test-after-train flow.
- Remove auxiliary P3/P4 heads, auxiliary loss parameters, `ds`/`ds_p3` presets, and the deep-supervision regression test.
- Preserve completed experiment directories and historical documentation.

## Alternatives

- Continue reducing auxiliary weights. Rejected because P3/P4 was already neutral-to-negative and P3-only produced a large regression.
- Retain disabled auxiliary modules. Rejected because they increase parameters and checkpoint size while serving no active function.
- Delete failed experiment artifacts. Rejected because the results are reproducibility evidence.

## Impact

- Active model parameter count returns from `4,417,962` to `4,086,952`.
- Checkpoint compatibility returns to the confirmed learnable-gate baseline structure.
- Future improvement work should focus on cross-modal grounding and category balance rather than raw P3/P4 mask supervision.

## Related

- `ADR/0002-learnable-spatial-gate-weights.md`
- `ADR/0004-standardize-rrsisd-evaluation-metrics.md`
- `ADR/0005-fixed-p3-p4-deep-supervision.md`
- `HFSA-main/runs/semseg/ds_p3p4`
- `HFSA-main/runs/semseg/ds_p3`
