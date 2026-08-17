# ADR-0004: Standardize RRSIS-D Evaluation Metrics

## Status

Accepted

## Context

RRSIS-D papers report overall IoU (oIoU), sample-mean IoU (mIoU), and Pr@0.5 through Pr@0.9. The local training script previously stored foreground aggregate IoU in the `miou` field, stored the official sample-mean result only as `sample_miou`, and computed per-category aggregate IoU rather than the per-category sample mean used in published comparisons. This made otherwise correct values easy to mislabel in reports.

The competition guidance also requires official dataset protocols, reproducible threshold handling, and resource-efficiency evidence. Test data must not be used to choose the mask threshold.

## Decision

- Expose foreground aggregate IoU as `oiou`.
- Expose the average of per-sample IoUs as `official_miou` while retaining `sample_miou` and legacy fields for compatibility.
- Compute per-category mIoU by grouping per-sample IoUs by `class_idx` and averaging within each category.
- Keep per-category aggregate IoU separately as `class_oiou`.
- Report Pr@0.5, Pr@0.6, Pr@0.7, Pr@0.8, and Pr@0.9.
- Allow validation checkpoint selection by either oIoU or official mIoU.
- When test evaluation is enabled, load the best validation checkpoint and freeze its selected threshold for the test split.
- Record basic test resource evidence: parameter counts, checkpoint size, total evaluation time, mean time per evaluated sample, and peak allocated GPU memory.

## Alternatives

- Rename and remove all legacy fields immediately. Rejected because existing result readers and historical CSV files depend on them.
- Continue using the best threshold independently on the test split. Rejected because this leaks test information and is not comparable with published protocols.
- Treat background/foreground binary mIoU as the paper mIoU. Rejected because RRSIS-D papers define mIoU as the mean of per-sample foreground IoUs.

## Impact

- New result files can be compared directly with RMSIN, FIANet, SBANet, and related RRSIS-D work.
- Historical `target_iou` remains equivalent to oIoU and historical `sample_miou` remains equivalent to official mIoU.
- Historical `class_iou` is category aggregate IoU and must not be presented as category mIoU.
- Full competition efficiency reporting still needs a dedicated batch-1 latency benchmark with P95 latency and peak CPU memory.

## Related

- `HFSA-main/train_semseg.py`
- `HFSA-main/run_semseg_preset.sh`
- `PROJECT_RULES.md`
- `DEVELOPMENT_LOG.md`
