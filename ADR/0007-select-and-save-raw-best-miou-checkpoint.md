# ADR-0007: Select and Save the Raw-Best mIoU Checkpoint

## Status

Accepted

## Context

The standard RRSIS-D baseline previously selected checkpoints with the legacy oIoU alias. In addition, `best.pt` was updated only when the validation selection score exceeded the last accepted score by `min_delta`. This is useful for early stopping, but it can discard a real metric maximum when the gain is smaller than `min_delta`.

## Decision

- The standard `baseline` preset explicitly selects validation thresholds and checkpoints by official sample mIoU.
- Keep `best.pt` and its `min_delta` rule unchanged for early-stopping compatibility.
- Save `best_raw.pt` whenever the current selection score is strictly greater than the historical raw maximum, independent of `min_delta`.
- When `--test-after-train` is enabled, prefer `best_raw.pt`; fall back to `best.pt` for legacy runs.
- Reuse the selected checkpoint's frozen validation threshold on test and never rescan thresholds on the test split.

## Alternatives

- Set `min_delta=0`. Rejected because this also changes early-stopping sensitivity and patience behavior.
- Replace `best.pt` with raw-best semantics. Rejected because existing workflows rely on its min-delta behavior.
- Select the best epoch after training from `results.csv`. Rejected because the corresponding weights may already have been overwritten by `last.pt`.

## Impact

- A small but real official-mIoU gain is no longer lost.
- Each run can contain `last.pt`, early-stopping-compatible `best.pt`, and strict metric maximum `best_raw.pt`.
- Existing commands and old run directories remain compatible.
- `best_raw.pt` adds one checkpoint-sized file per saved run.

## Related

- `ADR/0004-standardize-rrsisd-evaluation-metrics.md`
- `HFSA-main/train_semseg.py`
- `HFSA-main/run_semseg_preset.sh`
