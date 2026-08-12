# ADR-0001: Project Baseline

## Status

Accepted

## Context

HFSA contains an existing multimodal remote-sensing codebase. The current personal development branch is responsible for text-guided referring semantic segmentation, while the main project framework, backbone, neck, OpenCLIP encoder, and much of the general YOLO/Ultralytics integration were completed before this branch work.

## Decision

Use `PROJECT_RULES.md` as the single source of truth for project rules. Treat the current semantic segmentation branch as an incremental branch-feature implementation, not a full project rewrite.

The semantic segmentation branch may modify dataset adapters, segmentation training flow, segmentation-head integration, loss/metric policy, experiment logging, and documentation. Changes to backbone, neck, OpenCLIP encoder, or general Ultralytics internals require explicit architecture review and user confirmation.

## Alternatives

- Rewrite the whole project around a new segmentation framework. Rejected because it violates the team boundary and increases risk.
- Modify backbone/neck to chase immediate metric improvements. Rejected as the default path because the senior teammate explicitly constrained that area.

## Impact

This keeps the branch narrowly scoped and makes future experiments easier to compare. The tradeoff is that some performance limitations from the fixed image/text feature extractors must be addressed through the segmentation head, data policy, loss policy, and training schedule first.

## Related

- `PROJECT_RULES.md`
- `ARCHITECTURE.md`
- `DEVELOPMENT_LOG.md`
- `HFSA-main/train_semseg.py`
- `HFSA-main/dataset/rrsisd_refseg_dataset.py`
