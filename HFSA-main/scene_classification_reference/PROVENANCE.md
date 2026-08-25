# Scene Classification Reference Provenance

- Source: `https://github.com/zhuoletian-collab/changjingfenlei.git`
- Imported branch: `main`
- Imported commit: `688c2a9ec281febb5877dcd6c29bf9eeec8c02bb`
- Imported on: 2026-08-25

This directory is retained as auditable reference source for the integrated scene-classification task. The nested Git metadata, pretrained checkpoints, datasets, runs, caches, and personal machine configuration are excluded from the release repository. The integrated implementation preserves the source `ClassifyHeadV2` multi-scale spatial-attention + GeM architecture, frozen YOLOv12m feature-extractor protocol, class-balanced sampling, warmup/cosine schedule, cross-entropy objective, and accuracy/F1 evaluation flow while replacing personal paths with task configuration and class-based applications.
