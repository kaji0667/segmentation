# ADR-0029: RefSeg 严格单 checkpoint 策略

## 状态

Accepted

## 背景

RefSeg 历史流程会保存 `last.pt`、`best.pt` 和 `best_raw.pt`，并支持 optimizer、scheduler、RNG
恢复。用户最终确认不需要旧文件名兼容或训练恢复 checkpoint，要求只保留一个可用于独立 test、
单图推理和 Web Adapter 的 checkpoint。

## 决策

1. RefSeg 训练唯一 checkpoint 为 `weights/best_raw.pt`。
2. 删除 `--resume` 和完整 optimizer/scheduler/RNG 状态保存与恢复实现。
3. 训练后自动 test 固定读取当前运行目录的 `weights/best_raw.pt`，不回退 `best.pt`。
4. checkpoint 使用 `hfsa_refseg_deployment_v1` payload，只保存模型、配置、epoch 和指标。
5. 删除只服务于恢复与旧文件名选择的历史测试；严格单-checkpoint校验并入现有 RefSeg 测试。
6. 不修改 Backbone、Neck、OpenCLIP、`TextPromptSegment`、Loss、数据协议或其他任务 checkpoint 策略。

## 备选方案

- 保留默认关闭的恢复状态：拒绝。用户明确不要其他 checkpoint。
- 只删除测试但保留运行时代码：拒绝。测试不决定训练输出。
- 修改通用 Ultralytics checkpoint 行为：拒绝。其他任务不在本次范围内。

## 影响

- RefSeg `weights/` 只会写入 `best_raw.pt`。
- 训练中断后不能从 optimizer/scheduler 状态继续，只能重新开始。
- 旧实验的 `best.pt` 不再被自动发现。

## 关联

- ADR-0007
- `HFSA-main/tasks/refseg/checkpoint.py`
- `HFSA-main/tasks/refseg/engine.py`
- `HFSA-main/tests/test_refseg_scripts.py`
