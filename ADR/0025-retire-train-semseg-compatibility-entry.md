# ADR-0025: 删除 train_semseg 兼容入口

## 状态

Accepted

## 背景

ADR-0023 将指代分割实现迁入 `tasks/refseg/`，并暂时保留根目录 `train_semseg.py` 作为兼容转发。团队现已确认不再保留该历史入口；继续保留会让测试、文档和服务器代码误把兼容文件当作正式实现。

## 决策

1. 删除 `HFSA-main/train_semseg.py`。
2. 正式训练与测试入口保持为 `train_refseg.py` 和 `test_refseg.py`。
3. 测试直接从 `tasks.refseg.engine` 或 `tasks.refseg.checkpoint` 导入其实际被测接口。
4. 历史 preset 若继续保留，改为调用 `train_refseg.py`。

## 备选方案

- 永久保留兼容转发：拒绝。当前没有需要支持的已发布外部 API，且会继续制造入口歧义。

## 影响

- 历史命令 `python train_semseg.py ...` 不再可用。
- 模型、数据、Loss、checkpoint 格式和训练行为不变。
- 新代码和未来路由只能依赖 `tasks/refseg/` 的真实接口。

## 关联

- ADR-0023
- `HFSA-main/train_refseg.py`
- `HFSA-main/test_refseg.py`
- `HFSA-main/tasks/refseg/`
