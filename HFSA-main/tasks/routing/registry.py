"""Static registry for the three integrated HFSA inference tasks."""

from __future__ import annotations

from .config import InputFieldConfig, OutputFieldConfig, TaskConfig


IMAGE_INPUT = InputFieldConfig(
    key="image",
    label="上传遥感图像",
    kind="image",
    placeholder="拖放图片或点击选择文件",
    help_text="支持 JPG、PNG、WEBP 和 TIFF；实际模型接口接入后再执行推理。",
    accept="image/jpeg,image/png,image/webp,image/tiff",
)


_TASKS: tuple[TaskConfig, ...] = (
    TaskConfig(
        task_id="classification",
        title="场景分类",
        technical_name="Scene Classification",
        description="判断遥感图像所属的场景类别",
        detail="上传一张遥感图像，系统将返回最可能的场景类别及其概率。",
        icon="scene",
        accent_color="#2563EB",
        action_label="开始分类",
        result_title="场景分类结果",
        checkpoint="runs/classification/vrsbench_scene/weights/best.pt",
        adapter_path="tasks.routing.adapters.classification.SceneClassificationAdapter",
        inputs=(IMAGE_INPUT,),
        outputs=(
            OutputFieldConfig("classes", "候选场景", "ranking"),
            OutputFieldConfig("probabilities", "类别概率", "probability_bars"),
        ),
        runtime_defaults={"imgsz": 640, "topk": 5},
    ),
    TaskConfig(
        task_id="counting",
        title="目标计数",
        technical_name="Text-guided Object Counting",
        description="统计图像中指定类别的目标数量",
        detail="上传图像并输入要统计的目标类别，系统将返回数量、位置和置信度。",
        icon="counting",
        accent_color="#EA580C",
        action_label="开始统计",
        result_title="目标计数结果",
        checkpoint="runs/counting/counting-VRSBench-ViT-L-14/weights/best.pt",
        adapter_path="tasks.routing.adapters.counting.CountingAdapter",
        inputs=(
            IMAGE_INPUT,
            InputFieldConfig(
                key="target_class",
                label="要统计的目标",
                kind="text",
                placeholder="例如：airplane、ship、windmill",
                help_text="请输入一个明确的目标类别。",
            ),
        ),
        outputs=(
            OutputFieldConfig("count", "目标数量", "number"),
            OutputFieldConfig("boxes", "检测框", "boxes"),
            OutputFieldConfig("scores", "置信度", "scores"),
            OutputFieldConfig("visualization", "标注结果图", "image", downloadable=True),
        ),
        runtime_defaults={"imgsz": 800, "conf_thres": 0.15, "iou_thres": 0.5, "max_det": 300},
    ),
    TaskConfig(
        task_id="refseg",
        title="语义分割",
        technical_name="Text-guided Semantic Segmentation",
        description="根据文字描述提取目标区域",
        detail="上传图像并描述目标，系统将生成目标 Mask、概率图和叠加效果。",
        icon="refseg",
        accent_color="#7C3AED",
        action_label="开始分割",
        result_title="语义分割结果",
        checkpoint="runs/semseg/srp_yolov12m_axis/weights/best_raw.pt",
        adapter_path="tasks.routing.adapters.refseg.RefSegAdapter",
        inputs=(
            IMAGE_INPUT,
            InputFieldConfig(
                key="text",
                label="描述需要提取的目标",
                kind="text",
                placeholder="例如：左侧灰色的小型风力发电机",
                help_text="位置、颜色、大小等描述可以帮助系统区分目标。",
            ),
        ),
        outputs=(
            OutputFieldConfig("mask", "二值 Mask", "image", downloadable=True),
            OutputFieldConfig("probability", "概率图", "image", downloadable=True),
            OutputFieldConfig("overlay", "叠加效果", "image", downloadable=True),
        ),
        runtime_defaults={"imgsz": None, "threshold": None},
    ),
)

_TASKS_BY_ID = {task.task_id: task for task in _TASKS}
if len(_TASKS_BY_ID) != len(_TASKS):
    raise RuntimeError("HFSA task IDs must be unique.")


def list_task_configs() -> tuple[TaskConfig, ...]:
    """Return task definitions in their intended interface display order."""

    return _TASKS


def get_task_config(task_id: str) -> TaskConfig:
    """Return one task definition or raise a clear lookup error."""

    normalized = str(task_id).strip().lower()
    try:
        return _TASKS_BY_ID[normalized]
    except KeyError as exc:
        choices = ", ".join(_TASKS_BY_ID)
        raise KeyError(f"Unknown HFSA task {task_id!r}; expected one of: {choices}") from exc
