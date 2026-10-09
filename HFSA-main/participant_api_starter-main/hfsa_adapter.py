"""Official protocol adapter; existing HFSA predictors own all model inference.

Question routing here is deliberately bounded and independent of the manual Web
router. Unknown questions raise an error rather than fabricate a valid answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# Aliases describe the actual 21-class checkpoint, not arbitrary scene classes.
SCENE_ALIASES = {
    "airport": ("airport", "机场", "飞机场"),
    "baseball-diamond": ("baseball diamond", "baseball field", "棒球场"),
    "basketball-court": ("basketball court", "篮球场"),
    "bridge": ("bridge", "桥", "桥梁", "大桥"),
    "chimney": ("chimney", "烟囱"),
    "dam": ("dam", "大坝", "水坝"),
    "expressway-service-area": ("expressway service area", "highway service area", "高速公路服务区"),
    "expressway-toll-station": ("expressway toll station", "highway toll station", "高速公路收费站"),
    "golffield": ("golffield", "golf field", "golf course", "高尔夫球场"),
    "ground-track-field": ("ground track field", "running track", "track and field", "田径场", "操场"),
    "harbor": ("harbor", "harbour", "port", "港口", "港湾"),
    "helipad": ("helipad", "直升机停机坪"),
    "overpass": ("overpass", "立交桥"),
    "roundabout": ("roundabout", "环岛"),
    "soccer-ball-field": ("soccer ball field", "soccer field", "football field", "足球场"),
    "stadium": ("stadium", "体育场"),
    "storage-tank": ("storage tank", "oil storage tank", "储罐", "储油罐"),
    "swimming-pool": ("swimming pool", "游泳池"),
    "tennis-court": ("tennis court", "网球场"),
    "trainstation": ("trainstation", "train station", "railway station", "火车站"),
    "windmill": ("windmill", "wind turbine", "风车", "风力发电机"),
}


def normalize_label(value: str) -> str:
    value = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
    return re.sub(r"^(?:a|an|the)\s+", "", value).strip(" 。.?!？！")


_SCENE_LOOKUP = {
    normalize_label(alias): label
    for label, aliases in SCENE_ALIASES.items()
    for alias in (label, *aliases)
}
_TARGET_LOOKUP = {
    alias: {"golffield": "golf field", "trainstation": "train station"}.get(label, label.replace("-", " "))
    for alias, label in _SCENE_LOOKUP.items()
}
_TARGET_LOOKUP.update({
    alias + "s": label for alias, label in list(_TARGET_LOOKUP.items())
    if re.fullmatch(r"[a-z ]+", alias) and not alias.endswith("s")
})
_TARGET_LOOKUP.update({
    "golffield": "golf field", "golf course": "golf field", "高尔夫球场": "golf field",
    "trainstation": "train station", "railway station": "train station", "火车站": "train station",
    "airplane": "airplane", "airplanes": "airplane", "plane": "airplane", "planes": "airplane", "飞机": "airplane",
    "ship": "ship", "ships": "ship", "boat": "ship", "boats": "ship", "船": "ship", "船舶": "ship", "轮船": "ship",
    "vehicle": "vehicle", "vehicles": "vehicle", "car": "vehicle", "cars": "vehicle", "汽车": "vehicle", "车辆": "vehicle",
    "windmills": "windmill", "wind turbines": "windmill", "storage tanks": "storage tank",
})
_SCENE_QUESTION = re.compile(
    r"\bclassif(?:y|ication)\b|(?:what|which).*(?:type|category|class).*(?:scene|image|picture|land.use)|"
    r"(?:what|which).*(?:scene|image|picture).*(?:type|category|class)|"
    r"^(?:what|which)\s+scene\s+(?:is|does)|(?:什么|哪种|哪个).*(?:场景|类别|类型)|"
    r"(?:场景|图像|图片|影像).*(?:分类|类别|类型)|(?:分类|场景识别)", re.I,
)
_COUNT_PATTERNS = (
    r"(?:how many)\s+(.+?)(?:\s+(?:are there|are visible|can you see|are present|are shown|are))?(?:\s+(?:in|on)\s+(?:the|this)\s+(?:image|picture|photo))?",
    r"(?:please\s+)?count\s+(.+?)(?:\s+(?:in|on)\s+(?:the|this)\s+(?:image|picture|photo))?",
    r"(?:what is\s+)?(?:the\s+)?number of\s+(.+?)(?:\s+(?:in|on)\s+(?:the|this)\s+(?:image|picture|photo))?",
    r"(?:请)?(?:统计|数出|数一数)(?:图中|图像中|图片中)?(?:的)?(.+?)(?:的)?(?:数量|数目|个数)?",
    r"(?:图中|图像中|图片中)?(?:有|共有|一共有)?(?:多少|几)(?:个|架|艘|辆|座|处|只)?(.+)",
    r"(?:图中|图像中|图片中)?(?:的)?(.+?)(?:有|共有|一共有)?(?:多少|几)(?:个|架|艘|辆|座|处|只)?",
)
_PRESENCE_PATTERNS = (
    r"(?:is|are) there\s+(.+?)(?:\s+(?:in|on)\s+(?:the|this)\s+(?:image|picture|photo))?",
    r"does\s+(?:the|this)\s+(?:image|picture|photo)\s+(?:contain|show|have)\s+(.+)",
    r"(?:图中|图像中|图片中)(?:是否有|有没有|有无|是否存在)(.+)",
)


@dataclass(frozen=True)
class QuestionPlan:
    task: str
    target: str = ""


class AdapterFailure(ValueError):
    """Expected adapter rejection with a fixed, safe diagnostic code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _capture_replay_request(request: dict[str, Any]) -> None:
    """Optional private request copy for local replay, excluding credentials/gold."""
    directory = os.environ.get("HFSA_API_REPLAY_DIR", "").strip()
    if not directory:
        return
    try:
        fields = ("protocol_version", "request_id", "item_id", "question", "answer_type",
                  "choices", "image_width", "image_height")
        payload = {name: request[name] for name in fields if name in request}
        payload["images"] = [
            {name: spec[name] for name in ("asset_id", "sha256", "mime_type") if name in spec}
            for spec in request.get("images", [])
        ]
        fields = ("type", "values", "case_insensitive", "question_form", "coordinate_format",
                  "length", "min_length", "minimum")
        payload["response_constraint"] = {
            name: request["response_constraint"][name]
            for name in fields if name in request.get("response_constraint", {})
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")
        if len(encoded) > 128 * 1024:
            raise ValueError("Replay request exceeds protocol size limit")
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        path = target / (hashlib.sha256(encoded).hexdigest() + ".json")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
    except Exception as error:
        # Capture errors must not fail inference or reveal request/exception text.
        try:
            print(json.dumps({"event": "hfsa_api_replay_capture", "status": "failed",
                              "error_type": type(error).__name__}), file=sys.stderr, flush=True)
        except OSError:
            pass


def _prediction_diagnostic(request: dict[str, Any], image_count: int,
                           plan: QuestionPlan | None, started: float,
                           error: Exception | None = None) -> None:
    # Never log question/choices/answers, image paths, headers or exception text.
    def identifier(value: Any) -> str | None:
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
            return value
        return None

    kind = request.get("response_constraint", {}).get("type")
    record = {
        "event": "hfsa_api_prediction",
        "item_id": identifier(request.get("item_id")),
        "request_id": identifier(request.get("request_id")),
        "constraint_type": kind if kind in {"enum", "single_choice", "short_text", "integer", "bbox"} else None,
        "image_count": image_count,
        "task": plan.task if plan is not None else None,
        "status": "failed" if error is not None else "succeeded",
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }
    if error is not None:
        record["reason"] = error.code if isinstance(error, AdapterFailure) else "unexpected_exception"
        record["error_type"] = type(error).__name__
    try:
        print(json.dumps(record, ensure_ascii=True, separators=(",", ":")), file=sys.stderr, flush=True)
    except OSError:
        # An unavailable log destination must not change the prediction result.
        pass


def _question_text(question: str) -> str:
    # Strip only explicit answer-format instructions, never target qualifiers.
    text = re.split(
        r"[.?!。？！]\s*(?:(?:please\s+)?(?:answer|respond|return|only\s+(?:answer|return))\b|回答|请回答|仅回答|只回答)",
        question, maxsplit=1, flags=re.I,
    )[0]
    text = re.sub(r"^请问", "", text.strip())
    return text.strip().rstrip("。？！.?!").strip()


def _match_target(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.fullmatch(pattern, text, re.I)
        if match:
            target = match.group(1).strip()
            target = re.sub(r"^(?:the|a|an|any)\s+", "", target, flags=re.I)
            target = re.sub(r"^(?:图中|图像中|图片中)(?:的)?", "", target)
            return target.strip()
    return None


def parse_question(request: dict[str, Any]) -> QuestionPlan:
    kind = request["response_constraint"]["type"]
    if kind == "bbox":
        # Protocol wrappers describe the output before the referring expression.
        # Extract before stripping answer instructions, which may precede it.
        parts = re.split(r"\bdescription\s*:\s*", request["question"], maxsplit=1, flags=re.I)
        if (len(parts) == 2 and re.search(r"\bbounding\s+box\b", parts[0], re.I)
                and re.search(r"\b(?:identify|locate|find|draw|return|provide|give|predict)\b", parts[0], re.I)):
            target = _question_text(parts[1])
            if not target:
                raise AdapterFailure("missing_target", "Missing target description")
            return QuestionPlan("refseg", target)
    text = _question_text(request["question"])
    if not text or request["response_constraint"].get("question_form") == "change_region":
        raise AdapterFailure("unsupported_change_region_or_empty_question", "Unsupported empty question or change-region task")
    count = _match_target(text, _COUNT_PATTERNS)
    if count is not None:
        if kind not in {"integer", "short_text", "single_choice", "enum"}:
            raise AdapterFailure("count_constraint_mismatch", "Counting requires a numeric answer constraint")
        # The existing counting checkpoint has no spatial/relation count policy.
        if re.search(r"\b(and|or|not|except|left|right|top|bottom|near|between|above|below)\b|左|右|上方|下方|附近|之间|之外|和|或", count, re.I):
            raise AdapterFailure("unsupported_spatial_or_multi_category_count", "Spatial, relational or multi-category counting is unsupported")
        return QuestionPlan("counting", count)
    if kind == "bbox":
        detection = re.match(r"^(?:please\s+)?detect\s+|^(?:请)?检测(?:出)?(?:图中|图像中|图片中)?(?:的)?", text, re.I)
        if detection:
            target = text[detection.end():].strip()
            target = re.sub(r"\s+(?:in|on)\s+(?:the|this)\s+(?:image|picture|photo)$", "", target, flags=re.I)
            if re.search(r"\b(all|every|each|left|right|top|bottom|near|between)\b|所有|全部|左|右|上方|下方|附近|之间", target, re.I):
                raise AdapterFailure("unsupported_detection_selector", "Detection bbox accepts one unqualified target; use referring localization for qualifiers")
            return QuestionPlan("detection", target)
        target = re.sub(
            r"^(?:请)?(?:框出|定位|分割出)(?:图中的|图中|图像中的|图像中|图片中的|图片中)?\s*|"
            r"^(?:please\s+)?(?:draw\s+(?:a\s+)?(?:bounding\s+)?box\s+(?:around|for)|locate|find|segment|ground)\s+",
            "", text, count=1, flags=re.I,
        )
        target = re.sub(r"\s+(?:in|on)\s+(?:the|this)\s+(?:image|picture|photo)$", "", target, flags=re.I)
        if target == text and re.search(r"\?|？|\b(?:what|which|how|why|is|are)\b", text, re.I):
            raise AdapterFailure("unsupported_bbox_question", "Unsupported bbox question")
        return QuestionPlan("refseg", target)
    presence = _match_target(text, _PRESENCE_PATTERNS)
    if presence is not None and kind in {"enum", "single_choice", "short_text"}:
        if re.search(r"\b(anything|something|objects?|things?|and|or|not)\b|任何|物体|东西|和|或", presence, re.I):
            raise AdapterFailure("unsupported_presence_target", "Presence questions require one concrete target category")
        return QuestionPlan("presence", presence)
    if _SCENE_QUESTION.search(text) and kind in {"short_text", "enum", "single_choice"}:
        return QuestionPlan("classification")
    raise AdapterFailure("unsupported_question_form", "Question is outside the supported classification/detection/counting/refseg forms")


def _constrained_candidates(request: dict[str, Any]) -> list[tuple[str, str]]:
    constraint = request["response_constraint"]
    if constraint["type"] == "single_choice":
        return [(key, request["choices"][key]) for key in constraint["values"]]
    return [(value, value) for value in constraint["values"]]


class HFSAAdapter:
    def __init__(self, classification: Any, detection: Any, counting: Any, refseg: Any,
                 translate_prompt: Callable[[str], tuple[str, bool]]) -> None:
        self.classification = classification
        self.detection = detection
        self.counting = counting
        self.refseg = refseg
        self.translate_prompt = translate_prompt

    @classmethod
    def from_local(cls) -> HFSAAdapter:
        code_dir = Path(__file__).resolve().parents[1]
        weights_dir = Path(os.environ.get("HFSA_MODELS_DIR", str(code_dir.parent / "HFSA_models"))).expanduser().resolve()
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from tasks.classification.inference import SceneClassificationPredictor
        from tasks.counting.inference import CountingPredictor
        from tasks.detection.inference import TextGuidedDetectionPredictor, TextPromptEncoder
        from tasks.refseg.inference import RefSegPredictor
        from tasks.routing.adapters.refseg import translate_refseg_prompt

        device = os.environ.get("HFSA_API_DEVICE", "auto").strip() or "auto"
        checkpoints = {
            "classification": weights_dir / "runs/classification/vrsbench_scene/weights/best.pt",
            "detection": weights_dir / "runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt",
            "counting": weights_dir / "runs/counting/counting-VRSBench-ViT-L-14/weights/best.pt",
            "refseg": weights_dir / "runs/semseg/srp_yolov12m_axis/weights/best_raw.pt",
            "pretrained": weights_dir / "pretrain_model/yolov12m.pt",
        }
        for name, path in checkpoints.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} weights: {path}")

        # Share the existing global-text encoder using its supported factory hook.
        shared_encoder: Any = None
        encoder_config: dict[str, Any] | None = None

        def encoder_factory(**kwargs: Any) -> Any:
            nonlocal shared_encoder, encoder_config
            if shared_encoder is None:
                shared_encoder = TextPromptEncoder(**kwargs)
                encoder_config = kwargs
            elif kwargs != encoder_config:
                raise ValueError("Detection and counting text encoder configurations differ")
            return shared_encoder

        print("HFSA API: loading scene classification", flush=True)
        classification = SceneClassificationPredictor(
            checkpoint=checkpoints["classification"], pretrained_weights=checkpoints["pretrained"],
            model_yaml=str(code_dir / "ultralytics/cfg/models/v12/yolov12m-classification.yaml"), device=device,
        )
        print("HFSA API: loading detection and shared text encoder", flush=True)
        detection = TextGuidedDetectionPredictor(
            checkpoint=checkpoints["detection"], device=device, prompt_encoder_factory=encoder_factory,
        )
        print("HFSA API: loading counting", flush=True)
        counting = CountingPredictor(
            checkpoint=checkpoints["counting"], device=device, prompt_encoder_factory=encoder_factory,
        )
        print("HFSA API: loading referring segmentation", flush=True)
        refseg = RefSegPredictor(
            checkpoint=checkpoints["refseg"], device=device,
            model_yaml=str(code_dir / "ultralytics/cfg/models/v12/yolov12m-semseg.yaml"),
        )
        return cls(classification, detection, counting, refseg, translate_refseg_prompt)

    def _target(self, text: str) -> str:
        if not text.strip():
            raise AdapterFailure("missing_target", "Missing target description")
        simple = _TARGET_LOOKUP.get(normalize_label(text))
        if simple:
            return simple
        translated, _ = self.translate_prompt(text)
        if not translated.strip():
            raise AdapterFailure("missing_target", "Missing target description")
        return translated

    @staticmethod
    def _check_category(target: str) -> None:
        # Detection/counting checkpoints cover named remote-sensing categories.
        # Keep qualifying words intact, but reject abstract/general VQA targets.
        normalized = normalize_label(target)
        labels = {
            label for alias, label in _TARGET_LOOKUP.items()
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", normalized)
        }
        if len(labels) != 1:
            raise AdapterFailure("unsupported_target_category", "Detection/counting require one supported concrete target category")

    def _scene(self, request: dict[str, Any], image: Path) -> str:
        kind = request["response_constraint"]["type"]
        if kind == "short_text":
            return str(self.classification.predict(image, topk=1)["predictions"][0]["class_name"])
        candidates = _constrained_candidates(request)
        mapped = [(key, _SCENE_LOOKUP.get(normalize_label(text))) for key, text in candidates]
        available = set(self.classification.class_names)
        if any(label is None or label not in available for _, label in mapped):
            raise AdapterFailure("unsupported_scene_options", "Scene options include a class unsupported by the checkpoint")
        if len({label for _, label in mapped}) != len(mapped):
            raise AdapterFailure("ambiguous_scene_options", "Scene options map ambiguously to the same class")
        predictions = self.classification.predict(image, topk=len(available))["predictions"]
        scores = {row["class_name"]: row["probability"] for row in predictions}
        return max(mapped, key=lambda item: scores[item[1]])[0]

    @staticmethod
    def _number(request: dict[str, Any], count: int) -> str:
        answer = str(count)
        constraint = request["response_constraint"]
        if constraint["type"] in {"integer", "short_text"}:
            if count < constraint.get("minimum", 0):
                raise AdapterFailure("count_below_minimum", "Predicted count violates the answer minimum")
            return answer
        matches = []
        for key, value in _constrained_candidates(request):
            numeric = re.fullmatch(r"\s*(\d+)\s*(?:个|架|艘|辆|座|处|只)?\s*", value)
            if numeric and int(numeric.group(1)) == count:
                matches.append(key)
        if len(matches) != 1:
            raise AdapterFailure("count_option_unmatched", "Predicted count has no unique matching answer option")
        return matches[0]

    @staticmethod
    def _presence(request: dict[str, Any], found: bool) -> str:
        if request["response_constraint"]["type"] == "short_text":
            return "Yes" if found else "No"
        yes, no = {"yes", "true", "是", "有", "存在"}, {"no", "false", "否", "无", "没有", "不存在"}
        candidates = _constrained_candidates(request)
        if any(normalize_label(text) not in yes | no for _, text in candidates):
            raise AdapterFailure("unsupported_presence_options", "Presence options must express Yes/No")
        matches = [key for key, text in candidates if normalize_label(text) in (yes if found else no)]
        if len(matches) != 1:
            raise AdapterFailure("ambiguous_presence_options", "No unique Yes/No option for the model result")
        return matches[0]

    def predict(self, request: dict[str, Any], image_paths: list[Path]) -> str | list[int]:
        started = time.monotonic()
        plan = None
        _capture_replay_request(request)
        try:
            if len(image_paths) != 1:
                raise AdapterFailure("unsupported_image_count", "HFSA API currently supports one image; two-image change/VQA tasks are unsupported")
            plan = parse_question(request)
            answer = self._predict_plan(request, image_paths[0], plan)
        except Exception as error:
            _prediction_diagnostic(request, len(image_paths), plan, started, error)
            raise
        _prediction_diagnostic(request, len(image_paths), plan, started)
        return answer

    def _predict_plan(self, request: dict[str, Any], image: Path,
                      plan: QuestionPlan) -> str | list[int]:
        if plan.task == "classification":
            return self._scene(request, image)
        target = self._target(plan.target)
        if plan.task != "refseg":
            self._check_category(target)
        if plan.task == "counting":
            result = self.counting.predict(image, target)
            return self._number(request, result.count)
        if plan.task == "presence":
            result = self.detection.predict(image, target)
            return self._presence(request, result.detection_count > 0)

        height, width = request["image_height"], request["image_width"]
        if plan.task == "refseg":
            result = self.refseg.predict(image, target)
            if result.mask.shape != (height, width):
                raise AdapterFailure("image_dimensions_mismatch", "Request dimensions differ from the original image")
            ys, xs = result.mask.nonzero()
            if len(xs) == 0:
                raise AdapterFailure("empty_refseg_mask", "RefSeg did not find the target")
            return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        result = self.detection.predict(image, target)
        if result.original_size != (height, width):
            raise AdapterFailure("image_dimensions_mismatch", "Request dimensions differ from the original image")
        if not result.detection_count:
            raise AdapterFailure("empty_detections", "Detector did not find the target")
        box = result.boxes[int(result.scores.argmax())]
        if not all(math.isfinite(float(value)) for value in box):
            raise AdapterFailure("non_finite_bbox", "Detector returned non-finite coordinates")
        answer = [max(0, math.floor(float(box[0]))), max(0, math.floor(float(box[1]))),
                  min(width, math.ceil(float(box[2]))), min(height, math.ceil(float(box[3])))]
        if answer[0] >= answer[2] or answer[1] >= answer[3]:
            raise AdapterFailure("empty_bbox", "Detector returned an empty bbox")
        return answer
