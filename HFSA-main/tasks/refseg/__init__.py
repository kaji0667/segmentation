"""Text-guided referring-segmentation task package."""

from .application import RefSegEvaluationApplication, RefSegTrainingApplication

__all__ = (
    "RefSegEvaluationApplication",
    "RefSegPrediction",
    "RefSegPredictor",
    "RefSegTrainingApplication",
)


def __getattr__(name):
    if name in {"RefSegPrediction", "RefSegPredictor"}:
        from .inference import RefSegPrediction, RefSegPredictor

        return {"RefSegPrediction": RefSegPrediction, "RefSegPredictor": RefSegPredictor}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
