"""Thin class-based entry applications for referring segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import engine


@dataclass
class RefSegTrainingApplication:
    """Run the established referring-segmentation training pipeline."""

    argv: Sequence[str] | None = None

    @classmethod
    def from_cli(cls, argv: Sequence[str] | None = None) -> "RefSegTrainingApplication":
        return cls(argv=argv)

    def run(self) -> None:
        arguments = list(self.argv) if self.argv is not None else None
        if arguments is not None and "--eval-only" in arguments:
            raise ValueError("Use test_refseg.py for evaluation-only execution.")
        engine.main(arguments)


@dataclass
class RefSegEvaluationApplication:
    """Evaluate one referring-segmentation checkpoint on the test split."""

    argv: Sequence[str] | None = None

    @classmethod
    def from_cli(cls, argv: Sequence[str] | None = None) -> "RefSegEvaluationApplication":
        return cls(argv=argv)

    def run(self) -> None:
        arguments = list(self.argv) if self.argv is not None else []
        if "--eval-only" not in arguments:
            arguments.insert(0, "--eval-only")
        engine.main(arguments)
