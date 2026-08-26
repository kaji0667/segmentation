"""Thin evaluation entry for scene classification."""

from tasks.classification.test import SceneClassificationEvaluationApplication


def main() -> None:
    SceneClassificationEvaluationApplication.from_cli().run()


if __name__ == "__main__":
    main()
