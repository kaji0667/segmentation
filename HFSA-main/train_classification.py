"""Thin training entry for scene classification."""

from tasks.classification.train import SceneClassificationTrainingApplication


def main() -> None:
    SceneClassificationTrainingApplication.from_cli().run()


if __name__ == "__main__":
    main()
