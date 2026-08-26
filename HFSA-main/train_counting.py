"""Thin training entry for text-guided object counting."""

from tasks.counting.train import CountingTrainingApplication


def main() -> None:
    CountingTrainingApplication.from_cli().run()


if __name__ == "__main__":
    main()
