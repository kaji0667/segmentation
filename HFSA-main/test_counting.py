"""Thin evaluation entry for text-guided object counting."""

from tasks.counting.test import CountingEvaluationApplication


def main() -> None:
    CountingEvaluationApplication.from_cli().run()


if __name__ == "__main__":
    main()
