"""Evaluate text-guided referring segmentation."""

import sys

from tasks.refseg import RefSegEvaluationApplication


def main() -> None:
    RefSegEvaluationApplication.from_cli(sys.argv[1:]).run()


if __name__ == "__main__":
    main()
