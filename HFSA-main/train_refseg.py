"""Train text-guided referring segmentation."""

from tasks.refseg import RefSegTrainingApplication


def main() -> None:
    RefSegTrainingApplication.from_cli().run()


if __name__ == "__main__":
    main()
