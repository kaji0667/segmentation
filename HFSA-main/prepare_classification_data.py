"""Prepare VRSBench scene-classification folders without personal paths."""

import argparse
import json

from classification import VRSBenchSceneDatasetBuilder


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--voc-root", default="data/VRSBench")
    parser.add_argument("--output-dir", default="data/VRSBench_scene")
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--max-per-class", type=int, default=0)
    args = parser.parse_args()
    report = VRSBenchSceneDatasetBuilder(
        args.voc_root, args.output_dir, link_mode=args.link_mode, max_per_class=args.max_per_class
    ).build()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
