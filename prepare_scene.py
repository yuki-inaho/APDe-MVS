#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Utility script to normalise dataset layouts before running APDe-MVS.

Example:

    python prepare_scene.py \
        --data-dir /path/to/scene \
        --image-dir-name images undist/images \
        --ensure-symlink
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List

from scripts.dataset_loader import DatasetLayoutConfig, SceneDatasetLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensure APDe-MVS compatible layout for one or more scans."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory that contains scan subdirectories or a single scan path.",
    )
    parser.add_argument(
        "--scans",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of scan names. If omitted, all subdirectories are processed.",
    )
    parser.add_argument(
        "--image-dir-name",
        type=str,
        nargs="+",
        default=["images", "undist/images"],
        help="Candidate image directory names to probe.",
    )
    parser.add_argument(
        "--image-suffixes",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="Accepted image filename suffixes.",
    )
    parser.add_argument(
        "--ensure-symlink",
        action="store_true",
        help="Create an `images/` symlink when the dataset stores images elsewhere.",
    )
    return parser.parse_args()


def discover_scans(data_dir: str, scans: Iterable[str] | None) -> List[str]:
    if scans:
        return [os.path.join(data_dir, scan) for scan in scans]

    if os.path.isdir(os.path.join(data_dir, "images")):
        # Treat the provided directory as a single scan.
        return [data_dir]

    candidates = [
        os.path.join(data_dir, entry)
        for entry in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, entry))
    ]
    return candidates


def main() -> int:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    scan_paths = discover_scans(data_dir, args.scans)

    if not scan_paths:
        print("スキャンが見つかりませんでした。")
        return 1

    layout_config = DatasetLayoutConfig(
        image_dir_candidates=args.image_dir_name,
        image_suffixes=args.image_suffixes,
        create_symlink=args.ensure_symlink,
    )

    for scan_path in scan_paths:
        loader = SceneDatasetLoader(scan_path, layout_config)
        scan_name = os.path.basename(scan_path.rstrip(os.sep))
        try:
            if args.ensure_symlink:
                canonical_dir = loader.ensure_standard_image_dir()
            else:
                canonical_dir = loader.resolve_image_dir()
            images = loader.list_images()
        except (FileNotFoundError, FileExistsError) as exc:
            print(f"[{scan_name}] 準備失敗: {exc}")
            continue

        print(f"[{scan_name}] 画像枚数: {len(images)}")
        print(f"  検出ディレクトリ: {loader.resolve_image_dir()}")
        if args.ensure_symlink:
            print(f"  標準ディレクトリ: {canonical_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
