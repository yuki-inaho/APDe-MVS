# SPDX-License-Identifier: MIT
"""
Flexible dataset loader utilities.

The functions and classes provided here make it easy to work with scenes whose
image directories or filename suffixes do not strictly follow the canonical
`images/*.jpg` layout expected by APDe-MVS.  They can discover alternate
directory names (e.g. ``undist/images``) and automatically create the standard
`images/` alias so that downstream tools continue to operate unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


def _normalize_suffixes(suffixes: Iterable[str]) -> List[str]:
    """Return lower-case suffixes with a leading dot."""
    normalized = []
    for suffix in suffixes:
        if not suffix:
            continue
        suffix = suffix.lower()
        if not suffix.startswith("."):
            suffix = "." + suffix
        normalized.append(suffix)
    return normalized


def _split_candidate(candidate: str) -> Sequence[str]:
    """Split candidate paths on forward slashes to support nested inputs."""
    return tuple(part for part in candidate.split("/") if part)


@dataclass
class DatasetLayoutConfig:
    """Configuration describing how to locate scene assets on disk."""

    image_dir_candidates: Sequence[str] = field(
        default_factory=lambda: ("images", "undist/images")
    )
    image_suffixes: Sequence[str] = field(
        default_factory=lambda: (".jpg", ".jpeg", ".png")
    )
    target_dir_name: str = "images"
    create_symlink: bool = True

    def normalized_suffixes(self) -> List[str]:
        return _normalize_suffixes(self.image_suffixes)


class SceneDatasetLoader:
    """
    Resolve and optionally normalise the layout of a single scan directory.

    Parameters
    ----------
    scan_dir:
        Absolute path to the scan root (e.g. ``/data/foo/scan_01``).
    config:
        per-scene layout configuration.  Defaults mirror the original APDe-MVS
        expectations, but can be customised on demand.
    """

    def __init__(self, scan_dir: str, config: DatasetLayoutConfig | None = None):
        self.scan_dir = os.path.abspath(scan_dir)
        self.config = config or DatasetLayoutConfig()
        self._image_dir: str | None = None

    # --------------------------------------------------------------------- #
    # Directory resolution & normalisation
    # --------------------------------------------------------------------- #
    def resolve_image_dir(self) -> str:
        """Return the source image directory, raising if it cannot be located."""
        if self._image_dir:
            return self._image_dir

        for candidate in self.config.image_dir_candidates:
            candidate_path = os.path.join(self.scan_dir, *_split_candidate(candidate))
            if os.path.isdir(candidate_path):
                self._image_dir = candidate_path
                return candidate_path

        raise FileNotFoundError(
            f"画像ディレクトリが見つかりません: 候補={self.config.image_dir_candidates} / "
            f"scan_dir={self.scan_dir}"
        )

    def ensure_standard_image_dir(self) -> str:
        """
        Ensure that ``scan_dir/target_dir_name`` exists.

        If the canonical directory is missing but an alternate source is
        available, create a symlink (when ``create_symlink`` is True) so that
        downstream code can continue to rely on the conventional path.
        """
        source_dir = self.resolve_image_dir()
        canonical_dir = os.path.join(self.scan_dir, self.config.target_dir_name)

        if os.path.isdir(canonical_dir):
            # canonical_dir が既存の場合は、元ディレクトリと同一かどうかを確認
            if os.path.samefile(source_dir, canonical_dir):
                return canonical_dir
            # ディレクトリは存在するが別物の場合はそのまま利用
            return canonical_dir

        if os.path.exists(canonical_dir) and not os.path.isdir(canonical_dir):
            raise FileExistsError(
                f"{canonical_dir} が既に存在していますがディレクトリではありません。"
            )

        if not self.config.create_symlink:
            raise FileNotFoundError(
                f"{canonical_dir} が存在しません。シンボリックリンク作成が無効化されています。"
            )

        os.symlink(source_dir, canonical_dir)
        return canonical_dir

    # --------------------------------------------------------------------- #
    # Image enumeration helpers
    # --------------------------------------------------------------------- #
    def list_images(self) -> List[str]:
        """
        List images in the resolved directory filtered by the configured suffixes.
        """
        image_dir = self.resolve_image_dir()
        suffixes = self.config.normalized_suffixes()
        images = [
            entry
            for entry in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, entry))
            and os.path.splitext(entry)[1].lower() in suffixes
        ]
        images.sort()
        return images

    def count_images(self) -> int:
        return len(self.list_images())

    def has_standard_layout(self) -> bool:
        """
        Return True if the canonical target directory already points to the
        actual image directory.
        """
        canonical_dir = os.path.join(self.scan_dir, self.config.target_dir_name)
        if not os.path.isdir(canonical_dir):
            return False
        try:
            return os.path.samefile(self.resolve_image_dir(), canonical_dir)
        except FileNotFoundError:
            return False
