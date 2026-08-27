"""Shared image-directory scanning.

The original scripts each reimplemented this, and none of them validated
that ``image_dir`` actually existed -- ``Path("typo/dir").rglob("*")`` just
silently yields nothing, so a mistyped path quietly produced an empty file
list rather than an error pointing at the typo.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif"})


def collect_images(
    image_dir: str | Path,
    limit: Optional[int] = None,
    extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    random_sample: bool = False,
    seed: Optional[int] = None,
) -> list[Path]:
    """Recursively collect image files under ``image_dir``.

    Raises ``FileNotFoundError``/``NotADirectoryError`` instead of returning
    an empty list, so a bad path fails loudly at the start of a run instead
    of producing a script that "ran" but did nothing.

    ``limit`` alone takes the first ``limit`` files in sorted path order --
    unchanged from the original behavior, and fine for a quick smoke test.
    For a survey directory with sequentially/chronologically named files,
    that's a contiguous time slice, not a representative sample; pass
    ``random_sample=True`` to instead take a seeded random sample of
    ``limit`` files spread across the whole directory. The result is still
    returned in sorted order either way, so downstream processing order
    doesn't depend on ``random_sample``. ``seed`` is only meaningful with
    ``random_sample=True``; the same directory contents + limit + seed
    always produce the same sample.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    exts = {e.lower() for e in extensions}
    files = sorted(f for f in image_dir.rglob("*") if f.is_file() and f.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(
            f"No images with extensions {sorted(exts)} found under {image_dir} (searched recursively)."
        )

    if not limit or limit >= len(files):
        return files
    if random_sample:
        return sorted(random.Random(seed).sample(files, limit))
    return files[:limit]
