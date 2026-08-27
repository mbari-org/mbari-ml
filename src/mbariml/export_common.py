"""Shared image-manifest + standalone copy-script helper for exports whose
output lives in its own directory, separate from the source images
(``export voc``, ``export yolo``).

Every such export also writes ``image_manifest.csv`` (every distinct source
image it referenced, mapped to a collision-safe destination filename via
``mbariml.image_naming.disambiguated_stem``) plus a standalone
``copy_images.py`` next to it. Running that script pulls the actual images
down to a destination directory (default ``~/Desktop/<export_name>_images``)
-- handy for building a self-contained, trainable dataset out of an export
that (by design) only writes annotation files, not image copies, and handy
for grabbing a curated set of images to look at without walking the mission's
own nested directory structure by hand.

``copy_images.py`` is generated as a standalone stdlib-only script
(``argparse``/``csv``/``shutil``/``pathlib`` only) deliberately: it's meant to
be runnable later, from any machine that can see the recorded source paths,
without requiring mbariml (or even this repo) to be installed there.
"""

from __future__ import annotations

import csv
import string
from pathlib import Path
from typing import Iterable

from mbariml.image_naming import disambiguated_stem
from mbariml.logging_utils import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "image_manifest.csv"
COPY_SCRIPT_NAME = "copy_images.py"

_COPY_SCRIPT_TEMPLATE = string.Template('''\
#!/usr/bin/env python3
"""Copy every source image listed in image_manifest.csv (written alongside
this script by `mbariml export $export_name`) to a destination directory.

Standalone stdlib script -- no mbariml install required to run this.

    python3 copy_images.py [--dest DIR]

Default destination: ~/Desktop/$default_dest_name
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCRIPT_DIR / "image_manifest.csv"
DEFAULT_DEST = Path.home() / "Desktop" / "$default_dest_name"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Destination directory (default: %(default)s)")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        sys.exit(f"Manifest not found: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied, missing = 0, 0
    for i, row in enumerate(rows, start=1):
        src = Path(row["source_path"])
        if not src.exists():
            print(f"[{i}/{len(rows)}] MISSING: {src}")
            missing += 1
            continue
        shutil.copy2(src, dest_dir / row["dest_filename"])
        copied += 1
        if i % 100 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] {copied} copied so far...")

    print(f"Done: {copied} copied, {missing} missing (source not found), -> {dest_dir}")


if __name__ == "__main__":
    main()
''')


def write_image_manifest_and_script(
    image_paths: Iterable[Path | str], output_dir: Path, export_name: str
) -> tuple[Path, Path]:
    """Write ``image_manifest.csv`` and ``copy_images.py`` into
    ``output_dir``. ``image_paths`` may repeat/be unordered -- deduplicated
    and sorted here. Returns ``(manifest_path, script_path)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    distinct_paths = sorted({Path(p) for p in image_paths}, key=str)

    manifest_path = output_dir / MANIFEST_NAME
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_path", "dest_filename"])
        for image_path in distinct_paths:
            dest_filename = f"{disambiguated_stem(image_path)}{image_path.suffix}"
            writer.writerow([str(image_path), dest_filename])

    script_path = output_dir / COPY_SCRIPT_NAME
    script_path.write_text(
        _COPY_SCRIPT_TEMPLATE.substitute(
            export_name=export_name,
            default_dest_name=f"{export_name}_images",
        )
    )

    logger.info(
        "Wrote image manifest (%d image(s)) to %s -- run `python3 %s` to copy them to Desktop.",
        len(distinct_paths), manifest_path, script_path,
    )
    return manifest_path, script_path
