"""Shared image-manifest + standalone copy-script helper for exports whose
output lives in its own directory, separate from the source images
(``export voc``, ``export yolo``).

Every such export also writes ``image_manifest.csv`` (every distinct source
image it referenced, mapped to a collision-safe destination filename via
``mbariml.image_naming.export_stem_map``) plus a standalone
``copy_images.py`` next to it. Running that script pulls the actual images
down to a destination directory (default ``~/Desktop/<export_name>_images``,
or a required ``--dest`` when the caller passes ``require_dest=True``) -- handy for building a self-contained, trainable dataset out of an export
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

from mbariml.image_naming import export_stem_map
from mbariml.logging_utils import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "image_manifest.csv"
COPY_SCRIPT_NAME = "copy_images.py"

_COPY_SCRIPT_TEMPLATE = string.Template('''\
#!/usr/bin/env python3
"""Copy every source image listed in image_manifest.csv (written alongside
this script by `mbariml export $export_name`) to a destination directory.

Standalone stdlib script -- no mbariml install required to run this.

    python3 copy_images.py $usage_dest

$dest_doc
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCRIPT_DIR / "image_manifest.csv"
$default_dest_const

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    $dest_argument
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
    image_paths: Iterable[Path | str], output_dir: Path, export_name: str,
    stem_map: dict[str, str] | None = None, require_dest: bool = False,
) -> tuple[Path, Path]:
    """Write ``image_manifest.csv`` and ``copy_images.py`` into
    ``output_dir``. ``image_paths`` may repeat/be unordered -- deduplicated
    and sorted here. Returns ``(manifest_path, script_path)``.

    ``stem_map`` is the caller's :func:`mbariml.image_naming.export_stem_map`
    result. Callers that name other artifacts per image (``export yolo``'s
    label files and split lists) MUST pass the same map they used there --
    the manifest decides what ``copy_images.py`` copies each image TO, so a
    map computed twice, or computed here over a different set of paths, is
    how ``images/X.jpg`` stops lining up with ``labels/X.txt``.

    ``require_dest`` makes ``copy_images.py``'s ``--dest`` (alias
    ``--output-dir``) required instead of defaulting to the Desktop -- for
    exports whose images only work in one place (``export yolo``'s
    ``<output_dir>/images``, next to ``labels/``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    distinct_paths = sorted({Path(p) for p in image_paths}, key=str)
    if stem_map is None:
        stem_map = export_stem_map(distinct_paths)

    manifest_path = output_dir / MANIFEST_NAME
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_path", "dest_filename"])
        for image_path in distinct_paths:
            dest_filename = f"{stem_map[str(image_path)]}{image_path.suffix}"
            writer.writerow([str(image_path), dest_filename])

    script_path = output_dir / COPY_SCRIPT_NAME
    if require_dest:
        dest_fields = dict(
            usage_dest="--dest DIR",
            dest_doc="--dest (alias --output-dir) is required.",
            default_dest_const="",
            dest_argument='parser.add_argument("--dest", "--output-dir", required=True, '
                          'help="Destination directory (required)")',
        )
    else:
        default_dest_name = f"{export_name}_images"
        dest_fields = dict(
            usage_dest="[--dest DIR]",
            dest_doc=f"Default destination: ~/Desktop/{default_dest_name}",
            default_dest_const=f'DEFAULT_DEST = Path.home() / "Desktop" / "{default_dest_name}"\n',
            dest_argument='parser.add_argument("--dest", default=str(DEFAULT_DEST), '
                          'help="Destination directory (default: %(default)s)")',
        )
    script_path.write_text(_COPY_SCRIPT_TEMPLATE.substitute(export_name=export_name, **dest_fields))

    run_hint = f"python3 {script_path} --dest DIR" if require_dest else f"python3 {script_path}"
    logger.info(
        "Wrote image manifest (%d image(s)) to %s -- run `%s` to copy them.",
        len(distinct_paths), manifest_path, run_hint,
    )
    return manifest_path, script_path
