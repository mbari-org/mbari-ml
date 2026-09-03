"""Naming helper shared by every step that flattens per-image output files
(export XML/label files, image manifests, the stats matrix) into one
directory.

Two images with the same filename in different survey/dive subdirectories
(e.g. ``dive01/img_0001.jpg`` and ``dive02/img_0001.jpg``) collide once
flattened into a single output directory -- this is the same bug class
``export_voc`` was fixed for (grouping by full ``image_path`` instead
of bare filename; see its docstring and the README's "What changed"
section). Every place that writes/names one output file per source image
should go through this so the fix lives in one place.
"""

from __future__ import annotations

from pathlib import Path


def disambiguated_stem(image_path: Path) -> str:
    """``<parent_dir_name>_<stem>`` -- collision-safe once flattened, even
    across nested per-dive subdirectories that share a filename."""
    return f"{image_path.parent.name}_{image_path.stem}"
