"""Naming helper shared by every step that flattens per-image output files
(export XML/label/.id files, image manifests, the stats matrix) into one
directory.

Output files are named after the source image's own filename -- that is what
anyone matching an export back to the imagery expects to see, and a
``PROSILICA_L_`` style prefix on every file is noise when the filenames are
already unique (survey imagery is typically named by timestamp, which it is).

The prefix is not merely cosmetic, though. Two images with the same filename
in different survey/dive subdirectories (``dive01/img_0001.jpg`` and
``dive02/img_0001.jpg``) collide the moment they are flattened into one
output directory, and the second silently overwrites the first -- the same
bug class ``export_voc`` was fixed for (grouping by full ``image_path``
instead of bare filename; see its docstring and the README's "What changed").

So naming is decided for the export as a WHOLE, by :func:`export_stem_map`:
plain filenames when they are unique, and the disambiguating parent-directory
prefix applied only to the ones that actually clash. Every place that writes
or names one output file per source image goes through that, so a collision
can never silently drop annotations and an export that has no collisions
never pays for the possibility.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from mbariml.logging_utils import get_logger

logger = get_logger(__name__)


def disambiguated_stem(image_path: Path) -> str:
    """``<parent_dir_name>_<stem>`` -- collision-safe once flattened, even
    across nested per-dive subdirectories that share a filename.

    Used only for the paths :func:`export_stem_map` finds to be ambiguous;
    call that instead of reaching for this directly.
    """
    return f"{image_path.parent.name}_{image_path.stem}"


def export_stem_map(image_paths: Iterable[Path | str]) -> dict[str, str]:
    """``{image_path_string: output_stem}`` for one export.

    A source image keeps its own bare filename stem, unless another image in
    the same export shares that stem -- then every image involved in that
    clash takes the ``<parent>_<stem>`` form instead, and the clash is
    logged. Only the colliding names are rewritten, so one unlucky pair in a
    10,000-image survey does not prefix the other 9,998.

    Decided across the whole export rather than per file, because that is the
    only level at which "is this name unique?" can be answered. Callers must
    use one map for every artifact they name (label files, the manifest, the
    split lists), or `images/X.jpg` stops lining up with `labels/X.txt`.
    """
    paths = [Path(p) for p in dict.fromkeys(str(p) for p in image_paths)]

    by_stem: dict[str, list[Path]] = defaultdict(list)
    for image_path in paths:
        by_stem[image_path.stem].append(image_path)

    stem_map: dict[str, str] = {}
    collisions: dict[str, list[Path]] = {}
    for stem, group in by_stem.items():
        if len(group) == 1:
            stem_map[str(group[0])] = stem
            continue
        collisions[stem] = group
        for image_path in group:
            stem_map[str(image_path)] = disambiguated_stem(image_path)

    if collisions:
        shown = "; ".join(
            f"{stem} ({len(group)} images)" for stem, group in list(collisions.items())[:5]
        )
        logger.warning(
            "%d filename(s) appear on more than one source image in this export, so those "
            "output files keep the <parent_dir>_<filename> prefix to avoid overwriting each "
            "other: %s%s", len(collisions), shown,
            ", ..." if len(collisions) > 5 else "",
        )

    # A disambiguated name could itself collide with a different image's bare
    # stem. Vanishingly unlikely, and silently losing a file to it would be
    # indistinguishable from the bug this function exists to prevent.
    if len(set(stem_map.values())) != len(stem_map):
        raise RuntimeError(
            "Could not assign unique output filenames: two source images resolve to the same "
            "name even after disambiguation. Rename one of them, or export them separately."
        )
    return stem_map
