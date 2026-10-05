"""Import: bring an existing YOLO-format dataset into a curation database.

The reverse of ``export yolo``: one ``<stem>.txt`` per image, each line
``class_id x_center y_center width height`` normalized to [0, 1], plus a
names file giving the class name for each index. Also accepted, so the
output of other tools imports without conversion:

- a 6th value, ``... height confidence`` (Ultralytics ``save_conf``), stored
  as the row's confidence;
- segmentation polygons, ``class_id x1 y1 x2 y2 ...``, imported as their
  bounding box.

Each label file is paired with its image by relative path first
(``labels/train/a.txt`` <-> ``images/train/a.jpg``, the Ultralytics layout),
then by bare filename stem when that's unambiguous -- see
``mbariml.import_common.ImageIndex``. Every label file is parsed and checked
against the names file BEFORE anything is written, so a names file that
doesn't match the labels fails immediately instead of after half an import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mbariml.import_common import ImageIndex, ImportedBox, ImportedImage, import_annotations
from mbariml.logging_utils import get_logger
from mbariml.yolo_utils import read_names_file

app = typer.Typer(help="Import a YOLO-format dataset (images + label .txt files + names file).")
logger = get_logger(__name__)

# Label files darknet-style datasets keep beside the labels that aren't labels.
_NON_LABEL_FILES = {"classes.txt", "names.txt", "notes.txt"}


def _parse_line(line: str, names: list[str], where: str) -> ImportedBox:
    parts = line.split()
    try:
        class_id = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
    except ValueError:
        raise typer.BadParameter(f"{where}: not a YOLO label line: {line!r}")

    if not 0 <= class_id < len(names):
        raise typer.BadParameter(
            f"{where}: class {class_id} is out of range for the names file ({len(names)} name(s), "
            f"indices 0-{len(names) - 1}). Is it the right names file for these labels?"
        )

    confidence: Optional[float] = None
    if len(values) in (4, 5):
        cx, cy, w, h = values[:4]
        if len(values) == 5:
            confidence = values[4]
        x_min, x_max = cx - w / 2, cx + w / 2
        y_min, y_max = cy - h / 2, cy + h / 2
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    else:
        raise typer.BadParameter(
            f"{where}: expected 'class cx cy w h [conf]' or a polygon 'class x1 y1 x2 y2 ...', got {line!r}"
        )

    return ImportedBox(
        label=names[class_id], class_id=class_id, confidence=confidence,
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
    )


def _parse_labels(labels_dir: Path, names: list[str], names_file: Path, limit: Optional[int]) -> list[tuple[Path, str, list[ImportedBox]]]:
    """(label_file, relative stem, boxes) for every label file, fully parsed."""
    skip = {names_file.resolve()}
    label_files = sorted(
        f for f in labels_dir.rglob("*.txt")
        if f.is_file() and f.resolve() not in skip and f.name.lower() not in _NON_LABEL_FILES
    )
    if not label_files:
        raise typer.BadParameter(f"No YOLO label (.txt) files found under {labels_dir} (searched recursively).")
    if limit:
        label_files = label_files[:limit]

    parsed = []
    for label_file in label_files:
        boxes = []
        for line_number, line in enumerate(label_file.read_text().splitlines(), start=1):
            if line.strip():
                boxes.append(_parse_line(line, names, f"{label_file}:{line_number}"))
        rel_stem = label_file.relative_to(labels_dir).with_suffix("").as_posix()
        parsed.append((label_file, rel_stem, boxes))
    return parsed


@app.command()
def import_yolo(
    images_dir: str = typer.Argument(..., help="Directory of images (searched recursively)."),
    labels_dir: str = typer.Argument(..., help="Directory of YOLO label .txt files, one per image (searched recursively)."),
    names_file: str = typer.Argument(
        ..., help="Class names in class-index order: a names.txt (one per line) or an Ultralytics dataset .yaml."
    ),
    output_dir: str = typer.Argument(
        ..., help="Directory for yolo_predictions.duckdb -- created if new, appended to if it already exists."
    ),
    verified: bool = typer.Option(
        True, "--verified/--unverified",
        help="Mark imported boxes verified (human-labeled ground truth -- what exports select) or "
             "unverified (e.g. another model's predictions, to review first).",
    ),
    skip_existing: bool = typer.Option(
        True, "--skip-existing/--no-skip-existing",
        help="Skip images already in the database, so re-running an import doesn't duplicate its boxes.",
    ),
    limit: Optional[int] = typer.Option(None, help="Import only the first N label files (for a quick test run)."),
) -> None:
    """Import a YOLO dataset's images and labels into OUTPUT_DIR/yolo_predictions.duckdb.

    Every box becomes an ordinary localization with its ROI crop stored, so
    review, embed, cluster and every export work on it exactly as on
    detector output. Images with an empty label file (YOLO's background
    images) have nothing to store and are counted, not imported.
    """
    labels_path = Path(labels_dir)
    names_path = Path(names_file)
    if not labels_path.is_dir():
        raise typer.BadParameter(f"Labels directory not found: {labels_path}")
    if not names_path.is_file():
        raise typer.BadParameter(f"Names file not found: {names_path}")

    names = read_names_file(names_path)
    if not names:
        raise typer.BadParameter(f"Names file {names_path} lists no class names.")
    logger.info("%d class name(s) from %s", len(names), names_path)

    parsed = _parse_labels(labels_path, names, names_path, limit)
    logger.info("Parsed %d label file(s), %d box(es)", len(parsed), sum(len(b) for _, _, b in parsed))

    index = ImageIndex(images_dir)
    logger.info("Found %d image(s) under %s", len(index.paths), images_dir)

    items = [
        ImportedImage(annotation_path=label_file, image_path=index.find(rel_stem), boxes=boxes, normalized=True)
        for label_file, rel_stem, boxes in parsed
    ]
    if index.ambiguous:
        logger.warning(
            "%d label file(s) matched more than one image by filename and were not paired -- keep "
            "labels/ mirroring images/ subdirectories so each pairs by relative path.", index.ambiguous,
        )
    if limit is None:
        matched = {item.image_path for item in items if item.image_path}
        unlabeled = len(index.paths) - len(matched)
        if unlabeled:
            logger.info("%d image(s) have no label file (YOLO background images) -- nothing to import.", unlabeled)

    import_annotations(
        items, output_dir,
        source_description=f"imported (yolo) from {labels_path.resolve()}",
        verified=verified, skip_existing=skip_existing, names=names,
    )


if __name__ == "__main__":
    app()
