"""Import: bring an existing Pascal VOC dataset into a curation database.

The reverse of ``export voc``: one XML file per image, each ``<object>``
holding a ``<name>`` and a ``<bndbox>`` in absolute pixel coordinates. A
``<confidence>`` element, which ``export voc`` writes, is kept as the row's
confidence, so a VOC export imports back with nothing lost; files from
other tools (LabelImg, CVAT, ...) simply don't have one.

Each XML is paired with its image by its ``<filename>`` -- in the same
relative subdirectory under IMAGES_DIR as the XML sits under XML_DIR, or
anywhere if that filename is unique (``<folder>`` breaking a tie) -- then
by the XML's own stem. The
``<path>`` element is ignored: it records where the image was on whichever
machine annotated it, and is wrong on every other one.

VOC has no class indices, so ``class_id`` is left NULL, as for a box drawn
in the review GUI.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import typer

from mbariml.import_common import ImageIndex, ImportedBox, ImportedImage, import_annotations
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Import a Pascal VOC dataset (images + XML annotation files).")
logger = get_logger(__name__)


def _float(element: Optional[ET.Element], tag: str) -> Optional[float]:
    text = element.findtext(tag) if element is not None else None
    try:
        return float(text) if text is not None and text.strip() else None
    except ValueError:
        return None


def _parse_xml(xml_file: Path) -> tuple[Optional[str], Optional[str], Optional[tuple[int, int]], list[ImportedBox], int]:
    """(filename, folder, declared (width, height), boxes, objects skipped) for one XML."""
    try:
        root = ET.parse(xml_file).getroot()
    except ET.ParseError as exc:
        raise typer.BadParameter(f"{xml_file}: not valid XML ({exc})")

    filename = (root.findtext("filename") or "").strip() or None
    folder = (root.findtext("folder") or "").strip() or None
    size = root.find("size")
    width, height = _float(size, "width"), _float(size, "height")
    declared = (int(width), int(height)) if width and height else None

    boxes = []
    skipped = 0
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        bndbox = obj.find("bndbox")
        coords = [_float(bndbox, tag) for tag in ("xmin", "ymin", "xmax", "ymax")]
        if not name or None in coords:
            skipped += 1
            continue
        x_min, y_min, x_max, y_max = coords
        boxes.append(ImportedBox(
            label=name, confidence=_float(obj, "confidence"),
            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        ))
    return filename, folder, declared, boxes, skipped


@app.command()
def import_voc(
    images_dir: str = typer.Argument(..., help="Directory of images (searched recursively)."),
    xml_dir: str = typer.Argument(..., help="Directory of Pascal VOC .xml files, one per image (searched recursively)."),
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
    limit: Optional[int] = typer.Option(None, help="Import only the first N XML files (for a quick test run)."),
) -> None:
    """Import a Pascal VOC dataset's images and XML annotations into OUTPUT_DIR/yolo_predictions.duckdb.

    Every box becomes an ordinary localization with its ROI crop stored, so
    review, embed, cluster and every export work on it exactly as on
    detector output.
    """
    xml_path = Path(xml_dir)
    if not xml_path.is_dir():
        raise typer.BadParameter(f"XML directory not found: {xml_path}")
    xml_files = sorted(f for f in xml_path.rglob("*") if f.is_file() and f.suffix.lower() == ".xml")
    if not xml_files:
        raise typer.BadParameter(f"No .xml files found under {xml_path} (searched recursively).")
    if limit:
        xml_files = xml_files[:limit]

    index = ImageIndex(images_dir)
    logger.info("Found %d image(s) under %s, %d XML file(s) under %s",
                len(index.paths), images_dir, len(xml_files), xml_path)

    items = []
    skipped_objects = 0
    for xml_file in xml_files:
        filename, folder, declared, boxes, skipped = _parse_xml(xml_file)
        skipped_objects += skipped
        rel_stem = xml_file.relative_to(xml_path).with_suffix("").as_posix()
        items.append(ImportedImage(
            annotation_path=xml_file, image_path=index.find(rel_stem, filename, folder),
            boxes=boxes, declared_size=declared,
        ))

    if skipped_objects:
        logger.warning("%d <object>(s) had no <name> or an incomplete <bndbox> and were skipped.", skipped_objects)
    if index.ambiguous:
        logger.warning(
            "%d XML file(s) matched more than one image by filename and were not paired -- keep the "
            "XML directory mirroring the image subdirectories so each pairs by relative path.", index.ambiguous,
        )

    import_annotations(
        items, output_dir,
        source_description=f"imported (voc) from {xml_path.resolve()}",
        verified=verified, skip_existing=skip_existing,
    )


if __name__ == "__main__":
    app()
