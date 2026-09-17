"""Emit: export curated labels to Pascal VOC XML annotation files.

Selects every VERIFIED localization and names it ``new_label`` where the
reviewer retyped it, the original detector ``label`` where they confirmed
it unchanged -- ``mbariml.db.curated_where``, the rule shared by every
export.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import typer
from tqdm import tqdm

from mbariml import db
from mbariml.export_common import write_image_manifest_and_script
from mbariml.image_naming import export_stem_map
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Export curated labels to Pascal VOC XML annotation files.")
logger = get_logger(__name__)


def _export_to_pascal_voc(conn, output_dir: Path) -> tuple[int, list[str]]:
    """Bug fixed here: this used to group detections by bare image_name and
    reconstruct the path as image_dir/image_name. For a mission with nested
    per-dive subdirectories, two images with the same filename in different
    dives collided into one key and both got mapped onto whichever single
    flat path happened to exist, silently merging detections from different
    dives onto the wrong image. Grouping is now by the full image_path
    already recorded at detection time, which is unambiguous."""
    voc_dir = output_dir / "pascal_voc"
    voc_dir.mkdir(parents=True, exist_ok=True)

    # `WHERE new_label != 'noise'` looked permissive but was not: SQL
    # three-valued logic makes `NULL != 'noise'` evaluate to NULL, not TRUE,
    # so every verified-but-unrelabelled row was silently dropped here too --
    # the same defect as the yolo/id exports, just wearing a disguise. Now
    # the one shared rule from mbariml.db.
    predictions = conn.execute(
        f"""
        SELECT image_path, {db.EFFECTIVE_LABEL_SQL}, confidence, x_min, y_min, x_max, y_max
        FROM predictions
        {db.curated_where()}
        """
    ).fetchall()

    grouped: dict[str, list[dict]] = {}
    for image_path, new_label, confidence, x_min, y_min, x_max, y_max in predictions:
        grouped.setdefault(image_path, []).append({
            "name": new_label,
            "confidence": confidence,
            "xmin": int(x_min), "ymin": int(y_min),
            "xmax": int(x_max), "ymax": int(y_max),
        })

    # One map for the whole export, same rule as every other export: the
    # image's own filename, prefixed by its parent directory only where two
    # source images would otherwise write to the same XML file.
    stem_map = export_stem_map(grouped)

    written = 0
    missing = 0
    for image_path_str, boxes in tqdm(grouped.items(), desc="Exporting to Pascal VOC"):
        image_path = Path(image_path_str)
        if not image_path.exists():
            missing += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            missing += 1
            continue
        height, width = image.shape[:2]

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = image_path.parent.name
        ET.SubElement(annotation, "filename").text = image_path.name
        ET.SubElement(annotation, "path").text = str(image_path)
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Unknown"
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(annotation, "segmented").text = "0"

        for box in boxes:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = box["name"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "occluded").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            ET.SubElement(obj, "confidence").text = str(box["confidence"])
            bndbox = ET.SubElement(obj, "bndbox")
            for tag in ("xmin", "ymin", "xmax", "ymax"):
                ET.SubElement(bndbox, tag).text = str(box[tag])

        # Prefix with the parent directory name too: two dives can each have
        # their own "img_0001.jpg", and a flat output directory needs unique
        # filenames even after the grouping fix above.
        xml_name = f"{stem_map[image_path_str]}.xml"
        ET.ElementTree(annotation).write(voc_dir / xml_name, xml_declaration=True, encoding="utf-8")
        written += 1

    if missing:
        logger.warning("%d image(s) recorded in the database could not be found on disk and were skipped.", missing)
    return written, list(grouped.keys())


def _export_new_names(conn, output_dir: Path) -> Path:
    """The distinct labels actually used by the XML files written above.

    Must apply exactly the same rule as _export_to_pascal_voc, or this file
    describes a different label set than the annotations beside it. Sorted
    so re-exporting the same database produces a byte-identical list.
    """
    names = conn.execute(
        f"""
        SELECT DISTINCT {db.EFFECTIVE_LABEL_SQL} AS name
        FROM predictions
        {db.curated_where()}
        ORDER BY name
        """
    ).fetchall()
    new_names_file = output_dir / "new_names.txt"
    new_names_file.write_text("\n".join(name for (name,) in names) + "\n")
    return new_names_file


@app.command()
def export_voc(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    output_dir: str = typer.Argument(
        ..., help="Directory to write pascal_voc/, new_names.txt, and the image manifest/copy script into."
    ),
) -> None:
    """Export curated labels to Pascal VOC XML files, plus a list of the distinct new labels used.

    Images are located via the path recorded at detection time -- no
    separate image directory argument needed (removed in v0.3.0; it was a
    source of the image-name-collision bug described above). Also writes
    image_manifest.csv and copy_images.py (see mbariml.export_common) so the
    matching source images can be pulled down later, e.g. to Desktop.
    """
    output_dir_path = Path(output_dir)

    with db.connect(db_path, must_exist=True) as conn:
        db.require_verified_column(conn, db_path)
        written, image_paths = _export_to_pascal_voc(conn, output_dir_path)
        names_file = _export_new_names(conn, output_dir_path)

    write_image_manifest_and_script(image_paths, output_dir_path, export_name="voc")

    logger.info("Wrote %d Pascal VOC annotation file(s) to %s", written, output_dir_path / "pascal_voc")
    logger.info("Wrote distinct label list to %s", names_file)


if __name__ == "__main__":
    app()
