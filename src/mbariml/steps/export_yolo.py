"""Export curated labels to YOLO-format label files, plus a names file.

Writes one ``labels/<disambiguated_stem>.txt`` per source image with at
least one curated identification (``new_label`` set, excluding ``noise`` --
same convention as ``export voc``), each line
``class_id x_center y_center width height`` normalized to [0, 1] against
that image's actual pixel dimensions (read from the image itself, same as
``export voc``), plus ``names.txt`` (one label per line, in class-id order --
the file Ultralytics training configs expect for ``names:``).

Deliberately does not also write copies of the source images into an
``images/`` directory here -- label files are cheap text, but copying every
image would duplicate the same JPEGs already sitting on the survey volume
this ran against. Pairs with ``image_manifest.csv``/``copy_images.py``
(also written here, see ``mbariml.export_common``) to actually assemble a
self-contained, trainable ``images/`` + ``labels/`` dataset directory: run
``python3 copy_images.py --dest <output_dir>/images`` after this.

Images are located via the path recorded at detection time, the same as
``export voc``/``export html`` -- no separate image_dir argument needed.

One of the Emit-phase exports, alongside ``export voc``/``export id``/
``export html``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import typer
from tqdm import tqdm

from mbariml import db
from mbariml.export_common import write_image_manifest_and_script
from mbariml.image_naming import disambiguated_stem
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Export curated labels to YOLO-format label files, plus a names file.")
logger = get_logger(__name__)


def _fetch_curated(conn) -> list[tuple]:
    return conn.execute(
        """
        SELECT image_path, new_label, x_min, y_min, x_max, y_max
        FROM predictions
        WHERE new_label IS NOT NULL AND new_label != 'noise'
        """
    ).fetchall()


def _export_labels(conn, output_dir: Path) -> tuple[int, list[str], list[str]]:
    """Returns (files_written, sorted distinct names, distinct image_path strings)."""
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows = _fetch_curated(conn)
    names = sorted({new_label for _, new_label, *_ in rows})
    class_index = {name: i for i, name in enumerate(names)}

    grouped: dict[str, list[tuple]] = {}
    for image_path, new_label, x_min, y_min, x_max, y_max in rows:
        grouped.setdefault(image_path, []).append((new_label, x_min, y_min, x_max, y_max))

    written = 0
    missing = 0
    for image_path_str, boxes in tqdm(grouped.items(), desc="Exporting to YOLO"):
        image_path = Path(image_path_str)
        if not image_path.exists():
            missing += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            missing += 1
            continue
        height, width = image.shape[:2]

        lines = []
        for new_label, x_min, y_min, x_max, y_max in boxes:
            cx = ((x_min + x_max) / 2) / width
            cy = ((y_min + y_max) / 2) / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height
            lines.append(f"{class_index[new_label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path = labels_dir / f"{disambiguated_stem(image_path)}.txt"
        label_path.write_text("\n".join(lines) + "\n")
        written += 1

    if missing:
        logger.warning("%d image(s) recorded in the database could not be found on disk and were skipped "
                        "(their labels were not written -- normalizing boxes needs actual image dimensions).", missing)
    return written, names, list(grouped.keys())


def _write_names(names: list[str], output_dir: Path) -> Path:
    names_path = output_dir / "names.txt"
    names_path.write_text("\n".join(names) + ("\n" if names else ""))
    return names_path


@app.command()
def export_yolo(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    output_dir: str = typer.Argument(
        ..., help="Directory to write labels/, names.txt, and the image manifest/copy script into."
    ),
) -> None:
    """Export curated labels to YOLO-format label files, plus a names file.

    Images are located via the path recorded at detection time -- no
    separate image_dir argument needed (same convention as `export voc`).
    """
    output_dir_path = Path(output_dir)

    with db.connect(db_path) as conn:
        written, names, image_paths = _export_labels(conn, output_dir_path)

    names_path = _write_names(names, output_dir_path)
    write_image_manifest_and_script(image_paths, output_dir_path, export_name="yolo")

    logger.info("Wrote %d YOLO label file(s) to %s", written, output_dir_path / "labels")
    logger.info("Wrote %d distinct label(s) to %s", len(names), names_path)


if __name__ == "__main__":
    app()
