"""Export *.id identification sidecar files into the mission directory structure.

For every source image that has at least one curated identification
(``new_label`` set, excluding ``noise`` -- the same convention as the
Pascal VOC export in step 6), writes a ``<image_stem>.id`` file *next to
that image* (wherever it actually lives on disk -- this walks whatever path
was recorded in ``image_path`` at detection time, so it naturally follows
the mission's own directory structure without needing a separate
``image_dir`` argument).

Each bounding box is written as a 4-vertex polygon (top-left, top-right,
bottom-right, bottom-left), with pixel coordinates filled in immediately and
lon/lat/depth left as ``0.0`` placeholders -- per-vertex geolocation is
expected to be appended later by a separate process once navigation data is
available, by re-parsing/updating these same files.

Not part of the numbered step chain (like ``backfill-sharpness``) -- run it
directly whenever you want fresh sidecar files for a curated database.
"""

from __future__ import annotations

import getpass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from mbariml import __version__
from mbariml import db
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Export *.id identification sidecar files next to each source image.")
logger = get_logger(__name__)


def _resolve_model_description(conn, model_override: Optional[str]) -> str:
    if model_override:
        return model_override
    row = conn.execute("SELECT model_path FROM run_info ORDER BY detected_at DESC LIMIT 1").fetchone()
    return row[0] if row and row[0] else "unknown"


def _format_vertex(x: float, y: float) -> str:
    # Pixel coordinates now; lon/lat/depth are placeholders appended later.
    return f"{int(round(x))},{int(round(y))},0.0,0.0,0.0"


def _build_id_file_content(
    image_name: str, model_desc: str, username: str, generated_at: str, detections: list[tuple]
) -> str:
    lines = [
        "# mbariml identification file",
        f"# generator: mbariml v{__version__}",
        f"# generated_by: {username}",
        f"# generated_at: {generated_at}",
        f"# model: {model_desc}",
        f"# source_image: {image_name}",
        f"# count: {len(detections)}",
        "#",
        "# index label confidence  vertices(TL,TR,BR,BL as px_x,px_y,lon,lat,depth)",
    ]
    for i, (label, confidence, x_min, y_min, x_max, y_max) in enumerate(detections):
        vertices = "  ".join([
            _format_vertex(x_min, y_min),  # top-left
            _format_vertex(x_max, y_min),  # top-right
            _format_vertex(x_max, y_max),  # bottom-right
            _format_vertex(x_min, y_max),  # bottom-left
        ])
        lines.append(f"{i} {label} {confidence:.4f}  {vertices}")
    return "\n".join(lines) + "\n"


@app.command()
def export_ids(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    model: Optional[str] = typer.Option(
        None, help="Model identifier to record in each file's header. Defaults to what "
        "`mbariml detect` recorded for this database, or 'unknown' if that's not available."
    ),
) -> None:
    """Write a *.id sidecar file next to each source image with at least one curated identification."""
    with db.connect(db_path) as conn:
        model_desc = _resolve_model_description(conn, model)

        rows = conn.execute(
            """
            SELECT image_path, new_label, confidence, x_min, y_min, x_max, y_max
            FROM predictions
            WHERE new_label IS NOT NULL AND new_label != 'noise'
            """
        ).fetchall()

    if not rows:
        logger.warning("No curated identifications found (new_label set, not 'noise'); no .id files written.")
        return

    grouped: dict[str, list[tuple]] = {}
    for image_path, label, confidence, x_min, y_min, x_max, y_max in rows:
        grouped.setdefault(image_path, []).append((label, confidence, x_min, y_min, x_max, y_max))

    username = getpass.getuser()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    written, failed = 0, 0
    for image_path, detections in tqdm(grouped.items(), desc="Writing .id files"):
        image_path_obj = Path(image_path)
        id_path = image_path_obj.with_suffix(".id")
        try:
            id_path.write_text(
                _build_id_file_content(image_path_obj.name, model_desc, username, generated_at, detections)
            )
            written += 1
        except OSError:
            failed += 1
            logger.exception("Failed to write %s", id_path)

    logger.info("Wrote %d .id file(s) (model: %s, user: %s).", written, model_desc, username)
    if failed:
        logger.error("%d .id file(s) failed to write -- see tracebacks above.", failed)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
