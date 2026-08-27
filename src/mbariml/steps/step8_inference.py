"""Step 8: run a trained YOLO model over a new batch of images.

Writes to the SAME curation schema as step 1 (ROI crop blob + sharpness,
with embedding/new_label left NULL) -- not a separate lighter schema, which
is what earlier versions of this file did. That mattered: `mbariml review`,
`cluster`, `refine`, `export voc`, and `remap-labels` all require columns
(`roi_index`, `roi`, `embedding`, `new_label`) that the old lighter schema
never had, so none of them could run against step 8's output at all. Now
they can -- run `mbariml infer-images` on a new survey, then go straight to
`mbariml embed`/`mbariml review`/`mbariml export html` on what it produced,
with no need to run step 1 first. This step is still fully standalone
otherwise: it doesn't need any earlier step's database to run.

ROIs are cropped directly from ``result.orig_img`` (the image array
Ultralytics already loaded for that prediction), not by re-reading each file
from disk a second time.

Bug fixed here (this was the root cause of "code seemed to run, but no
results saved, no db generated"): the previous implementation accumulated
*every* detection row from *every* batch in an in-memory list and only
called ``conn.executemany(...)`` once, after the entire image set had been
processed, with that single call sitting outside any try/except. YOLO would
visibly run and even save annotated images to disk via ``save=True``, but if
anything went wrong anywhere in that single final insert (or the process was
interrupted before reaching it), none of the detections ever reached the
database. Rows are now inserted -- and the connection is flushed via
``mbariml.db.connect``'s context manager -- after *every* batch, so progress
is durable as the run proceeds. Each batch's insert also goes through
``db.fast_executemany`` (an explicit transaction) rather than a bare
``conn.executemany``, since DuckDB's Python driver otherwise commits/fsyncs
per statement even within one ``executemany()`` call -- see ``mbariml.db``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import typer

from mbariml import db
from mbariml.image_quality import compute_sharpness
from mbariml.images import collect_images
from mbariml.logging_utils import get_logger
from mbariml.yolo_utils import load_model, resolve_device

app = typer.Typer(help="Run a trained YOLO model over a new batch of images.")
logger = get_logger(__name__)


def _results_to_rows(results, batch_files, model, roi_index_start: int) -> tuple[list[tuple], int]:
    """Build curation-schema rows for a batch, cropping each ROI from
    result.orig_img (already loaded by Ultralytics -- no extra file reads).
    Returns (rows, next_free_roi_index)."""
    rows = []
    roi_index = roi_index_start

    for result, image_file in zip(results, batch_files):
        image = result.orig_img  # BGR numpy array
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[class_id]

            roi = image[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                logger.warning("Empty ROI for %s at (%d,%d,%d,%d); skipping", image_file.name, x_min, y_min, x_max, y_max)
                continue
            ok, roi_encoded = cv2.imencode(".jpg", roi)
            if not ok:
                logger.warning("Failed to encode ROI for %s; skipping", image_file.name)
                continue

            rows.append((
                roi_index, image_file.name, str(image_file), roi_index,
                float(x_min), float(y_min), float(x_max), float(y_max),
                class_id, confidence, label,
                None,  # embedding, filled in by `mbariml embed`
                None,  # new_label, filled in by clustering/review
                roi_encoded.tobytes(),
                compute_sharpness(roi),
                0,  # verified, set by the review GUI
            ))
            roi_index += 1

    return rows, roi_index


def _insert_rows(conn, rows: list[tuple]) -> None:
    if not rows:
        return
    # Explicit column list (not a bare "VALUES (?, ...)") so a column added
    # to CURATION_SCHEMA_SQL after the fact -- e.g. `verified`, via ALTER
    # TABLE, see mbariml.db -- doesn't silently break this insert or require
    # staying in lockstep by position; DuckDB rejects a positional VALUES
    # list whose length doesn't match the table's full column count,
    # DEFAULT or not.
    db.fast_executemany(
        conn,
        """INSERT INTO predictions
           (id, image_name, image_path, roi_index, x_min, y_min, x_max, y_max,
            class_id, confidence, label, embedding, new_label, roi, sharpness, verified)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


def _run_batches(model, image_files, batch_size: int, conn, yolo_params: dict) -> tuple[int, int]:
    """Process every batch, committing rows to the database as soon as each
    batch finishes. Returns (rows_inserted, batches_failed)."""
    total_rows = 0
    failed_batches = 0
    roi_index = 0
    num_batches = (len(image_files) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(image_files), batch_size), start=1):
        batch_files = image_files[i : i + batch_size]
        try:
            results = model.predict(source=[str(f) for f in batch_files], **yolo_params)
            rows, roi_index = _results_to_rows(results, batch_files, model, roi_index)
            _insert_rows(conn, rows)
            total_rows += len(rows)
            logger.info(
                "Batch %d/%d: %d images -> %d detections (running total: %d rows)",
                batch_num, num_batches, len(batch_files), len(rows), total_rows,
            )
        except Exception:
            failed_batches += 1
            logger.exception("Batch %d/%d failed (%d images); continuing with remaining batches",
                              batch_num, num_batches, len(batch_files))

    return total_rows, failed_batches


def _export_csv(conn, csv_path: Path) -> None:
    distinct_images = [row[0] for row in conn.execute("SELECT DISTINCT image_name FROM predictions").fetchall()]
    if not distinct_images:
        logger.warning("No predictions in the database; skipping CSV export (nothing to write to %s).", csv_path)
        return

    sanitized_columns = [
        f"MAX(CASE WHEN image_name = '{name}' THEN count ELSE 0 END) AS \"{name.replace(chr(34), chr(34)*2)}\""
        for name in distinct_images
    ]
    conn.execute(
        f"""
        COPY (
            SELECT
                label AS taxa,
                {', '.join(sanitized_columns)}
            FROM (
                SELECT label, image_name, COUNT(*) AS count
                FROM predictions
                GROUP BY label, image_name
            ) subquery
            GROUP BY label
            ORDER BY label ASC
        ) TO '{csv_path}' WITH (HEADER, DELIMITER ',')
        """
    )
    logger.info("Results exported to %s", csv_path)


@app.command()
def infer(
    model_path: str = typer.Argument(..., help="Path to a YOLO .pt model, or a stock Ultralytics model name."),
    input_dir: str = typer.Argument(..., help="Directory containing input images (searched recursively)."),
    output_dir: str = typer.Argument(..., help="Directory to save the database, annotated images, and (optionally) a CSV."),
    export_csv: bool = typer.Option(False, help="Also export a taxa-by-image count matrix to predictions.csv."),
    limit: Optional[int] = typer.Option(None, help="Limit the number of images processed (for a quick test run)."),
    batch_size: int = typer.Option(50, help="Number of images sent to YOLO per batch."),
    conf: float = typer.Option(0.08, help="Confidence threshold."),
    iou: float = typer.Option(0.15, help="IoU threshold for NMS."),
    max_det: int = typer.Option(500, help="Maximum detections per image."),
    imgsz: int = typer.Option(992, help="Inference image size."),
    device: str = typer.Option("auto", help="Device to run on: 'auto', 'mps', 'cuda', or 'cpu'."),
) -> None:
    """Run YOLO inference over every image under INPUT_DIR and store detections
    (with ROI crops, ready for review/clustering/export) in OUTPUT_DIR/yolo_predictions.duckdb."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    logger.info("Using device: %s", resolved_device)

    image_files = collect_images(input_dir, limit)
    logger.info("Found %d image(s) under %s", len(image_files), input_dir)

    model = load_model(model_path)
    (output_dir_path / "names.txt").write_text("\n".join(model.names.values()))

    yolo_params = {
        "line_width": 1,
        "agnostic_nms": True,
        "save": True,
        "project": str(output_dir_path),
        "name": "predictions",
        "exist_ok": True,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "device": resolved_device,
        "imgsz": imgsz,
        "augment": True,
        "quantize": 16 if resolved_device != "cpu" else None,  # FP16; Ultralytics deprecated half= in favor of this
    }

    db_path = output_dir_path / "yolo_predictions.duckdb"
    with db.init_curation_db(db_path) as conn:
        total_rows, failed_batches = _run_batches(model, image_files, batch_size, conn, yolo_params)

        if export_csv:
            _export_csv(conn, output_dir_path / "predictions.csv")

        final_count = db.row_count(conn)

        conn.execute("DELETE FROM run_info")
        conn.execute("INSERT INTO run_info VALUES (?, CURRENT_TIMESTAMP)", (model_path,))

    logger.info(
        "Done: %d image(s) processed, %d row(s) written to %s, annotated images in %s",
        len(image_files), final_count, db_path, output_dir_path / "predictions",
    )
    if final_count:
        logger.info(
            "Ready to continue with: mbariml embed %s | mbariml review %s | mbariml export html %s <out_dir>",
            db_path, db_path, db_path,
        )

    if failed_batches:
        logger.error("%d batch(es) failed -- see the tracebacks above for details.", failed_batches)
        raise typer.Exit(code=1)

    if final_count == 0:
        logger.warning(
            "No detections were made on any image. This can be legitimate (nothing above the "
            "confidence threshold), but double-check --model-path and --conf if you expected results."
        )


if __name__ == "__main__":
    app()
