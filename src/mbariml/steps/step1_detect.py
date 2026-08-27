"""Step 1: detect objects with YOLO, extract ROI crops, and seed the curation database.

This is the first link in the curation chain (1 -> 2 embed -> 3 cluster ->
4 refine -> 5 review GUI -> 6/7 export -> 8 query -> 10 remap). Its database
uses the richer ``CURATION`` schema (see ``mbariml.db``), which stores each
ROI as a JPEG blob so later steps can embed/cluster/review it without
re-reading the original imagery.

Each image's detections are inserted via ``db.fast_executemany`` rather than
a bare ``conn.executemany`` -- DuckDB's Python driver commits (and fsyncs to
disk) after every individual statement by default when writing to a
file-backed database, even within one ``executemany()`` call, which is
dramatically slower than wrapping the same call in one explicit transaction
(measured: ~14x on identical hardware/storage). See ``mbariml.db`` for the
full writeup -- this was the root cause of several "much slower than it
should be" reports across this pipeline, not just here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import typer
from tqdm import tqdm

from mbariml import db
from mbariml.image_quality import compute_sharpness
from mbariml.images import collect_images
from mbariml.logging_utils import get_logger
from mbariml.yolo_utils import load_model, resolve_device

app = typer.Typer(help="Detect objects with YOLO, extract ROIs, and store results in a DuckDB database.")
logger = get_logger(__name__)


def _process_image(image_file: Path, model, conn, roi_index: int, predict_kwargs: dict) -> int:
    """Run YOLO on a single image and insert its detections. Returns the next
    free roi_index. Rows are inserted immediately (not buffered across the
    whole run) so progress survives an interruption partway through."""
    image = cv2.imread(str(image_file))
    if image is None:
        logger.warning("Failed to load image (skipping): %s", image_file)
        return roi_index

    results = model.predict(source=image, save=False, **predict_kwargs)

    rows = []
    for result in results:
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
                x_min, y_min, x_max, y_max,
                class_id, confidence, label,
                None,  # embedding, filled in by step 2
                None,  # new_label, filled in by clustering/review
                roi_encoded.tobytes(),
                compute_sharpness(roi),
                0,  # verified, set by the review GUI
            ))
            roi_index += 1

    if rows:
        # Explicit column list (not a bare "VALUES (?, ...)") so a column
        # added to CURATION_SCHEMA_SQL after the fact -- e.g. `verified`,
        # via ALTER TABLE, see mbariml.db -- doesn't silently break this
        # insert or require staying in lockstep by position; DuckDB rejects
        # a positional VALUES list whose length doesn't match the table's
        # full column count, DEFAULT or not.
        db.fast_executemany(
            conn,
            """INSERT INTO predictions
               (id, image_name, image_path, roi_index, x_min, y_min, x_max, y_max,
                class_id, confidence, label, embedding, new_label, roi, sharpness, verified)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    return roi_index


@app.command()
def detect(
    model_path: str = typer.Argument(..., help="Path to the YOLO model."),
    image_dir: str = typer.Argument(..., help="Directory containing input images."),
    output_dir: str = typer.Argument(..., help="Directory to save the database and class-name file."),
    limit: Optional[int] = typer.Option(None, help="Limit the number of images processed."),
    random_sample: bool = typer.Option(
        False,
        "--random/--no-random",
        help=(
            "With --limit, take a seeded random sample spread across image_dir "
            "instead of the first N in sorted order. Useful for right-sizing a "
            "real run (e.g. --limit 1000 out of 9000+ images), not just a quick "
            "test -- sorted-order limiting on a chronologically-named survey is "
            "a contiguous time slice, not a representative sample."
        ),
    ),
    seed: Optional[int] = typer.Option(
        None, help="Random seed for --random. Same directory + --limit + --seed always picks the same images."
    ),
    conf: float = typer.Option(0.005, help="Confidence threshold."),
    iou: float = typer.Option(0.05, help="IoU threshold."),
    max_det: int = typer.Option(500, help="Maximum detections per image."),
    imgsz: int = typer.Option(1952, help="Inference image size."),
    device: str = typer.Option("auto", help="Device to run on: 'auto', 'mps', 'cuda', or 'cpu'."),
) -> None:
    """Detect objects using YOLO, extract ROIs, and store results in a DuckDB database."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    logger.info("Using device: %s", resolved_device)

    image_files = collect_images(image_dir, limit, random_sample=random_sample, seed=seed)
    logger.info("Found %d image(s) under %s", len(image_files), image_dir)

    model = load_model(model_path)
    (output_dir_path / "names.txt").write_text("\n".join(model.names.values()))

    predict_kwargs = {
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "device": resolved_device,
        "imgsz": imgsz,
        "agnostic_nms": True,
        "quantize": 16 if resolved_device != "cpu" else None,  # FP16; Ultralytics deprecated half= in favor of this
        "augment": True,
    }

    db_path = output_dir_path / "yolo_predictions.duckdb"
    roi_index = 0
    failures = 0
    with db.init_curation_db(db_path) as conn:
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                roi_index = _process_image(image_file, model, conn, roi_index, predict_kwargs)
            except Exception:
                failures += 1
                logger.exception("Error processing %s; continuing with remaining images", image_file)

        final_count = db.row_count(conn)

        # Record which model produced this run, for provenance in later steps
        # (e.g. `mbariml export id`). Replaces any previous record -- this
        # reflects only the most recent `detect` run against this database.
        conn.execute("DELETE FROM run_info")
        conn.execute("INSERT INTO run_info VALUES (?, CURRENT_TIMESTAMP)", (model_path,))

    logger.info("Done: %d image(s) processed, %d detection row(s) written to %s", len(image_files), final_count, db_path)
    if failures:
        logger.error("%d image(s) failed to process -- see tracebacks above.", failures)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
