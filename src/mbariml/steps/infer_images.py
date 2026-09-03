"""``mbariml infer images``: run a YOLO model over a directory of images,
crop every detection as an ROI, and store the results in a curation database.

This is the merge of what used to be two nearly-identical commands: `detect`
(step 1: seed a curation database) and `infer-images` (step 8: run a trained
model over new imagery). They wrote the *same schema*, cropped ROIs the same
way, and recorded the same run_info -- the entire real difference was
batching, whether annotated images were saved, and a 16x gap in the default
confidence threshold. None of that is architecture; it's flags and defaults.
Keeping two copies meant every fix had to be made twice (and, in practice,
sometimes wasn't). The intent distinction survives as ``--preset``:

    --preset curate   conf 0.005, imgsz 1952, no annotated images
                      Mine everything, then cluster/review and throw away the
                      noise. This is the default, deliberately: an
                      over-permissive threshold is recoverable (filter later
                      -- the review GUI even has a min-confidence slider),
                      while a too-strict one silently drops detections you
                      can only get back with a full re-run.
    --preset predict  conf 0.08, imgsz 992, saves annotated images
                      Believable predictions over new imagery.

Any individual option still wins over the preset -- pass ``--conf`` and it's
used regardless of which preset supplied the rest.

ROIs are cropped directly from ``result.orig_img`` (the array Ultralytics
already decoded for that prediction), not by re-reading each file from disk a
second time.

Bug fixed here, inherited from the `infer-images` side of the merge (this was
the root cause of "code seemed to run, but no results saved, no db
generated"): the original accumulated *every* detection row from *every*
batch in memory and called ``conn.executemany(...)`` once, after the whole
image set had been processed, outside any try/except. YOLO would visibly run
and even save annotated images, but if anything went wrong in that single
final insert -- or the process was interrupted before reaching it -- nothing
ever reached the database. Rows are now inserted, and the connection flushed
via ``mbariml.db.connect``'s context manager, after *every* batch. Each
batch's insert goes through ``db.fast_executemany`` (an explicit transaction)
rather than a bare ``conn.executemany``, since DuckDB's Python driver
otherwise commits/fsyncs per statement even inside one ``executemany()`` call
-- see ``mbariml.db``.
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

app = typer.Typer(help="Run a YOLO model over a directory of images.")
logger = get_logger(__name__)

# (conf, iou, imgsz, save_annotated) per preset -- see the module docstring
# for what each one is for and why `curate` is the default.
PRESETS: dict[str, dict] = {
    "curate": {"conf": 0.005, "iou": 0.05, "imgsz": 1952, "save_annotated": False},
    "predict": {"conf": 0.08, "iou": 0.15, "imgsz": 992, "save_annotated": True},
}
DEFAULT_PRESET = "curate"


def _resolve_preset(preset: str, **overrides) -> dict:
    """Preset defaults, with any explicitly-passed option winning.

    Every overridable option is declared ``Optional[...] = None`` on the
    command precisely so this can tell "user didn't pass it" (None -> take the
    preset's value) from "user passed exactly the preset's value" -- which a
    plain typer default could not.
    """
    if preset not in PRESETS:
        raise typer.BadParameter(f"--preset must be one of {sorted(PRESETS)}")
    resolved = dict(PRESETS[preset])
    resolved.update({key: value for key, value in overrides.items() if value is not None})
    return resolved


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


def insert_rows(conn, rows: list[tuple]) -> None:
    """Insert curation-schema rows for image-derived detections.

    Explicit column list (not a bare "VALUES (?, ...)") so a column added to
    CURATION_SCHEMA_SQL after the fact -- e.g. `verified`, or the video
    provenance columns, both via ALTER TABLE, see mbariml.db -- doesn't
    silently break this insert or require staying in lockstep by position;
    DuckDB rejects a positional VALUES list whose length doesn't match the
    table's full column count, DEFAULT or not. The video columns are simply
    left unset here, which is what NULL-for-image-rows means.
    """
    if not rows:
        return
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
    # Continue the database's own id counter rather than restarting at 0 --
    # see db.next_free_id: starting at 0 made a second run into an existing
    # output directory collide on the UNIQUE id index instead of appending.
    roi_index = db.next_free_id(conn)
    num_batches = (len(image_files) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(image_files), batch_size), start=1):
        batch_files = image_files[i : i + batch_size]
        try:
            results = model.predict(source=[str(f) for f in batch_files], **yolo_params)
            rows, roi_index = _results_to_rows(results, batch_files, model, roi_index)
            insert_rows(conn, rows)
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


def export_taxa_matrix(conn, csv_path: Path) -> None:
    """Write a taxa-by-image count matrix (one column per image).

    Bug fixed here: image_name is interpolated straight into SQL (DuckDB can't
    parameterize a dynamic column list), so both the string literal AND the
    quoted-identifier alias need their own escaping -- only the alias was
    escaped before, so any image filename containing a single quote broke this
    query with a syntax error.
    """
    distinct_images = [row[0] for row in conn.execute("SELECT DISTINCT image_name FROM predictions").fetchall()]
    if not distinct_images:
        logger.warning("No predictions in the database; skipping CSV export (nothing to write to %s).", csv_path)
        return

    sanitized_columns = [
        f"MAX(CASE WHEN image_name = '{name.replace(chr(39), chr(39) * 2)}' THEN count ELSE 0 END) "
        f"AS \"{name.replace(chr(34), chr(34) * 2)}\""
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
def infer_images(
    model_path: str = typer.Argument(..., help="Path to a YOLO .pt model, or a stock Ultralytics model name."),
    input_dir: str = typer.Argument(..., help="Directory containing input images (searched recursively)."),
    output_dir: str = typer.Argument(..., help="Directory to save the database, annotated images, and (optionally) a CSV."),
    preset: str = typer.Option(
        DEFAULT_PRESET,
        help="'curate' (conf 0.005, imgsz 1952, no annotated images -- mine everything for "
        "clustering/review) or 'predict' (conf 0.08, imgsz 992, saves annotated images -- "
        "believable predictions over new imagery). Any option below overrides its preset value.",
    ),
    limit: Optional[int] = typer.Option(None, help="Limit the number of images processed (for a quick test run)."),
    random_sample: bool = typer.Option(
        False,
        "--random/--no-random",
        help="With --limit, take a seeded random sample spread across input_dir instead of the "
        "first N in sorted order. Useful for right-sizing a real run (e.g. --limit 1000 out of "
        "9000+ images), not just a quick test -- sorted-order limiting on a chronologically-named "
        "survey is a contiguous time slice, not a representative sample.",
    ),
    seed: Optional[int] = typer.Option(
        None, help="Random seed for --random. Same directory + --limit + --seed always picks the same images."
    ),
    batch_size: int = typer.Option(50, help="Number of images sent to YOLO per batch."),
    conf: Optional[float] = typer.Option(None, help="Confidence threshold. [default: from --preset]"),
    iou: Optional[float] = typer.Option(None, help="IoU threshold for NMS. [default: from --preset]"),
    max_det: int = typer.Option(500, help="Maximum detections per image."),
    imgsz: Optional[int] = typer.Option(None, help="Inference image size. [default: from --preset]"),
    device: str = typer.Option("auto", help="Device to run on: 'auto', 'mps', 'cuda', or 'cpu'."),
    save_annotated: Optional[bool] = typer.Option(
        None, "--save-annotated/--no-save-annotated",
        help="Also write annotated copies of each image. [default: from --preset]",
    ),
    export_csv: bool = typer.Option(False, help="Also export a taxa-by-image count matrix to predictions.csv."),
) -> None:
    """Run YOLO over every image under INPUT_DIR and store detections (with ROI
    crops, ready for review/clustering/export) in OUTPUT_DIR/yolo_predictions.duckdb."""
    settings = _resolve_preset(
        preset, conf=conf, iou=iou, imgsz=imgsz, save_annotated=save_annotated
    )
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    logger.info(
        "Preset '%s': conf=%.4g, iou=%.4g, imgsz=%d, save_annotated=%s | device: %s",
        preset, settings["conf"], settings["iou"], settings["imgsz"], settings["save_annotated"], resolved_device,
    )

    # Resolved to absolute: image_path is how `review` and every export find
    # the source image again, from whatever directory they're run in. A
    # relative input_dir used to be recorded verbatim, so those later commands
    # quietly reported images "not found on disk" whenever they ran from
    # anywhere but the original working directory.
    image_files = [f.resolve() for f in collect_images(input_dir, limit, random_sample=random_sample, seed=seed)]
    logger.info("Found %d image(s) under %s", len(image_files), input_dir)

    model = load_model(model_path)
    (output_dir_path / "names.txt").write_text("\n".join(model.names.values()))

    yolo_params = {
        "agnostic_nms": True,
        "augment": True,
        "conf": settings["conf"],
        "iou": settings["iou"],
        "imgsz": settings["imgsz"],
        "max_det": max_det,
        "device": resolved_device,
        "quantize": 16 if resolved_device != "cpu" else None,  # FP16; Ultralytics deprecated half= in favor of this
        "save": settings["save_annotated"],
    }
    if settings["save_annotated"]:
        yolo_params.update({"line_width": 1, "project": str(output_dir_path), "name": "predictions", "exist_ok": True})

    db_path = output_dir_path / "yolo_predictions.duckdb"
    with db.init_curation_db(db_path) as conn:
        total_rows, failed_batches = _run_batches(model, image_files, batch_size, conn, yolo_params)

        if export_csv:
            export_taxa_matrix(conn, output_dir_path / "predictions.csv")

        final_count = db.row_count(conn)

        conn.execute("DELETE FROM run_info")
        conn.execute("INSERT INTO run_info VALUES (?, CURRENT_TIMESTAMP)", (model_path,))

    logger.info("Done: %d image(s) processed, %d row(s) written to %s", len(image_files), final_count, db_path)
    if settings["save_annotated"]:
        logger.info("Annotated images in %s", output_dir_path / "predictions")
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
            "confidence threshold), but double-check the model path and --conf if you expected results."
        )


if __name__ == "__main__":
    app()
