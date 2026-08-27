"""Backfill sharpness scores for ROIs already in a curation database.

Databases created before ``mbariml detect`` started actually computing
sharpness have every row hardcoded to ``0.0``. This recomputes a real score
from each row's already-stored ROI blob -- no need to re-run detection or
touch the original survey images. Not part of the numbered step chain; run
it once against any existing database before using the "sort by sharpness"
option in the review GUI (step 5).
"""

from __future__ import annotations

import cv2
import numpy as np
import typer
from tqdm import tqdm

from mbariml import db
from mbariml.image_quality import compute_sharpness
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Backfill sharpness (blur) scores for ROIs already in a curation database.")
logger = get_logger(__name__)


@app.command()
def backfill_sharpness(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    force: bool = typer.Option(False, help="Recompute even for rows that already have a non-zero sharpness value."),
) -> None:
    """Compute and store a sharpness (blur) score for every ROI, from its stored crop."""
    with db.connect(db_path) as conn:
        query = "SELECT id, roi FROM predictions" if force else "SELECT id, roi FROM predictions WHERE sharpness IS NULL OR sharpness = 0.0"
        rows = conn.execute(query).fetchall()
        if not rows:
            logger.info("Nothing to backfill; every row already has a sharpness score (use --force to recompute anyway).")
            return

        logger.info("Computing sharpness for %d ROI(s)...", len(rows))
        updated, skipped = 0, 0
        for roi_id, roi_blob in tqdm(rows, desc="Computing sharpness"):
            if roi_blob is None:
                skipped += 1
                continue
            roi = cv2.imdecode(np.frombuffer(roi_blob, np.uint8), cv2.IMREAD_COLOR)
            if roi is None:
                logger.warning("Could not decode ROI blob for id=%s; skipping", roi_id)
                skipped += 1
                continue
            conn.execute("UPDATE predictions SET sharpness = ? WHERE id = ?", (compute_sharpness(roi), roi_id))
            updated += 1

    logger.info("Done: %d row(s) updated, %d skipped.", updated, skipped)


if __name__ == "__main__":
    app()
