"""Enrich: re-cluster a single label's ROIs into finer-grained sub-clusters.

Bug fixed here: the original script needed the *original* full-frame images
to build its review grids (it re-read and re-cropped each ROI from disk via
``image_dir / image_name``), but its CLI never actually exposed an
``image_dir`` argument -- it hardcoded ``Path(".")``, so it only produced
correct grids if you happened to run the command from inside the image
directory, and otherwise silently skipped every ROI (each ``image_path``
lookup missed and was quietly skipped). Since ingest already stores every
ROI as a JPEG blob in the database, this version decodes that stored blob
directly -- matching what `cluster` does -- instead of re-reading source
images at all, which removes the missing-argument bug entirely.

Also fixed here, same root cause as `cluster`'s slow-clustering bug: writing
each sub-cluster's new_label with its own executemany UPDATE, against the
indexed new_label column, which DuckDB is dramatically slower at than one
bulk UPDATE. Now written with a single ``mbariml.db.bulk_update`` call
covering every sub-cluster at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import typer
from tqdm import tqdm

from mbariml import db
from mbariml.logging_utils import get_logger
from mbariml.steps import cluster as cluster_step

app = typer.Typer(help="Re-cluster a single label's ROIs into finer-grained sub-clusters.")
logger = get_logger(__name__)

GRID_WIDTH, GRID_HEIGHT, GRID_PADDING = 2550, 3300, 5


def _fetch_rows(conn, label: Optional[str], limit: Optional[int]) -> list[tuple]:
    query = "SELECT id, embedding, roi FROM predictions WHERE embedding IS NOT NULL"
    params: list = []
    if label:
        query += " AND new_label = ?"
        params.append(label)
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    return conn.execute(query, params).fetchall()


def _cluster(
    rows: list[tuple],
    *,
    approx_n_clusters: Optional[int],
    base_min_cluster_size: int,
    noise_level: float,
    n_neighbors: int,
    min_samples: int,
    seed: Optional[int],
) -> dict[int, list[tuple]]:
    import evoc

    embeddings = np.array([row[1] for row in rows])
    clusterer = evoc.EVoC(
        approx_n_clusters=approx_n_clusters,
        base_min_cluster_size=base_min_cluster_size,
        noise_level=noise_level,
        n_neighbors=n_neighbors,
        min_samples=min_samples,
        random_state=seed,
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    clusters: dict[int, list[tuple]] = {}
    for row, cluster_label in zip(rows, cluster_labels):
        clusters.setdefault(int(cluster_label), []).append(row)
    return clusters


def _label_for_cluster(cluster_id: int, new_label: str) -> str:
    """The new_label value for one sub-cluster.

    cluster_id == -1 is EVoC's "still not clustered" bucket, mapped to the
    literal string "noise" -- matching `cluster`'s convention. Every export
    step (export voc, export yolo, export id, ...) filters on ``new_label != 'noise'``,
    so a still-noise point must keep exactly that label; an earlier revision
    of this function instead produced e.g. "noise_1" for it, which silently
    escaped that filter and would have counted as a curated identification.
    """
    return "noise" if cluster_id == -1 else f"{new_label}_{cluster_id + 1}"


def _update_db(conn, clusters: dict[int, list[tuple]], new_label: str) -> None:
    """Write every cluster's new_label in one bulk UPDATE. Was previously one
    executemany UPDATE per cluster (each touching the indexed new_label
    column) -- DuckDB is dramatically slower at many small row-by-row
    UPDATEs than at one bulk UPDATE; see mbariml.db.bulk_update."""
    updates = [
        (row[0], _label_for_cluster(cluster_id, new_label))
        for cluster_id, rows in clusters.items()
        for row in rows
    ]
    db.bulk_update(conn, "predictions", "id", "INTEGER", {"new_label": "TEXT"}, updates)


def _generate_roi_grids(clusters: dict[int, list[tuple]], output_dir: Path, new_label: str) -> None:
    grid_dir = output_dir / "filtered_roi_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id, rows in tqdm(clusters.items(), desc="Generating ROI grids"):
        if cluster_id == -1:
            continue  # matches `cluster`: no review grid for the noise bucket
        cluster_label = _label_for_cluster(cluster_id, new_label)
        num_rois = len(rows)
        n_rows = int(np.ceil(np.sqrt(num_rois)))
        n_cols = int(np.ceil(num_rois / n_rows))
        image_size = int(min(
            (GRID_WIDTH - GRID_PADDING * (n_cols - 1)) / n_cols,
            (GRID_HEIGHT - GRID_PADDING * (n_rows - 1)) / n_rows,
        ))

        grid_image = np.full((GRID_HEIGHT, GRID_WIDTH, 3), 255, dtype=np.uint8)
        x, y = 0, 0

        for row_id, _embedding, roi_blob in rows:
            roi_image = cv2.imdecode(np.frombuffer(roi_blob, np.uint8), cv2.IMREAD_UNCHANGED)
            if roi_image is None:
                logger.warning("Could not decode ROI blob for id=%s; skipping", row_id)
                continue
            roi_image = cv2.resize(roi_image, (image_size, image_size))

            if y + image_size > GRID_HEIGHT:
                break
            grid_image[y:y + image_size, x:x + image_size] = roi_image
            x += image_size + GRID_PADDING
            if x + image_size > GRID_WIDTH:
                x, y = 0, y + image_size + GRID_PADDING

        cv2.imwrite(str(grid_dir / f"{cluster_label}_filtered.jpg"), grid_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    logger.info("ROI grids written to %s", grid_dir)


@app.command()
def refine(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    output_dir: str = typer.Argument(..., help="Directory to write filtered_roi_grids/ into."),
    new_label: Optional[str] = typer.Option(None, help="Label to filter and re-cluster (e.g. 'Muusoctopus' or 'noise')."),
    limit: Optional[int] = typer.Option(None, help="Limit the number of rows processed, for testing."),
    approx_n_clusters: Optional[int] = typer.Option(
        None, help="Target number of sub-clusters EVoC aims for. Unset (default) lets EVoC choose based on the data."
    ),
    base_min_cluster_size: int = typer.Option(
        5, help="Minimum points to count as a sub-cluster at all (EVoC's base_min_cluster_size). Smaller allows more, smaller sub-clusters."
    ),
    noise_level: float = typer.Option(
        0.5,
        help=(
            "0.0-1.0: EVoC's primary noise/outlier-sensitivity knob. Lower tries to "
            "find structure in more of this subset (fewer points left as noise); "
            "higher is stricter (more left as noise). Worth lowering when refining "
            "a big 'noise' bucket specifically to dig out hidden distinct groups."
        ),
    ),
    n_neighbors: int = typer.Option(
        15, help="Nearest-neighbor graph size EVoC builds internally. Higher smooths over finer local structure; lower preserves it."
    ),
    min_samples: int = typer.Option(
        5, help="EVoC's density-estimation parameter (like HDBSCAN's min_samples). Higher makes noise classification more conservative."
    ),
    seed: Optional[int] = typer.Option(
        None, help="Random seed for EVoC (its random_state). Set this while iterating on the other options above so changes are comparable."
    ),
) -> None:
    """Cluster embeddings for a specific label into finer sub-clusters and generate ROI grids."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with db.connect(db_path, must_exist=True) as conn:
        logger.info("Fetching embeddings for label: %s", new_label or "all labels")
        rows = _fetch_rows(conn, new_label, limit)
        if not rows:
            logger.warning("No embeddings found for the specified label; nothing to do.")
            return
        min_required = max(cluster_step.MIN_ROWS_TO_CLUSTER, n_neighbors + 1)
        if len(rows) < min_required:
            # A small residual bucket (e.g. refining an already-mostly-resolved
            # "noise" label) is a realistic way to hit this -- see
            # MIN_ROWS_TO_CLUSTER's docstring in cluster.py.
            logger.warning(
                "Only %d row(s) for label %r -- too few to cluster (need at least %d with "
                "--n-neighbors %d). Lower --n-neighbors, pick a larger label, or skip refining this one.",
                len(rows), new_label, min_required, n_neighbors,
            )
            return

        logger.info("Clustering %d embedding(s)...", len(rows))
        clusters = _cluster(
            rows,
            approx_n_clusters=approx_n_clusters,
            base_min_cluster_size=base_min_cluster_size,
            noise_level=noise_level,
            n_neighbors=n_neighbors,
            min_samples=min_samples,
            seed=seed,
        )
        if not clusters:
            logger.warning("Clustering produced no clusters; nothing to do.")
            return

        label_for_naming = new_label or "cluster"
        n_real_clusters = sum(1 for cluster_id in clusters if cluster_id != -1)
        logger.info("Writing %d sub-cluster label(s) back to the database...", len(clusters))
        _update_db(conn, clusters, label_for_naming)
        _generate_roi_grids(clusters, output_path, label_for_naming)

    logger.info(
        "Done: %d sub-cluster(s) written to %s%s",
        n_real_clusters,
        output_path / "filtered_roi_grids",
        f" ({len(clusters[-1])} row(s) left as 'noise')" if -1 in clusters else "",
    )


if __name__ == "__main__":
    app()
