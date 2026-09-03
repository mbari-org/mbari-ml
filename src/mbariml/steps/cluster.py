"""Enrich: cluster embeddings with EVoC and generate ROI review grids.

Bug fixed here: the original ``--limit`` option ran
``DELETE FROM predictions WHERE rowid NOT IN (SELECT rowid FROM predictions LIMIT N)``
-- i.e. passing ``--limit`` for a quick test run *permanently deleted* every
row beyond the first N from the database. ``--limit`` now only limits how
many rows are read for clustering; it no longer touches the table.

Bug fixed here (this was why a real run on 300k+ embeddings took ~4 hours
with no visible progress): the actual clustering call (``evoc.fit_predict``)
was never the bottleneck -- measured directly, it clusters 50,000 embeddings
in ~2.6 seconds and scales roughly linearly, so 300k+ points finishes in
well under a minute. It isn't doing brute-force KNN either; it has its own
JIT-compiled approximate nearest-neighbor search internally, so swapping in
DuckDB's VSS/HNSW extension wouldn't touch this cost at all -- there's no
repeated similarity-search query happening here for an ANN index to help
with, just a one-shot batch clustering call that's already fast. The real
cost was writing results back with one ``UPDATE ... WHERE id = ?`` per row:
DuckDB's MVCC row-versioning overhead makes many small UPDATEs dramatically
slower than one bulk UPDATE, and it's worse here than in `embed` because this
UPDATE touches the *indexed* ``new_label`` column -- measured at 100,000
rows: 38.6 seconds via the old per-row pattern (still trending worse) vs.
~6 seconds via a staged bulk UPDATE (``mbariml.db.bulk_update``). Progress is
now also logged at each phase, so a long run is never silent about where it
actually is.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import typer
from tqdm import tqdm

from mbariml import db
from mbariml.logging_utils import get_logger

warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")

app = typer.Typer(help="Cluster ROI embeddings with EVoC and export review grids.")
logger = get_logger(__name__)

GRID_WIDTH, GRID_HEIGHT, GRID_PADDING = 2550, 3300, 5

# EVoC's internal graph/PCA construction needs a minimum number of samples
# regardless of n_neighbors (its embedding-dimension step wants more samples
# than dimensions); below this it fails with an opaque sklearn PCA
# ValueError rather than a useful message. This is a floor, not a precise
# derivation of EVoC's actual internal requirement -- just enough to turn a
# cryptic crash into an actionable warning for a too-small input.
MIN_ROWS_TO_CLUSTER = 10


def _cluster_embeddings(
    conn,
    limit: Optional[int],
    *,
    approx_n_clusters: Optional[int],
    base_min_cluster_size: int,
    noise_level: float,
    n_neighbors: int,
    min_samples: int,
    seed: Optional[int],
) -> int:
    """Cluster embeddings using EVoC and write evoc_clust/new_label back. Returns cluster count."""
    import evoc

    db.ensure_column(conn, "predictions", "evoc_clust", "INTEGER")

    logger.info("Fetching embedded rows...")
    query = "SELECT id, embedding, label FROM predictions WHERE embedding IS NOT NULL"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()

    if not rows:
        logger.warning("No embedded rows found; skipping clustering.")
        return 0
    min_required = max(MIN_ROWS_TO_CLUSTER, n_neighbors + 1)
    if len(rows) < min_required:
        # Most likely to happen with a small --limit (e.g. a quick test run).
        logger.warning(
            "Only %d embedded row(s) -- too few to cluster (need at least %d with "
            "--n-neighbors %d). Lower --n-neighbors or raise/drop --limit.",
            len(rows), min_required, n_neighbors,
        )
        return 0

    ids = [row[0] for row in rows]
    labels = [row[2] for row in rows]
    logger.info("Building embedding matrix for %d row(s)...", len(rows))
    embeddings = np.array([row[1] for row in rows])

    logger.info(
        "Clustering %d embedding(s) with EVoC (approx_n_clusters=%s, base_min_cluster_size=%d, "
        "noise_level=%.2f, n_neighbors=%d, min_samples=%d, seed=%s)...",
        len(rows), approx_n_clusters, base_min_cluster_size, noise_level, n_neighbors, min_samples, seed,
    )
    clusterer = evoc.EVoC(
        approx_n_clusters=approx_n_clusters,
        base_min_cluster_size=base_min_cluster_size,
        noise_level=noise_level,
        n_neighbors=n_neighbors,
        min_samples=min_samples,
        random_state=seed,
    )
    cluster_labels = [int(c) for c in clusterer.fit_predict(embeddings)]

    cluster_to_labels: dict[int, list[str]] = {}
    for label, cluster_label in zip(labels, cluster_labels):
        if cluster_label == -1:
            continue
        cluster_to_labels.setdefault(cluster_label, []).append(label)
    cluster_to_dominant_label = {c: max(set(ls), key=ls.count) for c, ls in cluster_to_labels.items()}

    logger.info("Writing cluster assignments back to the database...")
    db.bulk_update(
        conn, "predictions", "id", "INTEGER",
        {"evoc_clust": "INTEGER", "new_label": "TEXT"},
        [
            (row_id, c, cluster_to_dominant_label[c] if c != -1 else "noise")
            for row_id, c in zip(ids, cluster_labels)
        ],
    )
    logger.info("Clustered %d embedding(s) into %d cluster(s).", len(rows), len(cluster_to_dominant_label))
    return len(cluster_to_dominant_label)


def _generate_roi_grids(conn, output_dir: Path, group_by_dominant_label: bool) -> None:
    grid_dir = output_dir / "roi_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    if group_by_dominant_label:
        rows = conn.execute(
            "SELECT roi, id, new_label AS label FROM predictions WHERE new_label != 'noise'"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT roi, id, evoc_clust AS label FROM predictions WHERE evoc_clust != -1"
        ).fetchall()

    cluster_groups: dict = {}
    for roi_blob, roi_id, label in rows:
        cluster_groups.setdefault(label, []).append((roi_blob, roi_id))

    for cluster_name, rois in tqdm(cluster_groups.items(), desc="Generating ROI grids"):
        num_rois = len(rois)
        n_rows = int(np.ceil(np.sqrt(num_rois)))
        n_cols = int(np.ceil(num_rois / n_rows))
        image_size = int(min(
            (GRID_WIDTH - GRID_PADDING * (n_cols - 1)) / n_cols,
            (GRID_HEIGHT - GRID_PADDING * (n_rows - 1)) / n_rows,
        ))

        page_number = 1
        remaining = rois
        while remaining:
            grid_image = np.full((GRID_HEIGHT, GRID_WIDTH, 3), 255, dtype=np.uint8)
            x, y = 0, 0
            leftover = []

            for roi_blob, roi_id in remaining:
                roi_image = cv2.imdecode(np.frombuffer(roi_blob, np.uint8), cv2.IMREAD_UNCHANGED)
                if roi_image is None:
                    logger.warning("Could not decode ROI blob for id=%s; skipping", roi_id)
                    continue
                roi_image = cv2.resize(roi_image, (image_size, image_size))

                if y + image_size > grid_image.shape[0]:
                    leftover.append((roi_blob, roi_id))
                    continue

                grid_image[y:y + image_size, x:x + image_size] = roi_image
                x += image_size + GRID_PADDING
                if x + image_size > grid_image.shape[1]:
                    x, y = 0, y + image_size + GRID_PADDING

            output_path = grid_dir / f"{cluster_name}_page_{page_number}.jpg"
            cv2.imwrite(str(output_path), grid_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            page_number += 1
            remaining = leftover

    logger.info("ROI grids written to %s", grid_dir)


@app.command()
def cluster(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    limit: Optional[int] = typer.Option(None, help="Only read this many embedded rows for clustering (does not delete data)."),
    off: bool = typer.Option(False, help="Disable grouping ROI grids by dominant cluster label (use raw evoc_clust instead)."),
    approx_n_clusters: Optional[int] = typer.Option(
        18,
        help=(
            "Target number of clusters EVoC aims for. If your data actually has more "
            "distinct visual groups than this, EVoC will tend to merge some of them "
            "together or push borderline ones into noise -- raise this (or pass "
            "--approx-n-clusters=0 to let EVoC choose the count itself) if clearly "
            "distinct things keep landing in 'noise'."
        ),
    ),
    base_min_cluster_size: int = typer.Option(
        2, help="Minimum points to count as a cluster at all (EVoC's base_min_cluster_size). Smaller allows more, smaller clusters."
    ),
    noise_level: float = typer.Option(
        0.2,
        help=(
            "0.0-1.0: EVoC's primary noise/outlier-sensitivity knob. Lower tries to "
            "cluster more of the data (fewer points called noise, at some cost to "
            "cluster accuracy); higher is stricter about what counts as a real "
            "cluster (more noise, more accurate clusters)."
        ),
    ),
    n_neighbors: int = typer.Option(
        40, help="Nearest-neighbor graph size EVoC builds internally. Higher smooths over finer local structure; lower preserves it (more, smaller clusters)."
    ),
    min_samples: int = typer.Option(
        5, help="EVoC's density-estimation parameter (like HDBSCAN's min_samples). Higher makes noise classification more conservative (more points called noise)."
    ),
    seed: Optional[int] = typer.Option(
        None, help="Random seed for EVoC (its random_state). Unset means a different, non-reproducible result every run -- set this while iterating on the other options above so changes are comparable."
    ),
) -> None:
    """Cluster ROI embeddings with EVoC and export review grids."""
    output_dir = Path(db_path).parent

    with db.connect(db_path) as conn:
        n_clusters = _cluster_embeddings(
            conn,
            limit,
            approx_n_clusters=approx_n_clusters or None,
            base_min_cluster_size=base_min_cluster_size,
            noise_level=noise_level,
            n_neighbors=n_neighbors,
            min_samples=min_samples,
            seed=seed,
        )
        if n_clusters:
            _generate_roi_grids(conn, output_dir, group_by_dominant_label=not off)


if __name__ == "__main__":
    app()
