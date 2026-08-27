"""stats: label counts, boxes-per-image summary stats, and an optional
image x label count matrix CSV -- the raw numbers behind an ecological read
of a curation database (e.g. "how many Muusoctopus per image, and across how
many images").

Every query here uses ``COALESCE(new_label, label)`` as the effective label
-- the same per-row fallback ``html`` uses (see README's "What changed"),
so this is useful both before curation (nothing but the raw YOLO ``label``
set yet) and after (curated ``new_label`` set). Noise is included by default
(seeing how much of a database is still 'noise' is itself a useful number
while curating) -- pass --exclude-noise once you actually want ecological
counts of real identifications only.

Not part of the numbered step chain (like `backfill-sharpness`/`export id`)
-- run it directly, any time, read-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from mbariml import db
from mbariml.image_naming import disambiguated_stem
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Print label counts and per-image detection stats for a curation database.")
logger = get_logger(__name__)

_EFFECTIVE_LABEL_SQL = "COALESCE(new_label, label)"


def _where_clause(exclude_noise: bool) -> str:
    return f"WHERE {_EFFECTIVE_LABEL_SQL} != 'noise'" if exclude_noise else ""


def _label_counts(conn, exclude_noise: bool) -> pd.DataFrame:
    df = conn.execute(
        f"""
        SELECT {_EFFECTIVE_LABEL_SQL} AS label, COUNT(*) AS count
        FROM predictions
        {_where_clause(exclude_noise)}
        GROUP BY 1
        ORDER BY count DESC
        """
    ).df()
    total = df["count"].sum()
    df["percent"] = (df["count"] / total * 100).round(2) if total else df["count"]
    return df


def _boxes_per_image(conn, exclude_noise: bool) -> pd.DataFrame:
    return conn.execute(
        f"""
        WITH per_image AS (
            SELECT image_path, COUNT(*) AS boxes
            FROM predictions
            {_where_clause(exclude_noise)}
            GROUP BY image_path
        )
        SELECT
            COUNT(*) AS images,
            SUM(boxes) AS total_boxes,
            ROUND(AVG(boxes), 2) AS avg_per_image,
            MIN(boxes) AS min_per_image,
            MEDIAN(boxes) AS median_per_image,
            MAX(boxes) AS max_per_image
        FROM per_image
        """
    ).df()


def _write_matrix(conn, output_dir: Path, exclude_noise: bool) -> tuple[Path, int, int]:
    """Writes label_by_image_matrix.csv: rows=image (disambiguated stem,
    collision-safe across dives -- see mbariml.image_naming), columns=label,
    value=box count. Returns (path, n_images, n_labels)."""
    long_df = conn.execute(
        f"""
        SELECT image_path, {_EFFECTIVE_LABEL_SQL} AS label, COUNT(*) AS count
        FROM predictions
        {_where_clause(exclude_noise)}
        GROUP BY 1, 2
        """
    ).df()

    long_df["image"] = long_df["image_path"].apply(lambda p: disambiguated_stem(Path(p)))
    matrix = long_df.pivot_table(index="image", columns="label", values="count", fill_value=0, aggfunc="sum")
    matrix = matrix.sort_index()

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "label_by_image_matrix.csv"
    matrix.to_csv(matrix_path)
    return matrix_path, matrix.shape[0], matrix.shape[1]


@app.command()
def stats(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    output_dir: Optional[str] = typer.Option(
        None, help="If set, write label_by_image_matrix.csv here (rows=image, columns=label, value=box count)."
    ),
    exclude_noise: bool = typer.Option(
        False, "--exclude-noise/--no-exclude-noise",
        help="Exclude 'noise' from the label counts and the matrix. [default: no-exclude-noise]",
    ),
    top: Optional[int] = typer.Option(
        None, help="Only print the top N labels by count to the console (a written matrix CSV always includes every label)."
    ),
) -> None:
    """Print label counts and per-image detection stats; optionally write an
    image x label count matrix CSV for downstream ecological analysis.

    Uses new_label where curated, falling back to the raw detected label
    where it isn't (same convention `html` uses) -- so this works both
    before and after running through `review`/`cluster`.
    """
    with db.connect(db_path) as conn:
        total_rows = db.row_count(conn)
        if total_rows == 0:
            logger.warning("No predictions in %s; nothing to summarize.", db_path)
            raise typer.Exit(code=0)

        label_counts = _label_counts(conn, exclude_noise)
        box_stats = _boxes_per_image(conn, exclude_noise)

        matrix_info = _write_matrix(conn, Path(output_dir), exclude_noise) if output_dir else None

    noise_note = "excluding" if exclude_noise else "including"
    display = label_counts if top is None else label_counts.head(top)

    typer.echo(f"Label counts ({noise_note} noise):")
    typer.echo(display.to_string(index=False))
    if top is not None and len(label_counts) > top:
        typer.echo(f"... ({len(label_counts) - top} more label(s) not shown; increase --top to see more)")

    typer.echo(f"\nBoxes per image ({noise_note} noise):")
    typer.echo(box_stats.to_string(index=False))

    if matrix_info:
        matrix_path, n_images, n_labels = matrix_info
        typer.echo("")
        logger.info("Wrote image x label count matrix (%d image(s) x %d label(s)) to %s", n_images, n_labels, matrix_path)


if __name__ == "__main__":
    app()
