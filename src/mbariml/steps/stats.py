"""Emit: label counts, boxes-per-image summary stats, and an optional
image x label count matrix CSV -- the raw numbers behind an ecological read
of a curation database (e.g. "how many Muusoctopus per image, and across how
many images").

Every query here counts the same population the exports write: VERIFIED
localizations, named ``COALESCE(new_label, label)`` -- the curated name
where a reviewer retyped it, the raw YOLO ``label`` where they confirmed it
unchanged (``mbariml.db.curated_where``). Pass --include-unverified to count
raw un-reviewed detections too, which is what makes this useful on a
database fresh out of `infer`, before anything has been reviewed. Noise is included
by default (seeing how much of a database is still 'noise' is itself a
useful number while curating) -- pass --exclude-noise once you actually want
ecological counts of real identifications only.

Numbered like every other step for reference (this doc, the cheat sheet,
--help), but -- like review (5), query (7), infer-images (8), and
remap-labels (9) -- it isn't part of `mbariml run`'s scriptable chain (only
detect/embed/cluster/export are): run it directly, any time, read-only,
against whatever database you already have.
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

# Kept as a module-level alias so this file still reads the same, but the
# definition now lives in mbariml.db alongside the WHERE clause, so stats and
# the exports cannot drift apart again.
_EFFECTIVE_LABEL_SQL = db.EFFECTIVE_LABEL_SQL


def _where_clause(exclude_noise: bool, include_unverified: bool = False) -> str:
    """Counts VERIFIED localizations by default -- the same population the
    exports write, so `stats` can be trusted as a preview of what a training
    set will contain. That equivalence is the point: this command previously
    counted every row while `export yolo` wrote only relabelled ones, so a
    database could report 35,492 localizations across 24 classes and export
    2,805, with nothing anywhere saying the two numbers meant different
    things.

    --include-unverified restores counting raw un-reviewed detections, which
    is what makes this useful on a database fresh out of `infer`.
    """
    return db.curated_where(
        exclude_noise=exclude_noise, require_verified=not include_unverified
    )


def _label_counts(conn, exclude_noise: bool, include_unverified: bool) -> pd.DataFrame:
    df = conn.execute(
        f"""
        SELECT {_EFFECTIVE_LABEL_SQL} AS label, COUNT(*) AS count
        FROM predictions
        {_where_clause(exclude_noise, include_unverified)}
        GROUP BY 1
        ORDER BY count DESC
        """
    ).df()
    total = df["count"].sum()
    df["percent"] = (df["count"] / total * 100).round(2) if total else df["count"]
    return df


def _boxes_per_image(conn, exclude_noise: bool, include_unverified: bool) -> pd.DataFrame:
    return conn.execute(
        f"""
        WITH per_image AS (
            SELECT image_path, COUNT(*) AS boxes
            FROM predictions
            {_where_clause(exclude_noise, include_unverified)}
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


def _write_matrix(conn, output_dir: Path, exclude_noise: bool, include_unverified: bool) -> tuple[Path, int, int]:
    """Writes label_by_image_matrix.csv: rows=image (disambiguated stem,
    collision-safe across dives -- see mbariml.image_naming), columns=label,
    value=box count. Returns (path, n_images, n_labels)."""
    long_df = conn.execute(
        f"""
        SELECT image_path, {_EFFECTIVE_LABEL_SQL} AS label, COUNT(*) AS count
        FROM predictions
        {_where_clause(exclude_noise, include_unverified)}
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
    include_unverified: bool = typer.Option(
        False, "--include-unverified/--verified-only",
        help="Also count localizations nobody has verified yet. Off by default, so these counts "
             "match what `export yolo`/`voc`/`id` write; turn it on to summarize raw detector "
             "output before review. [default: verified-only]",
    ),
    top: Optional[int] = typer.Option(
        None, help="Only print the top N labels by count to the console (a written matrix CSV always includes every label)."
    ),
) -> None:
    """Print label counts and per-image detection stats; optionally write an
    image x label count matrix CSV for downstream ecological analysis.

    Counts verified localizations, named new_label where curated and the raw
    detected label where confirmed unchanged -- the same rule, and so the
    same numbers, as `export yolo`/`voc`/`id`. Pass --include-unverified to
    count raw detections as well, e.g. before any review has happened.
    """
    with db.connect(db_path) as conn:
        total_rows = db.row_count(conn)
        if total_rows == 0:
            logger.warning("No predictions in %s; nothing to summarize.", db_path)
            raise typer.Exit(code=0)

        if not include_unverified and not db.has_column(conn, "verified"):
            logger.warning(
                "%s predates the 'verified' column, so nothing counts as verified. Showing raw "
                "detections instead -- pass --include-unverified to silence this.", db_path,
            )
            include_unverified = True

        label_counts = _label_counts(conn, exclude_noise, include_unverified)
        box_stats = _boxes_per_image(conn, exclude_noise, include_unverified)

        matrix_info = (
            _write_matrix(conn, Path(output_dir), exclude_noise, include_unverified)
            if output_dir else None
        )

        # An empty result here is nearly always an unreviewed database, not an
        # empty one. Say which, rather than printing a table of zeroes.
        if label_counts.empty and not include_unverified and total_rows:
            logger.warning(
                "None of the %d localization(s) in %s are verified, so there is nothing to count. "
                "Verify some in `mbariml review`, or pass --include-unverified to summarize the "
                "raw detections.", total_rows, db_path,
            )
            raise typer.Exit(code=0)

    noise_note = "excluding" if exclude_noise else "including"
    verified_note = "verified + unverified" if include_unverified else "verified only"
    display = label_counts if top is None else label_counts.head(top)

    typer.echo(f"Label counts ({verified_note}, {noise_note} noise):")
    typer.echo(display.to_string(index=False))
    if top is not None and len(label_counts) > top:
        typer.echo(f"... ({len(label_counts) - top} more label(s) not shown; increase --top to see more)")

    typer.echo(f"\nBoxes per image ({verified_note}, {noise_note} noise):")
    typer.echo(box_stats.to_string(index=False))

    if matrix_info:
        matrix_path, n_images, n_labels = matrix_info
        typer.echo("")
        logger.info("Wrote image x label count matrix (%d image(s) x %d label(s)) to %s", n_images, n_labels, matrix_path)


if __name__ == "__main__":
    app()
