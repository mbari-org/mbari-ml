"""Shared DuckDB helpers.

One schema (``CURATION_SCHEMA_SQL``) is used across the whole pipeline,
including video ingest: it stores the ROI image itself (as a JPEG
blob), its embedding, and the human-curated ``new_label``. Image inference used to
write a separate, lighter schema (no ROI blob, no embedding) -- but that
meant its output couldn't be fed into `mbariml review`, `cluster`, `refine`,
`export voc`, or `remap-labels` at all, none of which is what you want from
a "run inference on a new survey, then review/export it" workflow. Every
step now reads/writes the same schema, so any step's output is usable by
any other step that needs what it has.

``connect()`` is a context manager that guarantees the connection is closed
(and therefore flushed) even if the caller raises. The original
``9_inference.py`` opened a DuckDB connection and never closed it, relying on
the interpreter to clean it up on exit; combined with buffering all insert
rows in memory until the very end of the run (see ``mbariml.steps.infer_images``),
that meant a run that hit any error, or was interrupted, could finish having
written nothing at all despite YOLO visibly having processed every image.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator

import duckdb
import typer

from mbariml.logging_utils import get_logger

logger = get_logger(__name__)

CURATION_SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER,
        image_name TEXT,
        image_path TEXT,
        roi_index INTEGER,
        x_min FLOAT,
        y_min FLOAT,
        x_max FLOAT,
        y_max FLOAT,
        class_id INTEGER,
        confidence FLOAT,
        label TEXT,
        embedding FLOAT[],
        new_label TEXT,
        roi BLOB,
        sharpness DOUBLE
    );
    CREATE UNIQUE INDEX IF NOT EXISTS predictions_id_idx ON predictions (id);
    CREATE INDEX IF NOT EXISTS idx_predictions_new_label ON predictions(new_label);
    CREATE INDEX IF NOT EXISTS idx_predictions_image_path ON predictions(image_path);
    CREATE INDEX IF NOT EXISTS idx_predictions_roi_index ON predictions(roi_index);

    -- Added after the fact rather than in the CREATE TABLE column list
    -- above: several commands (infer_images, infer_video) INSERT into
    -- this table positionally ("VALUES (?, ?, ..., ?)", no column names),
    -- so a column added to the CREATE TABLE list would silently require
    -- updating every one of those in lockstep or break them. ALTER TABLE
    -- ADD COLUMN sidesteps that entirely, and DEFAULT 0 backfills existing
    -- rows on an already-populated (pre-`verified`) production database,
    -- not just newly created ones.
    ALTER TABLE predictions ADD COLUMN IF NOT EXISTS verified INTEGER DEFAULT 0;

    -- Video provenance, NULL for every image-derived row (added the same
    -- ALTER TABLE way, and for the same reason, as `verified` above).
    -- `mbariml infer video` extracts the frame it detected on to a real JPEG
    -- on disk and points image_path at THAT file, so every downstream step
    -- (review, embed, cluster, all four exports, stats) treats a video row
    -- exactly like an image row with no special-casing anywhere. These
    -- columns exist so the trail back to the source footage isn't lost:
    -- which video, which frame, when in the video, and -- in tracking mode
    -- -- which track this ROI was chosen to represent and how many
    -- observations backed it (a 3-frame track is much weaker evidence than
    -- a 200-frame one). The review GUI's "Open Video" button uses
    -- video_path + frame_time_s.
    ALTER TABLE predictions ADD COLUMN IF NOT EXISTS video_path TEXT;
    ALTER TABLE predictions ADD COLUMN IF NOT EXISTS frame_number INTEGER;
    ALTER TABLE predictions ADD COLUMN IF NOT EXISTS frame_time_s DOUBLE;
    ALTER TABLE predictions ADD COLUMN IF NOT EXISTS track_id INTEGER;
    ALTER TABLE predictions ADD COLUMN IF NOT EXISTS track_length INTEGER;

    -- Records which model produced this database's detections, so later
    -- steps (e.g. the *.id export) can report accurate provenance without
    -- the caller having to remember/retype it. One row per `mbariml detect`
    -- run against this database; each ingest run replaces it.
    CREATE TABLE IF NOT EXISTS run_info (
        model_path TEXT,
        detected_at TIMESTAMP
    );
"""

def _suggest_nearby_databases(db_path: Path) -> str:
    """'did you mean' text listing real databases sitting next to db_path.

    A mistyped extension (``yolo_predictions.duckdbb``) is the overwhelmingly
    likely reason a database is missing, and the right one is almost always
    in the same directory -- so name it rather than making the caller go
    look.
    """
    if not db_path.parent.is_dir():
        return ""
    nearby = sorted(c.name for c in db_path.parent.glob("*.duckdb") if c.is_file())
    if not nearby:
        return ""
    if len(nearby) == 1:
        return f" Did you mean {nearby[0]}?"
    return f" Databases in that directory: {', '.join(nearby)}."


@contextlib.contextmanager
def connect(db_path: str | Path, *, must_exist: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a DuckDB connection that is always closed on the way out.

    Always use this (or ``init_curation_db``) instead of calling
    ``duckdb.connect`` directly -- an unclosed connection is the easiest way
    to end up with a database file that looks empty even though the run
    "seemed to work".

    ``must_exist=True`` for every command that READS an existing database.
    ``duckdb.connect`` silently CREATES a database at whatever path it is
    given, so without this a single mistyped character
    (``yolo_predictions.duckdbb``) leaves a new, empty database on disk and
    then fails several frames later with "Table with name predictions does
    not exist" -- which points at the schema, not at the typo that actually
    caused it. Confirmed directly: that typo created a 12 KB phantom
    database next to a real 1 GB one.
    """
    db_path = Path(db_path)
    if must_exist and not db_path.exists():
        # typer.BadParameter specifically, not a subclass and not a plain
        # exception: Typer renders exactly this class as a one-line "Error"
        # panel and nothing else as anything but a full rich traceback
        # (checked directly against click 8.5.0 -- a bare click.UsageError,
        # a bare ClickException and even a BadParameter *subclass* all print
        # a traceback). A mistyped path is a usage error, not a crash, and a
        # stack dump for one buries the single line that says what is wrong.
        raise typer.BadParameter(
            f"No such database: {db_path}.{_suggest_nearby_databases(db_path)}"
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    logger.debug("Opened DuckDB connection: %s", db_path)
    try:
        yield conn
    finally:
        conn.close()
        logger.debug("Closed DuckDB connection: %s", db_path)


@contextlib.contextmanager
def init_curation_db(db_path: str | Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open (creating if needed) the curation-schema database used by every
    command, images and video alike."""
    with connect(db_path) as conn:
        conn.execute(CURATION_SCHEMA_SQL)
        logger.info("Curation schema ready in %s", db_path)
        yield conn


def ensure_column(conn: duckdb.DuckDBPyConnection, table: str, column: str, sql_type: str) -> None:
    """Add ``column`` to ``table`` if it isn't already there."""
    conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {sql_type}")


# The one definition of "what label does this localization carry", used by
# every downstream consumer (all four exports, stats). Per row, not per
# database: `new_label` where a reviewer retyped it, the raw detector
# `label` where they looked at it and left it alone.
#
# This exists as one shared constant because it previously did not, and the
# three dataset exports (yolo/voc/id) drifted onto a bare `new_label`
# filter while stats/cluster/export-html used the COALESCE. Since the GUI's
# Verify button sets `verified = 1` WITHOUT writing new_label (only
# relabelling writes it -- see gui/annotation_service.py), that filter meant
# "boxes whose name I retyped", not "boxes I confirmed". On a real 35,492-row
# survey database it exported 2,805 boxes and dropped 32,687 confirmed ones
# -- and because 29,917 of those sat on images that WERE in the export, YOLO
# read them as unlabeled background and was actively trained against the
# reviewer's own identifications.
EFFECTIVE_LABEL_SQL = "COALESCE(new_label, label)"


def curated_where(*, exclude_noise: bool = True, require_verified: bool = True) -> str:
    """The shared WHERE clause selecting curated localizations.

    ``require_verified`` is the important half: an unverified row is raw
    detector output no human has confirmed, and it has no business in a
    training set or an identification sidecar. It is only ever relaxed by
    the two *diagnostic* consumers (`stats`, `export html`), which are
    documented to be useful against a database that has not been reviewed
    yet, and only behind an explicit --include-unverified flag.

    Note `cluster` deliberately does NOT use this: clustering exists to
    group and name data that has NOT been reviewed, so restricting it to
    verified rows would defeat its purpose.
    """
    conditions = []
    if require_verified:
        conditions.append("verified = 1")
    conditions.append(f"{EFFECTIVE_LABEL_SQL} IS NOT NULL")
    if exclude_noise:
        conditions.append(f"{EFFECTIVE_LABEL_SQL} != 'noise'")
    return "WHERE " + " AND ".join(conditions)


def has_column(conn: duckdb.DuckDBPyConnection, column: str, table: str = "predictions") -> bool:
    return column in {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}


def require_verified_column(conn: duckdb.DuckDBPyConnection, db_path: str | Path) -> None:
    """Fail loudly if this database predates the ``verified`` column.

    Such a database records no verification state at all, so a
    ``verified = 1`` filter would match nothing and write a perfectly
    well-formed EMPTY dataset -- silently, which is the exact failure mode
    the effective-label fix exists to end. Better to stop and say so.
    ``mbariml review`` opens databases with init_curation_db() and runs the
    ALTER TABLE migration, so pointing it at the database once repairs it.
    """
    if not conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'predictions'"
    ).fetchone()[0]:
        raise RuntimeError(
            f"{db_path} has no 'predictions' table, so it is not an mbariml curation database."
        )
    if not has_column(conn, "verified"):
        raise RuntimeError(
            f"{db_path} predates the 'verified' column, so it records no review state and "
            "nothing can be selected as curated. Open it once with `mbariml review` (which "
            "migrates the schema), verify some ROIs, then re-run this export."
        )


def next_free_id(conn: duckdb.DuckDBPyConnection) -> int:
    """One past the highest ``id`` already in ``predictions``.

    Every writer allocates ids from this counter (``id`` and ``roi_index`` are
    kept equal, and ``id`` is UNIQUE -- see ``predictions_id_idx`` above), so
    starting from the current max is what lets a second ingest run append to a
    database that already holds rows instead of colliding on the very first
    insert. That's what makes "one database for a whole deployment -- several
    videos, or images and video together" work, and it's why re-running an
    ingest command into an existing output directory no longer fails.
    """
    return conn.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM predictions").fetchone()[0]


def row_count(conn: duckdb.DuckDBPyConnection, table: str = "predictions") -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def fast_executemany(conn: duckdb.DuckDBPyConnection, sql: str, rows: list[tuple]) -> None:
    """``conn.executemany(sql, rows)``, wrapped in an explicit transaction.

    DuckDB's Python driver commits (and fsyncs to disk) after *every
    individual statement* by default when writing to a file-backed
    database -- even within one ``executemany()`` call. Measured directly:
    the identical 2000-row INSERT took 11.46s without an explicit
    transaction vs. 0.82s with one, on the same hardware and storage --
    a ~14x difference from wrapping alone. Use this instead of calling
    ``conn.executemany`` directly anywhere more than a handful of rows are
    being written to a persistent (non-temp) table.
    """
    if not rows:
        return
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.executemany(sql, rows)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def bulk_update(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    key_column: str,
    key_type: str,
    set_columns: dict[str, str],
    rows: list[tuple],
) -> None:
    """Apply many rows' worth of column updates with ONE bulk UPDATE instead
    of one UPDATE per row.

    DuckDB is a columnar/OLAP engine: many small row-by-row UPDATEs are
    *documented* to be dramatically slower than one bulk UPDATE, since each
    one carries MVCC row-versioning overhead that compounds as the count
    grows -- and it gets markedly worse when the updated column is indexed.
    Measured directly on this schema: 100,000 rows via
    ``executemany("UPDATE ... WHERE id = ?", ...)`` against the indexed
    ``new_label`` column took 38.6 seconds and was still on a worsening
    trend; the same 100,000 rows via this staged bulk UPDATE took 6.0
    seconds -- and the staging INSERT itself is also wrapped in an explicit
    transaction (see ``fast_executemany``), since that alone is a further
    ~14x on file-backed storage. This was the actual root cause of both a
    slow `embed` step and, worse (an indexed column), a `cluster` step that
    could run for hours despite the clustering math itself finishing in
    well under a minute.

    Args:
        table: table to update.
        key_column: column identifying which row to update (e.g. "id").
        key_type: SQL type of key_column (e.g. "INTEGER").
        set_columns: ``{column_name: sql_type}`` for every column being set.
        rows: each tuple is ``(key_value, value_for_first_set_column, ...)``,
            in the same order as ``set_columns``.
    """
    if not rows:
        return

    columns = [key_column] + list(set_columns.keys())
    types = [key_type] + list(set_columns.values())
    column_defs = ", ".join(f"{c} {t}" for c, t in zip(columns, types))
    placeholders = ", ".join("?" for _ in columns)
    set_clause = ", ".join(f"{c} = _bulk_update_staging.{c}" for c in set_columns)

    conn.execute(f"CREATE OR REPLACE TEMP TABLE _bulk_update_staging ({column_defs})")
    fast_executemany(conn, f"INSERT INTO _bulk_update_staging VALUES ({placeholders})", rows)
    conn.execute(
        f"""
        UPDATE {table}
        SET {set_clause}
        FROM _bulk_update_staging
        WHERE {table}.{key_column} = _bulk_update_staging.{key_column}
        """
    )
    conn.execute("DROP TABLE _bulk_update_staging")
