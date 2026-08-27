"""Shared DuckDB helpers.

One schema (``CURATION_SCHEMA_SQL``) is used across the whole pipeline,
including step 9 (inference): it stores the ROI image itself (as a JPEG
blob), its embedding, and the human-curated ``new_label``. Step 9 used to
write a separate, lighter schema (no ROI blob, no embedding) -- but that
meant its output couldn't be fed into `mbariml review`, `cluster`, `refine`,
`export-voc`, or `remap-labels` at all, none of which is what you want from
a "run inference on a new survey, then review/export it" workflow. Every
step now reads/writes the same schema, so any step's output is usable by
any other step that needs what it has.

``connect()`` is a context manager that guarantees the connection is closed
(and therefore flushed) even if the caller raises. The original
``9_inference.py`` opened a DuckDB connection and never closed it, relying on
the interpreter to clean it up on exit; combined with buffering all insert
rows in memory until the very end of the run (see ``mbariml.steps.step9_inference``),
that meant a run that hit any error, or was interrupted, could finish having
written nothing at all despite YOLO visibly having processed every image.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator

import duckdb

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
    -- above: several steps (step1_detect, step9_inference) INSERT into
    -- this table positionally ("VALUES (?, ?, ..., ?)", no column names),
    -- so a column added to the CREATE TABLE list would silently require
    -- updating every one of those in lockstep or break them. ALTER TABLE
    -- ADD COLUMN sidesteps that entirely, and DEFAULT 0 backfills existing
    -- rows on an already-populated (pre-`verified`) production database,
    -- not just newly created ones.
    ALTER TABLE predictions ADD COLUMN IF NOT EXISTS verified INTEGER DEFAULT 0;

    -- Records which model produced this database's detections, so later
    -- steps (e.g. the *.id export) can report accurate provenance without
    -- the caller having to remember/retype it. One row per `mbariml detect`
    -- run against this database; step 1 replaces it each time it runs.
    CREATE TABLE IF NOT EXISTS run_info (
        model_path TEXT,
        detected_at TIMESTAMP
    );
"""

@contextlib.contextmanager
def connect(db_path: str | Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a DuckDB connection that is always closed on the way out.

    Always use this (or ``init_curation_db``) instead of calling
    ``duckdb.connect`` directly -- an unclosed connection is the easiest way
    to end up with a database file that looks empty even though the run
    "seemed to work".
    """
    db_path = Path(db_path)
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
    step, including step 9 (inference)."""
    with connect(db_path) as conn:
        conn.execute(CURATION_SCHEMA_SQL)
        logger.info("Curation schema ready in %s", db_path)
        yield conn


def ensure_column(conn: duckdb.DuckDBPyConnection, table: str, column: str, sql_type: str) -> None:
    """Add ``column`` to ``table`` if it isn't already there."""
    conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {sql_type}")


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
