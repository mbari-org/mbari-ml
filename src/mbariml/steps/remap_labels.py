"""Curate: bulk-rename `new_label` values from a two-column changes file.

Bug fixed here: this used to apply each (old_label, new_label) pair as its
own sequential UPDATE. That's fine for independent renames, but a changes
file with any chained or overlapping rule -- most obviously a two-way swap
like `A,B` / `B,A` -- silently corrupted the data instead of doing what a
"remap" file obviously means: relabel everything based on where it started,
all at once. Confirmed directly: a swap file (`A,B` then `B,A`) applied
sequentially renamed every A to B first, then renamed *every* B (including
the rows that had just become B) back to A, leaving every row -- both
original As and original Bs -- as 'A', with no error and a misleadingly
"successful" per-rule row count. Every pair is now applied as ONE UPDATE
against a staged changes table (same ``UPDATE ... FROM`` staging idiom as
``mbariml.db.bulk_update``, just joined on ``new_label`` instead of a row
id), so every row's new value is computed from its *original* new_label,
matching every other row's, in a single atomic pass -- a swap file now
actually swaps.
"""

from __future__ import annotations

import csv
from pathlib import Path

import typer

from mbariml import db
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Bulk-rename new_label values in the database from a two-column changes file.")
logger = get_logger(__name__)


def _load_changes(changes_file: Path) -> dict[str, str]:
    changes: dict[str, str] = {}
    with changes_file.open("r", encoding="utf-8-sig") as f:
        for line_num, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"{changes_file}:{line_num}: expected 2 columns (old_label,new_label), got {row!r}")
            old_label, new_label = row
            changes[old_label.strip()] = new_label.strip()
    return changes


def _apply_changes(conn, changes: dict[str, str]) -> dict[str, int]:
    """Apply every (old_label, new_label) pair as one UPDATE against a
    staged changes table, so each row's new value is computed from its
    original new_label -- not from whatever an earlier pair in this same
    run may have already changed it to (see the module docstring for the
    swap-corruption bug this fixes). Returns {old_label: affected_count}.
    """
    conn.execute("CREATE OR REPLACE TEMP TABLE _remap_changes (old_label TEXT, new_label TEXT)")
    db.fast_executemany(
        conn, "INSERT INTO _remap_changes VALUES (?, ?)", list(changes.items())
    )

    counts = dict(
        conn.execute(
            """
            SELECT _remap_changes.old_label, COUNT(*)
            FROM predictions
            JOIN _remap_changes ON predictions.new_label = _remap_changes.old_label
            GROUP BY _remap_changes.old_label
            """
        ).fetchall()
    )
    conn.execute(
        """
        UPDATE predictions
        SET new_label = _remap_changes.new_label
        FROM _remap_changes
        WHERE predictions.new_label = _remap_changes.old_label
        """
    )
    conn.execute("DROP TABLE _remap_changes")
    return counts


@app.command()
def remap_labels(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    changes_file: str = typer.Argument(..., help="CSV file of 'old_label,new_label' rows."),
) -> None:
    """Update the new_label column in the database using OLD_LABEL,NEW_LABEL pairs from CHANGES_FILE."""
    changes = _load_changes(Path(changes_file))
    if not changes:
        logger.warning("No changes found in %s; nothing to do.", changes_file)
        return

    with db.connect(db_path) as conn:
        counts = _apply_changes(conn, changes)

    for old_label, new_label in changes.items():
        logger.info("'%s' -> '%s': %d row(s)", old_label, new_label, counts.get(old_label, 0))
    logger.info("Done: %d row(s) updated across %d label(s).", sum(counts.values()), len(changes))


if __name__ == "__main__":
    app()
