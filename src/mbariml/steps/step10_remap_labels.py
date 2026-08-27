"""Step 10: bulk-rename `new_label` values from a two-column changes file."""

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
        total_updated = 0
        for old_label, new_label in changes.items():
            affected = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE new_label = ?", (old_label,)
            ).fetchone()[0]
            if affected:
                conn.execute("UPDATE predictions SET new_label = ? WHERE new_label = ?", (new_label, old_label))
            total_updated += affected
            logger.info("'%s' -> '%s': %d row(s)", old_label, new_label, affected)

    logger.info("Done: %d row(s) updated across %d label(s).", total_updated, len(changes))


if __name__ == "__main__":
    app()
