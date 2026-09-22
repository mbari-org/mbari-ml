"""Emit: ad hoc SQL queries against a curation database.

The original ``8_query.py`` wasn't really a script -- it was a notebook-style
file with a hardcoded absolute path to one specific database on its
author's machine, and a fixed sequence of
queries (including one that nulled out every ``new_label``/``evoc_clust``
value, uncommented, ready to run against whatever path was hardcoded at the
time). That's a rewrite hazard waiting to happen. This replaces it with a
small, safe, reusable CLI: pass the database path and a query explicitly.
"""

from __future__ import annotations

import typer

from mbariml import db
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Run an ad hoc SQL query against a curation database.")
logger = get_logger(__name__)


@app.command()
def query(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    sql: str = typer.Argument(..., help="SQL to run, e.g. \"SELECT new_label, COUNT(*) FROM predictions GROUP BY 1\"."),
    limit: int = typer.Option(50, help="Cap on rows printed to the console (the full result is still returned)."),
) -> None:
    """Run SQL against DB_PATH and print the result as a table."""
    with db.connect(db_path, must_exist=True) as conn:
        result = conn.execute(sql)
        df = result.df()

    typer.echo(df.head(limit).to_string())
    if len(df) > limit:
        typer.echo(f"... ({len(df) - limit} more row(s) not shown; increase --limit to see more)")


if __name__ == "__main__":
    app()
