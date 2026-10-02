"""Curate: interactive PySide6 GUI for reviewing and relabeling ROIs.

This is inherently interactive, so it isn't part of the scriptable
"start at any step"/``mbariml run`` chain -- launch it directly.

The GUI itself lives in ``mbariml.gui`` (a threaded, ``QGraphicsView``-based
mosaic adapted from MBARI's vars-gridview -- see
``mbariml/gui/__init__.py`` and ``THIRD_PARTY_NOTICES.md``); this module is
just the Typer command that wires it up. See ``mbariml.gui.main_window`` for
the feature list and what changed from the original single-file GUI.
"""

from __future__ import annotations

import sys

import typer

app = typer.Typer(help="Interactive GUI for reviewing and relabeling ROIs.")


@app.command()
def review(
    database_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    label: str = typer.Option(None, help="Filter ROIs by label"),
    page_size: int = typer.Option(500, help="Number of ROIs to show per page."),
    sam3_model: str = typer.Option(
        None,
        "--sam3-model",
        help="Path to SAM3 weights (sam3.pt) to enable 'Add ROI with SAM3' and 'Tighten Boxes'. "
        "Optional; defaults to $MBARIML_SAM3_MODEL. Also needs Ultralytics' CLIP: "
        "pip install git+https://github.com/ultralytics/CLIP.git",
    ),
) -> None:
    """Launch the ROI review/labeling GUI."""
    # Imported lazily: PySide6 is only needed for this one interactive command.
    from PySide6.QtWidgets import QApplication

    from mbariml.gui.main_window import MainWindow

    qt_app = QApplication(sys.argv)
    main_window = MainWindow(database_path, label, page_size, sam3_model=sam3_model)
    main_window.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    app()
