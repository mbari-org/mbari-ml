"""Mutations (label, delete) for the ROI review GUI.

DuckDB-native replacement for vars-gridview's
``services/annotation_service.py`` (see ``THIRD_PARTY_NOTICES.md`` for the
parts of this GUI adapted from that project): there's no remote Annosaurus
API to push edits to, and no dirty-flag tracking on an association object --
just a DuckDB UPDATE/DELETE against the local ``predictions`` table, kept
here as the one place that performs writes so the mosaic/UI code never
touches SQL directly.
"""

from __future__ import annotations

import duckdb


def apply_label(
    conn: duckdb.DuckDBPyConnection, roi_indices: list[int], new_label: str
) -> None:
    """Set ``new_label`` for every ROI in *roi_indices*, and mark them
    ``verified`` at the same time.

    Labeling something *is* an act of review -- there's no point making the
    user separately click "Verify" right after every label they apply, so
    this sets both columns in one UPDATE. :func:`set_verified` remains the
    way to unverify (or re-verify without changing the label).

    One UPDATE with an IN-list rather than one UPDATE per row via
    ``executemany`` -- every selected ROI gets the same label, so there's no
    need to pay DuckDB's well-documented small-UPDATE overhead (see
    ``mbariml.db.bulk_update``, which targets the different case of many
    rows each getting a *different* value) for what could be hundreds of
    selected ROIs.
    """
    if not roi_indices:
        return
    placeholders = ", ".join("?" for _ in roi_indices)
    conn.execute(
        f"UPDATE predictions SET new_label = ?, verified = 1 WHERE roi_index IN ({placeholders})",
        [new_label, *roi_indices],
    )


def set_verified(
    conn: duckdb.DuckDBPyConnection, roi_indices: list[int], verified: bool
) -> None:
    """Set the ``verified`` flag for every ROI in *roi_indices*.

    Same one-UPDATE-with-an-IN-list shape as :func:`apply_label`, for the
    same reason: every selected ROI gets the same value.
    """
    if not roi_indices:
        return
    placeholders = ", ".join("?" for _ in roi_indices)
    conn.execute(
        f"UPDATE predictions SET verified = ? WHERE roi_index IN ({placeholders})",
        [1 if verified else 0, *roi_indices],
    )


def delete_rois(conn: duckdb.DuckDBPyConnection, roi_indices: list[int]) -> None:
    """Permanently delete the given ROIs."""
    if not roi_indices:
        return
    placeholders = ", ".join("?" for _ in roi_indices)
    conn.execute(
        f"DELETE FROM predictions WHERE roi_index IN ({placeholders})",
        list(roi_indices),
    )


def update_bbox(
    conn: duckdb.DuckDBPyConnection,
    roi_index: int,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    roi_blob: bytes | None,
) -> None:
    """Persist an edited box's geometry and its regenerated ROI crop.

    Box edits happen one at a time (a single drag-release in the detail
    view), so unlike :func:`apply_label` this is a plain single-row UPDATE
    rather than an IN-list -- there's no batch of rows sharing one new value
    to bulk-update.
    """
    conn.execute(
        """
        UPDATE predictions
        SET x_min = ?, y_min = ?, x_max = ?, y_max = ?, roi = ?
        WHERE roi_index = ?
        """,
        (x_min, y_min, x_max, y_max, roi_blob, roi_index),
    )


__all__ = ["apply_label", "set_verified", "delete_rois", "update_bbox"]
