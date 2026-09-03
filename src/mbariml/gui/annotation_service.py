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

from mbariml import db


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


def insert_roi(
    conn: duckdb.DuckDBPyConnection,
    image_name: str,
    image_path: str,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    label: str,
    roi_blob: bytes | None,
    sharpness: float,
) -> int:
    """Insert a brand-new, manually-drawn ROI (the review GUI's "Add New
    ROI" tool -- see ``MainWindow._on_new_box_drawn``) and return its
    ``roi_index``.

    ``id``/``roi_index`` are the same running counter every other writer
    (``infer_images``, ``infer_video``) uses -- see ``mbariml.db``'s
    schema docstring -- so the next free value is one past the current max
    ``id`` in this database; safe here since the review GUI is single-writer
    (no concurrent process is also inserting into this database). ``class_id``
    is left NULL (no YOLO class applies to a hand-drawn box) and
    ``confidence`` fixed at ``1.0`` (a human drew and named it -- there's no
    detector score to record). ``embedding`` is left NULL here -- the caller
    (``MainWindow._on_new_box_drawn``) fills it in moments later via a
    background worker and :func:`set_embedding`, so it doesn't sit NULL
    until someone remembers to run ``mbariml embed`` (though that remains a
    safe fallback if the background computation ever fails). Both ``label``
    (the raw-detection column) and ``new_label`` are set to the typed label
    immediately (there's no separate "raw" class to preserve), and
    ``verified`` is set to ``1`` -- a box a human just drew and labeled is
    definitionally reviewed, the same reasoning :func:`apply_label` already
    uses for an ordinary relabel.
    """
    next_id = db.next_free_id(conn)
    conn.execute(
        """
        INSERT INTO predictions
        (id, image_name, image_path, roi_index, x_min, y_min, x_max, y_max,
         class_id, confidence, label, embedding, new_label, roi, sharpness, verified)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 1.0, ?, NULL, ?, ?, ?, 1)
        """,
        (next_id, image_name, image_path, next_id, x_min, y_min, x_max, y_max, label, label, roi_blob, sharpness),
    )
    return next_id


def set_embedding(conn: duckdb.DuckDBPyConnection, roi_index: int, embedding: list[float]) -> None:
    """Store a freshly-computed embedding for one ROI.

    Used by the "Add New ROI" tool once its background embedding worker
    finishes (see ``MainWindow._on_new_roi_embedded``) -- a hand-drawn box
    would otherwise sit with ``embedding IS NULL`` (invisible to similarity
    search/clustering) until someone remembers to run ``mbariml embed``.
    A plain single-row UPDATE, same reasoning as :func:`update_bbox`: this
    happens one ROI at a time as each box is drawn, not in a batch worth
    ``mbariml.db.bulk_update``'s staged-bulk-UPDATE treatment (what
    ``mbariml embed`` itself uses for exactly that batched case).
    """
    conn.execute("UPDATE predictions SET embedding = ? WHERE roi_index = ?", (embedding, roi_index))


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


__all__ = ["apply_label", "set_verified", "delete_rois", "insert_roi", "set_embedding", "update_bbox"]
