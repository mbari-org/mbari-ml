"""DuckDB queries backing the ROI review GUI.

This is the DuckDB-native replacement for vars-gridview's
``controllers/query_controller.py`` and the 30-field VARS ``Row`` model in
``ui/mosaic/image_mosaic.py`` (see ``THIRD_PARTY_NOTICES.md``) -- there is
no TSV wire format, no Annosaurus query DSL, and no video/observation/
association hierarchy to parse; a "row" here is just one record from the
flat ``predictions`` table (schema: ``mbariml.db.CURATION_SCHEMA_SQL``).

These queries are the same ones the original monolithic GUI ran inline
(see git history / the previous revision of ``review.py``);
they're centralized here, typed, and meant to be called off the GUI thread
via ``mbariml.gui.runnables.Worker``.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np

from mbariml.logging_utils import get_logger

logger = get_logger(__name__)

# UI-facing sort option -> SQL column/expression. This doubles as an
# allowlist: the value is interpolated directly into an ORDER BY clause
# (DuckDB can't parameterize identifiers or expressions), so only values
# from this dict -- all fixed, developer-written strings, never user input
# -- may be used.
#
# "Aspect Ratio" has no stored column: it's width/height computed directly
# from the box's existing x_min/y_min/x_max/y_max, the same values already
# fetched with every row (see _ROW_COLUMNS) -- no schema change, no backfill
# needed for it to work on an existing database. NULLIF guards a
# zero-height box (shouldn't happen, but would otherwise be a division by
# zero) rather than erroring the whole sort.
SORT_COLUMNS = {
    "New Label": "new_label",
    "Original Label": "label",
    "Image Name": "image_path",
    "Sharpness": "sharpness",
    "Aspect Ratio": "(x_max - x_min) / NULLIF(y_max - y_min, 0)",
}
DEFAULT_SORT_OPTION = "New Label"

_ROW_COLUMNS = (
    "id, image_path, roi_index, new_label, roi, "
    "x_min, y_min, x_max, y_max, embedding, label, verified, "
    "video_path, frame_time_s"
)


@dataclass
class RoiRow:
    """One reviewable ROI: a single row of the ``predictions`` table.

    ``label`` is the curated ``new_label`` (what quick-label/free-text
    labeling writes, what filtering/sorting/similarity-search use); it's
    ``None`` for every row until clustering or manual review sets it.
    ``original_label`` is the raw YOLO class from ``mbariml detect`` --
    always present, but never written to by this GUI. Both are fetched
    together so the "show new vs. original label" display toggle can flip
    instantly with no requery. ``verified`` is a plain reviewed/not-reviewed
    flag (see ``annotation_service.set_verified``), unrelated to labeling.
    """

    id: int
    image_path: str
    roi_index: int
    label: str | None
    roi_blob: bytes | None
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    embedding: list[float] | None
    original_label: str | None
    verified: bool = False
    # NULL for image-derived rows. Set by `mbariml infer video`, which points
    # image_path at an extracted frame on disk (so everything downstream needs
    # no video awareness) while keeping the trail back to the footage here --
    # what the review GUI's "Open Video" button uses to jump to the moment
    # this ROI was detected. See mbariml.db's schema.
    video_path: str | None = None
    frame_time_s: float | None = None

    @classmethod
    def _from_tuple(cls, row: tuple) -> "RoiRow":
        return cls(
            id=row[0],
            image_path=row[1],
            roi_index=row[2],
            label=row[3],
            roi_blob=row[4],
            x_min=row[5],
            y_min=row[6],
            x_max=row[7],
            y_max=row[8],
            embedding=row[9],
            original_label=row[10],
            verified=bool(row[11]),
            video_path=row[12],
            frame_time_s=row[13],
        )


# Shared by count_rows, fetch_page's normal-sort branch, count_verified, and
# compute_similarity_order's ranking pool, so "what counts as the current
# dataset" -- a label/search filter, hiding verified ROIs, a minimum-
# confidence floor -- is applied consistently everywhere a row count or
# ranking is computed, not just in what's visually hidden after the fact.
#
# This matters more than it looks: *_count/*_order values back "Page X/Y"
# and pagination itself. Applying a filter only client-side (after a full
# page_size's worth of rows was already fetched/ranked) leaves rows that
# get hidden still occupying page slots and still counted in the total --
# pages end up silently under-filled, and the total looks inflated relative
# to what's actually reachable. Confirmed directly: with "Hide verified"
# on, a similarity search reported "Page 1/80" (len(similarity_order),
# including verified ROIs that would just be hidden from every page) --
# not remotely what a reviewer means by "spans everything I still need to
# look at".
#
# label_filter matches EITHER new_label (curated) or label (raw YOLO
# class), not just new_label alone: the search dropdown is populated from
# both columns (see fetch_known_labels), so a concept that only exists as
# an original YOLO class -- the common case before clustering/review has
# ever run -- would otherwise silently match nothing.
def _build_where(
    label_filter: str | None,
    params: list,
    *,
    exclude_verified: bool = False,
    min_confidence: float | None = None,
) -> str:
    conditions = []
    if label_filter:
        conditions.append("(new_label = ? OR label = ?)")
        params.extend([label_filter, label_filter])
    if exclude_verified:
        conditions.append("(verified IS NULL OR verified != 1)")
    if min_confidence is not None and min_confidence > 0:
        conditions.append("confidence >= ?")
        params.append(min_confidence)
    if not conditions:
        return ""
    return " WHERE " + " AND ".join(conditions)


def count_rows(
    conn: duckdb.DuckDBPyConnection,
    label_filter: str | None = None,
    *,
    exclude_verified: bool = False,
    min_confidence: float | None = None,
) -> int:
    """Total rows matching the given filters (or the whole table if none) --
    backs the "Page X/Y" count and the jump-to-page range."""
    params: list = []
    query = "SELECT COUNT(*) FROM predictions" + _build_where(
        label_filter, params, exclude_verified=exclude_verified, min_confidence=min_confidence
    )
    return conn.execute(query, params).fetchone()[0]


def count_verified(
    conn: duckdb.DuckDBPyConnection,
    label_filter: str | None = None,
    *,
    min_confidence: float | None = None,
) -> tuple[int, int]:
    """(verified_count, unverified_count) among rows matching the given
    filters (or the whole table if none) -- backs the review-progress
    counter next to the page counts.

    Deliberately independent of "Hide verified" and any active similarity
    sort -- neither changes what "the dataset" means for review-progress
    purposes, they just change what's currently drawn/ranked. label_filter
    and min_confidence, on the other hand, narrow "what am I actually
    reviewing right now" the same way they narrow the grid, so this counter
    tracks them too. Cheap even at survey scale: a columnar aggregate that
    never touches the ROI blob/embedding columns.
    """
    params: list = []
    query = (
        "SELECT COUNT(*) FILTER (WHERE verified = 1), "
        "COUNT(*) FILTER (WHERE verified IS NULL OR verified != 1) "
        "FROM predictions"
    ) + _build_where(label_filter, params, min_confidence=min_confidence)
    verified_count, unverified_count = conn.execute(query, params).fetchone()
    return verified_count, unverified_count


def fetch_page(
    conn: duckdb.DuckDBPyConnection,
    *,
    offset: int,
    page_size: int,
    label_filter: str | None = None,
    sort_option: str = DEFAULT_SORT_OPTION,
    similarity_order: list[int] | None = None,
    exclude_verified: bool = False,
    min_confidence: float | None = None,
) -> list[RoiRow]:
    """Fetch one page's worth of ROIs.

    In normal mode, sorts by ``sort_option`` (must be a key of
    ``SORT_COLUMNS``) and pages via LIMIT/OFFSET, applying exclude_verified/
    min_confidence at the query level. In similarity mode (``similarity_order``
    given), slices the precomputed ``roi_index`` ordering instead and
    preserves its rank order -- the whole ROI set was already re-ranked (and
    filtered by those same criteria) by :func:`compute_similarity_order`, so
    they aren't reapplied here.
    """
    if similarity_order is not None:
        page_indices = similarity_order[offset : offset + page_size]
        if not page_indices:
            return []
        placeholders = ", ".join("?" for _ in page_indices)
        fetched = conn.execute(
            f"SELECT {_ROW_COLUMNS} FROM predictions WHERE roi_index IN ({placeholders})",
            page_indices,
        ).fetchall()
        by_roi_index = {row[2]: row for row in fetched}
        return [
            RoiRow._from_tuple(by_roi_index[idx])
            for idx in page_indices
            if idx in by_roi_index
        ]

    sort_column = SORT_COLUMNS.get(sort_option)
    if sort_column is None:
        logger.warning("Unknown sort option %r, falling back to %s", sort_option, DEFAULT_SORT_OPTION)
        sort_column = SORT_COLUMNS[DEFAULT_SORT_OPTION]

    params: list = []
    query = f"SELECT {_ROW_COLUMNS} FROM predictions" + _build_where(
        label_filter, params, exclude_verified=exclude_verified, min_confidence=min_confidence
    )
    query += f" ORDER BY {sort_column} LIMIT ? OFFSET ?"
    params.extend([page_size, offset])

    rows = conn.execute(query, params).fetchall()
    return [RoiRow._from_tuple(row) for row in rows]


@dataclass
class FrameRoi:
    """One detection on a source image, for the full-image detail overlay.

    Lighter than :class:`RoiRow` on purpose: the detail view draws every
    detection on the currently-shown image (not just the current mosaic
    page's rows), so this skips the ROI blob and embedding that overlay
    boxes never need.
    """

    id: int
    roi_index: int
    label: str | None
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    original_label: str | None

    @classmethod
    def _from_tuple(cls, row: tuple) -> "FrameRoi":
        return cls(
            id=row[0],
            roi_index=row[1],
            label=row[2],
            x_min=row[3],
            y_min=row[4],
            x_max=row[5],
            y_max=row[6],
            original_label=row[7],
        )


def fetch_rois_for_image(
    conn: duckdb.DuckDBPyConnection, image_path: str
) -> list[FrameRoi]:
    """Return every detection on *image_path*, regardless of mosaic paging/sort.

    Backs the full-image detail view: other detections on the same source
    frame may be sorted onto a different mosaic page than the one currently
    being viewed, but they should still show up (and be editable) here.
    Uses ``idx_predictions_image_path`` (see ``mbariml.db``), so this is a
    cheap indexed lookup, not a table scan -- safe to call on the GUI thread.
    """
    rows = conn.execute(
        """
        SELECT id, roi_index, new_label, x_min, y_min, x_max, y_max, label
        FROM predictions
        WHERE image_path = ?
        """,
        (image_path,),
    ).fetchall()
    return [FrameRoi._from_tuple(row) for row in rows]


def fetch_known_labels(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Every distinct label currently in use, alphabetically -- backs the
    relabel dropdown's suggestions (it's still a free-typeable combo box,
    this just supplies autocomplete/quick-pick candidates).

    Pulls from both ``new_label`` (curated) and ``label`` (raw YOLO class),
    not just ``new_label`` alone: before clustering/review has ever run,
    ``new_label`` is blank for every row, so a curated-only list would be
    empty exactly when picking from raw detections is most useful. Sorted
    case-insensitively so e.g. "Zebra" doesn't sort before "anemone".
    """
    rows = conn.execute(
        """
        SELECT DISTINCT label
        FROM (
            SELECT new_label AS label FROM predictions WHERE new_label IS NOT NULL AND new_label != ''
            UNION ALL
            SELECT label FROM predictions WHERE label IS NOT NULL AND label != ''
        )
        ORDER BY LOWER(label) ASC
        """
    ).fetchall()
    return [row[0] for row in rows]


# label_filter is always matched against one of these two columns -- an
# allowlist since the column name is interpolated directly into the query
# (DuckDB can't parameterize identifiers), same reasoning as SORT_COLUMNS.
SIMILARITY_LABEL_COLUMNS = {"new": "new_label", "original": "label"}


def compute_similarity_order(
    conn: duckdb.DuckDBPyConnection,
    roi_index: int,
    label_filter: str | None = None,
    label_mode: str = "new",
    *,
    exclude_verified: bool = False,
    min_confidence: float | None = None,
) -> list[int] | None:
    """Rank every ROI (respecting *label_filter*, if given) by cosine
    similarity of its embedding to ``roi_index``'s.

    *label_mode* picks which column *label_filter* is matched against --
    "new" (new_label, the default: the curated label, what the global
    ``--label`` filter and "same label" always meant before the "show
    original label" display toggle existed) or "original" (the raw YOLO
    class). Matters for "find similar with same label": that label_filter
    comes from whatever's currently displayed on the clicked tile, so it
    must be checked against the same column that produced it, or the filter
    either does nothing (comparing an original-label string against
    new_label, which is often still blank) or silently returns the wrong
    matches.

    exclude_verified/min_confidence narrow the ranking POOL itself (not
    just what's later hidden from display) -- see _build_where's docstring
    for why that distinction matters: excluding them only at display time
    left "Page X/Y" (== len(this ranking)) counting rows that could never
    actually be shown, and pages silently under-filled by however many of
    their slots landed on an excluded ROI.

    Returns ``roi_index`` values ordered most-similar-first, or ``None`` if
    the reference ROI has no embedding yet (requires ``mbariml embed`` to
    have run first). Does real work (a full-table scan + a numpy matmul) --
    call this from a worker thread, not the GUI thread.
    """
    ref_row = conn.execute(
        "SELECT embedding FROM predictions WHERE roi_index = ?", (roi_index,)
    ).fetchone()
    if not ref_row or ref_row[0] is None:
        logger.warning("ROI #%s has no embedding yet -- run `mbariml embed` first.", roi_index)
        return None

    label_column = SIMILARITY_LABEL_COLUMNS.get(label_mode, "new_label")
    conditions = ["embedding IS NOT NULL"]
    params: list = []
    if label_filter:
        conditions.append(f"{label_column} = ?")
        params.append(label_filter)
    if exclude_verified:
        conditions.append("(verified IS NULL OR verified != 1)")
    if min_confidence is not None and min_confidence > 0:
        conditions.append("confidence >= ?")
        params.append(min_confidence)
    query = "SELECT roi_index, embedding FROM predictions WHERE " + " AND ".join(conditions)
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return []

    indices = [r[0] for r in rows]
    matrix = np.asarray([r[1] for r in rows], dtype=np.float32)
    reference = np.asarray(ref_row[0], dtype=np.float32)

    # Mean-center before normalizing. Raw deep/self-supervised embeddings
    # (DINOv3 included) carry a strong "common mode" -- a handful of
    # directions nearly every image shares regardless of content -- that
    # dominates a plain dot product and dilutes the actually-discriminative
    # signal, so cosine similarity ends up compressed into a narrow, weakly
    # separated range and true nearest neighbors don't stand out from the
    # pack. Subtracting the ranking pool's own mean embedding before
    # normalizing removes that shared bias; this is a standard fix in
    # image-retrieval literature and costs nothing extra here -- the matrix
    # is already loaded for the matmul below. Centered on the POOL (not the
    # whole database) deliberately: it's the population actually being
    # compared right now, so this also naturally sharpens "same label" mode
    # (centers on just that label's own common mode) without extra code.
    mean = matrix.mean(axis=0)
    matrix = matrix - mean
    reference = reference - mean

    # Cosine similarity: normalize both sides, then dot product.
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_norms[matrix_norms == 0] = 1.0
    reference_norm = np.linalg.norm(reference) or 1.0
    similarities = (matrix / matrix_norms) @ (reference / reference_norm)

    order = np.argsort(-similarities)  # descending: most similar first
    return [indices[i] for i in order]


__all__ = [
    "RoiRow",
    "FrameRoi",
    "SORT_COLUMNS",
    "DEFAULT_SORT_OPTION",
    "count_rows",
    "count_verified",
    "fetch_page",
    "fetch_rois_for_image",
    "fetch_known_labels",
    "compute_similarity_order",
    "SIMILARITY_LABEL_COLUMNS",
]
