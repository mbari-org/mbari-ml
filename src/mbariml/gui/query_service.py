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

from mbariml import db
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
#
# "New Label" sorts by the EFFECTIVE label (see EFFECTIVE_LABEL_SQL): a row
# with no new_label yet sorts under its original label, interleaved with the
# renamed rows by name, rather than all piling up unsorted at the end.
#
# NULLIF, unlike db.EFFECTIVE_LABEL_SQL's plain COALESCE: the GUI treats an
# empty-string new_label as "no new label" (it's falsy on a row), so SQL must
# too, or such a row would sort/match as '' instead of by its original label.
EFFECTIVE_LABEL_SQL = "COALESCE(NULLIF(new_label, ''), label)"

#
# Confidence comes both ways round: high-low to review the detector's surest
# calls first, low-high to clear out false positives. DuckDB sorts NULLs last
# either way, so a row with no score never leads the grid.
#
# Sharpness is the crop's Laplacian variance (higher = sharper), so it sorts
# descending: the sharpest crops first, the blurriest last.
SORT_COLUMNS = {
    "New Label": EFFECTIVE_LABEL_SQL,
    "Original Label": "label",
    "Image Name": "image_path",
    "Sharpness": "sharpness DESC",
    "Aspect Ratio": "(x_max - x_min) / NULLIF(y_max - y_min, 0)",
    "Confidence (high-low)": "confidence DESC",
    "Confidence (low-high)": "confidence ASC",
}
DEFAULT_SORT_OPTION = "New Label"

# What the Sort... dialog can stack (modelled on vars-gridview's sort dialog):
# name -> (SQL expression, descending by default, (ascending text, descending
# text)). Same allowlist rule as SORT_COLUMNS: a stacked ORDER BY is joined
# from these fixed expressions only.
SORT_KEYS = {
    "New Label": (EFFECTIVE_LABEL_SQL, False, ("A-Z", "Z-A")),
    "Original Label": ("label", False, ("A-Z", "Z-A")),
    "Image Name": ("image_path", False, ("A-Z", "Z-A")),
    "Sharpness": ("sharpness", True, ("blurriest first", "sharpest first")),
    "Confidence": ("confidence", True, ("lowest first", "highest first")),
    "Box Size": ("(x_max - x_min) * (y_max - y_min)", True, ("smallest first", "largest first")),
    "Aspect Ratio": ("(x_max - x_min) / NULLIF(y_max - y_min, 0)", False, ("tallest first", "widest first")),
}

# Stacked sorts made in the dialog, by the name shown in the Sort by
# dropdown: name -> (ORDER BY expression, [(key, descending), ...]).
_STACKED_SORTS: dict[str, tuple[str, list[tuple[str, bool]]]] = {}


def sort_key_text(key: str, descending: bool) -> str:
    """'Sharpness (sharpest first)'."""
    return f"{key} ({SORT_KEYS[key][2][descending]})"


def stacked_sort_option(keys: list[tuple[str, bool]]) -> str:
    """Register a stack of (SORT_KEYS name, descending) -- the first key
    sorts, each next one breaks the previous one's ties -- and return its
    sort option name, e.g. 'New Label (A-Z), then Sharpness (sharpest first)'."""
    if not keys:
        raise ValueError("a stacked sort needs at least one key")
    name = ", then ".join(sort_key_text(key, desc) for key, desc in keys)
    order = ", ".join(f"{SORT_KEYS[key][0]} {'DESC' if desc else 'ASC'}" for key, desc in keys)
    _STACKED_SORTS[name] = (order, list(keys))
    return name


def stacked_sort_keys(sort_option: str) -> list[tuple[str, bool]] | None:
    """The (key, descending) stack behind a stacked sort option, else None."""
    entry = _STACKED_SORTS.get(sort_option)
    return list(entry[1]) if entry else None


def sort_sql(sort_option: str) -> str | None:
    """The ORDER BY expression for a dropdown sort or a stacked one; None for
    sorts SQL can't express (outliers, by animal, list order)."""
    if sort_option in SORT_COLUMNS:
        return SORT_COLUMNS[sort_option]
    entry = _STACKED_SORTS.get(sort_option)
    return entry[0] if entry else None

# No embedding: no tile uses it, and fetching it cost more than everything
# else on a page together -- 500 tiles x 1024 floats became 512,000 Python
# floats per page. Measured on a 575,796-row database: page 1 went 364 ->
# 152 ms without it, the last page 1,240 -> 373 ms (and see fetch_page for
# the rest). Similarity search loads embeddings itself.
_ROW_COLUMNS = (
    "id, image_path, roi_index, new_label, roi, "
    "x_min, y_min, x_max, y_max, label, verified, "
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
            original_label=row[9],
            verified=bool(row[10]),
            video_path=row[11],
            frame_time_s=row[12],
        )


_CONFIDENCE_SLACK = 1e-6  # far above FLOAT rounding (~3e-8 near 0.5), far below the slider's 0.01 steps


def _confidence_conditions(min_confidence: float | None, max_confidence: float | None, params: list) -> list[str]:
    """The confidence-range part of a WHERE clause (both ends inclusive).
    Either end left at its extreme -- min 0 or None, max 1 or None -- adds no
    condition, so with the full range rows with a NULL confidence still show;
    any narrowing excludes them, since NULL is in no range.

    Both ends get a hair of slack (_CONFIDENCE_SLACK): ``confidence`` is a
    32-bit FLOAT, so a stored 0.30 is really 0.30000001 and a stored 0.70 is
    0.69999999 -- compared exactly, a 0.20-0.30 range left out the 0.30s and
    a 0.70 floor the 0.70s."""
    conditions = []
    if min_confidence is not None and min_confidence > 0:
        conditions.append("confidence >= ?")
        params.append(min_confidence - _CONFIDENCE_SLACK)
    if max_confidence is not None and max_confidence < 1:
        conditions.append("confidence <= ?")
        params.append(max_confidence + _CONFIDENCE_SLACK)
    return conditions


# Shared by count_rows, fetch_page's normal-sort branch, count_verified, and
# compute_similarity_order's ranking pool, so "what counts as the current
# dataset" -- a label/search filter, hiding verified ROIs, a confidence
# range -- is applied consistently everywhere a row count or
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
# label_filter matches each row's *effective* label: new_label (curated) where
# it's set, else label (the raw YOLO class) -- EFFECTIVE_LABEL_SQL, the same
# rule display, the "New Label" sort and `stats` use. Not new_label alone: a
# concept that exists only as an original class -- every row, before
# clustering/review has ever run -- would match nothing. And not "new_label
# OR label", as it once did: that also matched rows already renamed to
# something else, so searching "trash" kept showing tiles relabeled "animal".
def _roi_indices_conditions(roi_indices: list[int] | None, params: list) -> list[str]:
    """Restrict to *roi_indices* (an open box list) -- or nothing if None."""
    if roi_indices is None:
        return []
    params.append(roi_indices)
    return ["roi_index IN (SELECT unnest(?::INTEGER[]))"]


def _build_where(
    label_filter: str | None,
    params: list,
    *,
    exclude_verified: bool = False,
    exclude_unverified: bool = False,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    roi_indices: list[int] | None = None,
) -> str:
    conditions = []
    if label_filter:
        conditions.append(f"{EFFECTIVE_LABEL_SQL} = ?")
        params.append(label_filter)
    if exclude_verified:
        conditions.append("(verified IS NULL OR verified != 1)")
    if exclude_unverified:
        conditions.append("verified = 1")
    conditions += _confidence_conditions(min_confidence, max_confidence, params)
    conditions += _roi_indices_conditions(roi_indices, params)
    if not conditions:
        return ""
    return " WHERE " + " AND ".join(conditions)


def count_rows(
    conn: duckdb.DuckDBPyConnection,
    label_filter: str | None = None,
    *,
    exclude_verified: bool = False,
    exclude_unverified: bool = False,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> int:
    """Total rows matching the given filters (or the whole table if none) --
    backs the "Page X/Y" count and the jump-to-page range."""
    params: list = []
    query = "SELECT COUNT(*) FROM predictions" + _build_where(
        label_filter, params, exclude_verified=exclude_verified,
        exclude_unverified=exclude_unverified, min_confidence=min_confidence, max_confidence=max_confidence,
    )
    return conn.execute(query, params).fetchone()[0]


def count_verified(
    conn: duckdb.DuckDBPyConnection,
    label_filter: str | None = None,
    *,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> tuple[int, int]:
    """(verified_count, unverified_count) among rows matching the given
    filters (or the whole table if none) -- backs the review-progress
    counter next to the page counts.

    Deliberately independent of "Hide verified"/"Hide unverified" and any active similarity
    sort -- neither changes what "the dataset" means for review-progress
    purposes, they just change what's currently drawn/ranked. label_filter
    and the confidence range, on the other hand, narrow "what am I actually
    reviewing right now" the same way they narrow the grid, so this counter
    tracks them too. Cheap even at survey scale: a columnar aggregate that
    never touches the ROI blob/embedding columns.
    """
    params: list = []
    query = (
        "SELECT COUNT(*) FILTER (WHERE verified = 1), "
        "COUNT(*) FILTER (WHERE verified IS NULL OR verified != 1) "
        "FROM predictions"
    ) + _build_where(label_filter, params, min_confidence=min_confidence, max_confidence=max_confidence)
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
    exclude_unverified: bool = False,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> list[RoiRow]:
    """Fetch one page's worth of ROIs.

    In normal mode, sorts by ``sort_option`` (must be a key of
    ``SORT_COLUMNS``) and pages via LIMIT/OFFSET, applying exclude_verified/
    exclude_unverified/min_confidence/max_confidence at the query level. In similarity mode (``similarity_order``
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

    sort_column = sort_sql(sort_option)
    if sort_column is None:
        logger.warning("Unknown sort option %r, falling back to %s", sort_option, DEFAULT_SORT_OPTION)
        sort_column = SORT_COLUMNS[DEFAULT_SORT_OPTION]

    # Two steps: sort only the narrow (id, sort key) columns to find the
    # page's ids, then fetch just those rows -- ROI blobs and all -- by id.
    # Sorting whole rows dragged every ROI blob through the sort; on a
    # 575,796-row database this took page 1 from 152 to 30 ms and the last
    # page from 373 to 103 ms (1,240 ms before the embedding was dropped).
    # `id` breaks ties so paging is deterministic: with only a non-unique
    # key (a label shared by 100,000 rows) LIMIT/OFFSET may order the ties
    # differently on each query, repeating some ROIs across pages and
    # never showing others.
    params: list = []
    where_sql = _build_where(
        label_filter, params, exclude_verified=exclude_verified,
        exclude_unverified=exclude_unverified, min_confidence=min_confidence, max_confidence=max_confidence,
    )
    query = f"""
        WITH page AS (
            SELECT id, row_number() OVER () AS page_rank
            FROM (SELECT id FROM predictions{where_sql} ORDER BY {sort_column}, id LIMIT ? OFFSET ?)
        )
        SELECT {_ROW_COLUMNS} FROM predictions JOIN page USING (id) ORDER BY page_rank
    """
    params.extend([page_size, offset])

    rows = conn.execute(query, params).fetchall()
    return [RoiRow._from_tuple(row) for row in rows]


def count_hidden_by_confidence(
    conn: duckdb.DuckDBPyConnection,
    label_filter: str | None = None,
    *,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> list[tuple[str, int, int]]:
    """``(effective label, total, hidden)`` per class: how many ROIs the
    confidence range keeps out of view, most hidden first.

    Scoped like :func:`count_verified` -- by the search filter and nothing
    else, so it answers "what is the confidence range hiding from what I'm
    reviewing" whatever Hide verified/unverified is set to. "Hidden" uses
    the same test as the grid (:func:`_confidence_conditions`), so a NULL
    confidence counts as hidden once the range is narrowed. A columnar
    aggregate that never touches ROI blobs: ~5 ms on 575,796 rows.
    """
    params: list = []
    shown_conditions = _confidence_conditions(min_confidence, max_confidence, params)
    shown_sql = " AND ".join(shown_conditions) or "TRUE"
    where_sql = _build_where(label_filter, params)
    rows = conn.execute(
        f"""
        SELECT COALESCE({EFFECTIVE_LABEL_SQL}, '(no label)') AS l,
               COUNT(*) AS total,
               COUNT(*) - COUNT(*) FILTER (WHERE {shown_sql}) AS hidden
        FROM predictions{where_sql}
        GROUP BY l
        ORDER BY hidden DESC, total DESC, LOWER(l)
        """,
        params,
    ).fetchall()
    return [(label, int(total), int(hidden)) for label, total, hidden in rows]


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


def fetch_effective_labels(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Every distinct *effective* label (new_label where set, else label),
    alphabetically -- exactly the values a search (see ``_build_where``) can
    match, so every choice in the search dropdown finds something. An
    original class that every row has been renamed away from isn't listed."""
    rows = conn.execute(
        f"""
        SELECT DISTINCT {EFFECTIVE_LABEL_SQL} AS l
        FROM predictions
        WHERE l IS NOT NULL AND l != ''
        ORDER BY LOWER(l) ASC
        """
    ).fetchall()
    return [row[0] for row in rows]


# label_filter is always matched against one of these columns -- an
# allowlist since the column name is interpolated directly into the query
# (DuckDB can't parameterize identifiers), same reasoning as SORT_COLUMNS.
SIMILARITY_LABEL_COLUMNS = {
    "new": "new_label",
    "original": "label",
    "effective": EFFECTIVE_LABEL_SQL,
}


def similarity_embedding_column(conn: duckdb.DuckDBPyConnection) -> str:
    """The embedding column similarity search and the outlier sort compare:
    BioCLIP-2's (``mbariml embed --model bioclip2``) once a database has any,
    else DINOv3's. BioCLIP-2, trained on the tree of life, ranked deep-sea
    taxa better (see mbariml.steps.embed)."""
    if db.has_column(conn, "bioclip_embedding") and conn.execute(
        "SELECT 1 FROM predictions WHERE bioclip_embedding IS NOT NULL LIMIT 1"
    ).fetchone():
        return "bioclip_embedding"
    return "embedding"


def embed_command_for(column: str) -> str:
    """The command that fills *column*, for messages."""
    return "mbariml embed --model bioclip2" if column == "bioclip_embedding" else "mbariml embed"


def _build_similarity_pool_where(
    label_filter: str | None,
    label_mode: str,
    *,
    exclude_verified: bool = False,
    exclude_unverified: bool = False,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    require_embedding: bool,
    roi_indices: list[int] | None = None,
    embedding_column: str = "embedding",
) -> tuple[str, list]:
    """WHERE clause defining a similarity search's ranking pool.

    Shared by :func:`compute_similarity_order` (which ranks the pool) and
    :func:`count_similarity_pool` (which counts it *without* the embedding
    requirement), so the two can never drift into describing different
    populations -- the whole point of the count is to be a truthful
    denominator for the ranking.
    """
    label_column = SIMILARITY_LABEL_COLUMNS.get(label_mode, "new_label")
    conditions = []
    params: list = []
    if require_embedding:
        conditions.append(f"{embedding_column} IS NOT NULL")
    if label_filter:
        conditions.append(f"{label_column} = ?")
        params.append(label_filter)
    if exclude_verified:
        conditions.append("(verified IS NULL OR verified != 1)")
    if exclude_unverified:
        conditions.append("verified = 1")
    conditions += _confidence_conditions(min_confidence, max_confidence, params)
    conditions += _roi_indices_conditions(roi_indices, params)
    if not conditions:
        return "", params
    return " WHERE " + " AND ".join(conditions), params


def count_similarity_pool(
    conn: duckdb.DuckDBPyConnection,
    label_filter: str | None = None,
    label_mode: str = "new",
    *,
    exclude_verified: bool = False,
    exclude_unverified: bool = False,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    roi_indices: list[int] | None = None,
) -> int:
    """How many ROIs a similarity search *could* rank -- the same pool
    :func:`compute_similarity_order` ranks, minus its ``embedding IS NOT
    NULL`` requirement.

    Exists purely as a denominator the GUI can show. A similarity search
    always ranks the entire matching ROI set, every page of it -- but it can
    only rank rows that actually have an embedding, and a partial or
    interrupted `mbariml embed` (or one run with --limit) silently leaves
    some without. There was previously no way to tell those two situations
    apart from inside the GUI: a search that ranked 400 of 20,000 ROIs and
    one that ranked all 400 ROIs there were both just displayed as a page
    count, which reads exactly like "it only sorted what I was looking at".
    Reporting ranked-of-total makes the distinction visible, and points at
    the actual fix (run `mbariml embed`) when it is one.
    """
    where_sql, params = _build_similarity_pool_where(
        label_filter,
        label_mode,
        exclude_verified=exclude_verified,
        exclude_unverified=exclude_unverified,
        min_confidence=min_confidence, max_confidence=max_confidence,
        require_embedding=False, roi_indices=roi_indices,
    )
    return conn.execute("SELECT COUNT(*) FROM predictions" + where_sql, params).fetchone()[0]


def compute_similarity_order(
    conn: duckdb.DuckDBPyConnection,
    roi_index: int,
    label_filter: str | None = None,
    label_mode: str = "new",
    *,
    exclude_verified: bool = False,
    exclude_unverified: bool = False,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    roi_indices: list[int] | None = None,
) -> list[int] | None:
    """Rank every ROI (respecting *label_filter*, if given) by cosine
    similarity of its embedding to ``roi_index``'s.

    *label_mode* picks which column *label_filter* is matched against --
    "new" (new_label, the default), "original" (the raw YOLO class), or
    "effective" (new_label where set, else the original label -- what "Find
    similar with same label" uses, so a tile that was never renamed is
    matched by its original label and one that was renamed only by its new
    one). Matters for "find similar with same label": that label_filter
    comes from whatever's currently displayed on the clicked tile, so it
    must be checked against the same column that produced it, or the filter
    either does nothing (comparing an original-label string against
    new_label, which is often still blank) or silently returns the wrong
    matches.

    exclude_verified/exclude_unverified/the confidence range narrow the ranking POOL itself (not
    just what's later hidden from display) -- see _build_where's docstring
    for why that distinction matters: excluding them only at display time
    left "Page X/Y" (== len(this ranking)) counting rows that could never
    actually be shown, and pages silently under-filled by however many of
    their slots landed on an excluded ROI. *roi_indices* (an open box list)
    narrows the pool to those boxes.

    Returns ``roi_index`` values ordered most-similar-first, or ``None`` if
    the reference ROI has no embedding yet (requires ``mbariml embed`` to
    have run first). Does real work (a full-table scan + a numpy matmul) --
    call this from a worker thread, not the GUI thread.
    """
    column = similarity_embedding_column(conn)
    ref_row = conn.execute(
        f"SELECT {column} FROM predictions WHERE roi_index = ?", (roi_index,)
    ).fetchone()
    if not ref_row or ref_row[0] is None:
        logger.warning("ROI #%s has no %s yet -- run `%s` first.", roi_index, column, embed_command_for(column))
        return None

    where_sql, params = _build_similarity_pool_where(
        label_filter,
        label_mode,
        exclude_verified=exclude_verified,
        exclude_unverified=exclude_unverified,
        min_confidence=min_confidence, max_confidence=max_confidence,
        require_embedding=True, roi_indices=roi_indices, embedding_column=column,
    )
    indices, matrix = _embedding_matrix(conn, where_sql, params, len(ref_row[0]), column)
    if len(indices) == 0:
        return []
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
    return indices[order].tolist()


def _embedding_matrix(conn, where_sql: str, params: list, dim: int, column: str = "embedding") -> tuple[np.ndarray, np.ndarray]:
    """``(roi_index values, float32 matrix of their embeddings)`` for the rows *where_sql* selects.

    Fetched with fetchnumpy, which hands each row's list over as one numpy
    array, then stacked -- not fetchall(), which built a Python list of 1024
    Python floats per ROI and, on a 575,796-row database, spent 46 of a
    whole-table similarity search's 49 seconds doing it. ~2 s now for
    540,938 rows of either column. An earlier version fetched one row per
    float (``unnest``) and relied on each ROI's values coming out
    contiguous; DuckDB doesn't promise that, and for a column filled by a
    bulk UPDATE (bioclip_embedding) they came out interleaved.
    """
    query = f"SELECT roi_index, {column} AS value FROM predictions" + where_sql
    fetched = conn.execute(query, params).fetchnumpy()
    row_ids, values = fetched["roi_index"], fetched["value"]
    if len(values) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, dim), dtype=np.float32)
    try:
        matrix = np.stack(values).astype(np.float32, copy=False)
    except ValueError:
        matrix = None
    if matrix is None or matrix.shape[1] != dim:
        raise ValueError(f"{column} values of different lengths in one database -- re-run `{embed_command_for(column)} --force`")
    return np.asarray(row_ids), matrix


# Not in SORT_COLUMNS: no SQL expression can rank this, so it's computed in
# Python (compute_outlier_order) and paged like a similarity ranking.
OUTLIER_SORT_OPTION = "Outliers in label"


def compute_outlier_order(
    conn: duckdb.DuckDBPyConnection,
    label_filter: str | None = None,
    *,
    exclude_verified: bool = False,
    exclude_unverified: bool = False,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    roi_indices: list[int] | None = None,
) -> list[int]:
    """Every matching ROI's ``roi_index``, label by label (alphabetically,
    like the "New Label" sort), and within each label the least typical first.

    "Typical" is cosine similarity to the label's mean embedding, after
    centring on the whole pool's mean (the same common-mode correction as
    :func:`compute_similarity_order`). After a long run of near-identical
    tiles, the ones that don't look like the rest of their label -- a
    mislabel, a bad crop, a different animal -- come first instead of being
    buried on page 40. ROIs with no embedding come last in their label.
    Filters narrow the pool exactly as they do for similarity search;
    *roi_indices* (an open box list) narrows it to those boxes.
    Does real work (a full scan + numpy) -- call from a worker thread.
    """
    where_sql, params = _build_similarity_pool_where(
        label_filter, "effective",
        exclude_verified=exclude_verified, exclude_unverified=exclude_unverified,
        min_confidence=min_confidence, max_confidence=max_confidence,
        require_embedding=False, roi_indices=roi_indices,
    )
    pool = conn.execute(
        f"SELECT roi_index, COALESCE({EFFECTIVE_LABEL_SQL}, '') AS effective_label FROM predictions" + where_sql, params
    ).fetchnumpy()
    pool_ids = np.asarray(pool["roi_index"])
    if len(pool_ids) == 0:
        return []
    label_of = dict(zip(pool_ids.tolist(), pool["effective_label"].tolist()))

    column = similarity_embedding_column(conn)
    dim_row = conn.execute(f"SELECT len({column}) FROM predictions WHERE {column} IS NOT NULL LIMIT 1").fetchone()
    score: dict[int, float] = {}
    if dim_row is not None:
        emb_where, emb_params = _build_similarity_pool_where(
            label_filter, "effective",
            exclude_verified=exclude_verified, exclude_unverified=exclude_unverified,
            min_confidence=min_confidence, max_confidence=max_confidence,
            require_embedding=True, roi_indices=roi_indices, embedding_column=column,
        )
        indices, matrix = _embedding_matrix(conn, emb_where, emb_params, dim_row[0], column)
        if len(indices):
            matrix -= matrix.mean(axis=0)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix /= norms
            labels = np.array([label_of[i] for i in indices.tolist()], dtype=object)
            for label in np.unique(labels):
                rows = np.flatnonzero(labels == label)
                centroid = matrix[rows].mean(axis=0)
                centroid /= np.linalg.norm(centroid) or 1.0
                for i, sim in zip(indices[rows].tolist(), (matrix[rows] @ centroid).tolist()):
                    score[i] = sim
    # No embedding -> +inf: after every ranked ROI of its label.
    ranked = sorted(label_of, key=lambda i: (label_of[i].lower(), score.get(i, float("inf")), i))
    return ranked


# Only in the dropdown while a box list is open: the list's own (file) order.
# Every other sort then reorders just the listed boxes.
LIST_ORDER_OPTION = "List order"


def read_box_list(path: str) -> tuple[list[int], dict[int, str]]:
    """Box ids from a CSV, in file order, for review's Open List (and
    ``review --list``): a ``roi_index`` (or ``id``) column, plus an optional
    ``suggested`` or ``note`` column whose text each tile shows. Lists from
    other tools -- an anomaly sweep, a VARS report -- open as they are."""
    import csv

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = {name.lower().strip(): name for name in (reader.fieldnames or [])}
        id_col = fields.get("roi_index") or fields.get("id")
        if id_col is None:
            raise ValueError(f"{path}: needs a roi_index (or id) column; found {reader.fieldnames}")
        note_col = fields.get("suggested") or fields.get("note")
        ids: list[int] = []
        notes: dict[int, str] = {}
        for row in reader:
            value = (row.get(id_col) or "").strip()
            if not value:
                continue
            roi_index = int(float(value))
            if roi_index not in notes:
                ids.append(roi_index)
                notes[roi_index] = (row.get(note_col) or "").strip() if note_col else ""
    return ids, notes


def filter_box_list(
    conn: duckdb.DuckDBPyConnection,
    ids: list[int],
    label_filter: str | None = None,
    *,
    sort_option: str = LIST_ORDER_OPTION,
    exclude_verified: bool = False,
    exclude_unverified: bool = False,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
) -> list[int]:
    """The listed boxes that still exist and pass the current search, Hide
    verified/unverified and confidence range -- in list order, or sorted by
    *sort_option* (a ``SORT_COLUMNS`` key or a stacked sort) within the list."""
    params: list = []
    where = _build_where(
        label_filter, params, exclude_verified=exclude_verified, exclude_unverified=exclude_unverified,
        min_confidence=min_confidence, max_confidence=max_confidence, roi_indices=ids,
    )
    sort_column = sort_sql(sort_option)
    if sort_column is not None:
        return [r[0] for r in conn.execute(
            f"SELECT roi_index FROM predictions{where} ORDER BY {sort_column}, id", params
        ).fetchall()]
    keep = {r[0] for r in conn.execute("SELECT roi_index FROM predictions" + where, params).fetchall()}
    return [i for i in ids if i in keep]


# Not in SORT_COLUMNS either: one tile per animal from `mbariml fixtures`.
ANIMAL_SORT_OPTION = "By animal (fixed camera)"


@dataclass
class AnimalInfo:
    """One fixture from ``mbariml fixtures``, as the "By animal" sort shows it.

    ``label`` is the most common effective label among its boxes right now
    (it changes as boxes are relabelled), ``n_agree`` how many boxes carry
    it. ``rep_roi_index`` is the tile shown for the whole animal.
    """

    fixture: int
    rep_roi_index: int
    label: str
    n_agree: int
    n: int
    n_unverified: int
    n_agree_unverified: int  # unverified boxes that carry ``label``

    @property
    def confirmed(self) -> bool:
        """Every box carrying the animal's label is verified -- what V does.
        Boxes labelled otherwise may still be waiting (see caption)."""
        return self.n_agree_unverified == 0

    @property
    def caption(self) -> str:
        """Counts, not percentages: 12 odd boxes out of 7,330 round to "100%".
        Once confirmed, says how many boxes labelled otherwise are left, so a
        V on an animal with a few odd boxes visibly did something."""
        differ = self.n - self.n_agree
        text = f"{self.n:,} boxes" + (f", {differ:,} differ" if differ else "")
        if self.confirmed and self.n_unverified:
            text += f", {self.n_unverified:,} left"
        return text


def has_fixtures(conn: duckdb.DuckDBPyConnection) -> bool:
    return bool(conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name IN ('fixtures', 'fixture_info')"
    ).fetchone()[0] == 2)


def compute_animal_order(
    conn: duckdb.DuckDBPyConnection,
    label_filter: str | None = None,
    *,
    exclude_verified: bool = False,
    exclude_unverified: bool = False,
    fixtures: list[int] | None = None,
) -> list[AnimalInfo]:
    """Every fixture (or just *fixtures*): animals whose boxes all carry one
    label first, the most mixed last (count of boxes labelled otherwise),
    biggest first within a tie.

    A search matches the animal's majority label. Hide verified drops confirmed
    animals (every box carrying the label verified -- odd boxes left over show
    up in the box-by-box sorts); Hide unverified drops animals with nothing
    verified. The confidence range doesn't apply: an animal is one decision
    whatever its boxes scored. Labels compare case-insensitively, as search does.
    """
    rows = conn.execute(f"""
        WITH members AS (
            SELECT f.fixture, COALESCE({EFFECTIVE_LABEL_SQL}, '') AS lab, COALESCE(p.verified, 0) = 1 AS ver
            FROM fixtures f JOIN predictions p USING (roi_index)
            WHERE ?::INTEGER[] IS NULL OR f.fixture IN (SELECT unnest(?::INTEGER[]))),
        by_label AS (
            SELECT fixture, lab, count(*) AS k, count(*) FILTER (WHERE NOT ver) AS k_unverified,
                   row_number() OVER (PARTITION BY fixture ORDER BY count(*) DESC, lab) AS rn
            FROM members GROUP BY fixture, lab),
        totals AS (
            SELECT fixture, count(*) AS n, count(*) FILTER (WHERE NOT ver) AS n_unverified
            FROM members GROUP BY fixture)
        SELECT t.fixture, i.rep_roi_index, b.lab, b.k, t.n, t.n_unverified, b.k_unverified
        FROM totals t JOIN by_label b ON b.fixture = t.fixture AND b.rn = 1
        JOIN fixture_info i ON i.fixture = t.fixture
        ORDER BY t.n - b.k, t.n DESC, t.fixture""", [fixtures, fixtures]).fetchall()
    animals = [AnimalInfo(*r) for r in rows]
    if label_filter:
        wanted = label_filter.lower()
        animals = [a for a in animals if a.label.lower() == wanted]
    if exclude_verified:
        animals = [a for a in animals if not a.confirmed]
    if exclude_unverified:
        animals = [a for a in animals if a.n_unverified < a.n]
    return animals


def fetch_animal_members(
    conn: duckdb.DuckDBPyConnection,
    fixture: int,
    *,
    exclude_verified: bool = False,
    exclude_unverified: bool = False,
) -> list[int]:
    """One animal's boxes, least typical first: frames where something else
    sits in that spot, or the detector's box slipped, come before thousands of
    identical ones. Boxes with no typicality (no embedding) come last."""
    where = ""
    if exclude_verified:
        where = " AND COALESCE(p.verified, 0) != 1"
    elif exclude_unverified:
        where = " AND COALESCE(p.verified, 0) = 1"
    return [r[0] for r in conn.execute(
        f"""SELECT f.roi_index FROM fixtures f JOIN predictions p USING (roi_index)
            WHERE f.fixture = ?{where} ORDER BY f.typicality ASC NULLS LAST, f.roi_index""",
        [fixture],
    ).fetchall()]


__all__ = [
    "RoiRow",
    "LIST_ORDER_OPTION",
    "read_box_list",
    "filter_box_list",
    "AnimalInfo",
    "ANIMAL_SORT_OPTION",
    "has_fixtures",
    "compute_animal_order",
    "fetch_animal_members",
    "FrameRoi",
    "SORT_COLUMNS",
    "DEFAULT_SORT_OPTION",
    "SORT_KEYS",
    "sort_key_text",
    "stacked_sort_option",
    "stacked_sort_keys",
    "sort_sql",
    "count_rows",
    "count_verified",
    "count_hidden_by_confidence",
    "fetch_page",
    "fetch_rois_for_image",
    "fetch_known_labels",
    "fetch_effective_labels",
    "compute_similarity_order",
    "similarity_embedding_column",
    "embed_command_for",
    "compute_outlier_order",
    "OUTLIER_SORT_OPTION",
    "count_similarity_pool",
    "SIMILARITY_LABEL_COLUMNS",
]
