#!/usr/bin/env python3
"""Generate docs/SCHEMA.md from the live schema in ``mbariml.db``.

The column list, types and nullability are read back out of a real (in-memory)
DuckDB after applying ``CURATION_SCHEMA_SQL``, rather than transcribed by
hand -- so the document cannot drift from the code the way a hand-written
table would. Only the prose descriptions below are maintained by a human;
if a column is added to the schema without a description here, generation
fails loudly rather than emitting a silently incomplete table.

    python3 scripts/generate_schema_doc.py            # write docs/SCHEMA.md
    python3 scripts/generate_schema_doc.py --check    # fail if it's stale

Run it after changing CURATION_SCHEMA_SQL, alongside re-capturing the
cheat sheet's --help reference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import duckdb  # noqa: E402

from mbariml import db  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "docs" / "SCHEMA.md"

# column -> (written by, what it means). Maintained by hand; everything else
# on the page comes from the schema itself.
PREDICTIONS_DOCS: dict[str, tuple[str, str]] = {
    "id": (
        "ingest, review",
        "Unique row id, and the table's only UNIQUE index. Allocated from the "
        "database's own counter (`db.next_free_id`), so a second ingest run "
        "**appends** instead of colliding — which is what lets one database "
        "hold several videos, or images and video together. Always equal to "
        "`roi_index`.",
    ),
    "image_name": (
        "ingest",
        "Bare filename of the image this ROI came from. For video rows, the "
        "filename of the *extracted frame*, not the video.",
    ),
    "image_path": (
        "ingest",
        "**Absolute** path to the image on disk — for video rows, the extracted "
        "frame under `OUTPUT_DIR/frames/`. This is how `review` and every "
        "export find the pixels again later, from whatever directory they run "
        "in, so it is deliberately not relative.",
    ),
    "roi_index": (
        "ingest, review",
        "Same value as `id`. It's the key the review GUI and "
        "`annotation_service` address rows by.",
    ),
    "x_min": ("ingest, review", "Left edge of the box, in pixels of `image_path`."),
    "y_min": ("ingest, review", "Top edge of the box, in pixels of `image_path`."),
    "x_max": ("ingest, review", "Right edge of the box, in pixels of `image_path`."),
    "y_max": ("ingest, review", "Bottom edge of the box, in pixels of `image_path`."),
    "class_id": (
        "ingest",
        "The model's own class index. `NULL` for ROIs drawn by hand in the "
        "review GUI — no model class applies to those.",
    ),
    "confidence": (
        "ingest",
        "Detector confidence. Fixed at `1.0` for hand-drawn ROIs. The review "
        "GUI's min-confidence slider filters on this; nothing ever rewrites it.",
    ),
    "label": (
        "ingest",
        "The **raw** class name the model predicted, never rewritten by "
        "curation. For hand-drawn ROIs it's set to the typed label, since "
        "there's no model prediction to preserve.",
    ),
    "embedding": (
        "embed",
        "DINOv3 embedding of the ROI crop. `NULL` until `mbariml embed` runs "
        "(hand-drawn ROIs get one immediately, from the review GUI's "
        "background worker). **Never mix embeddings from two different models "
        "in one database** — clustering and similarity search would compare "
        "vectors from different spaces; re-embed with `--force` instead.",
    ),
    "new_label": (
        "cluster, refine, review, remap-labels",
        "The **curated** label. This is what `cluster` writes, what the review "
        "GUI edits, and what every export filters on — rows labeled `'noise'` "
        "are excluded from `export voc`/`yolo`/`id`. `NULL` until something "
        "curates it, which is why display and `stats` fall back to "
        "`COALESCE(new_label, label)`.",
    ),
    "roi": (
        "ingest, review",
        "The JPEG-encoded crop itself. `embed`, `cluster`, and the review "
        "mosaic all read this, which is why they never need the source image "
        "on disk. Regenerated when a box is dragged in the review GUI.",
    ),
    "sharpness": (
        "ingest",
        "Variance of the Laplacian — a cheap blur score, computed at ingest "
        "from the crop. Higher is sharper. Backs the review GUI's \"Sort by "
        "Sharpness\", which surfaces unusable crops for deletion.",
    ),
    "verified": (
        "review",
        "`1` once a human has reviewed the row. Applying a label sets it too — "
        "labeling something *is* an act of review. Backs the \"Hide verified\" "
        "filter and the review-progress counter.",
    ),
    "video_path": (
        "infer video",
        "Absolute path to the **source video**. `NULL` for image-derived rows. "
        "The review GUI's \"Open Video\" button uses this with `frame_time_s`.",
    ),
    "frame_number": (
        "infer video",
        "Which frame of `video_path` this ROI was detected on. `NULL` for "
        "image-derived rows.",
    ),
    "frame_time_s": (
        "infer video",
        "That frame's timestamp in seconds — what \"Open Video\" seeks to.",
    ),
    "track_id": (
        "infer video (track mode)",
        "The track this ROI represents. Exactly one row per track is written, "
        "so this is unique per video within a tracking run. `NULL` in stride "
        "mode and for image rows.",
    ),
    "track_length": (
        "infer video (track mode)",
        "How many observations backed that track. Useful provenance when "
        "judging an ROI: a 3-frame track is far weaker evidence than a "
        "300-frame one.",
    ),
    "evoc_clust": (
        "cluster",
        "Raw EVoC cluster number (`-1` is EVoC's noise bucket). Added "
        "dynamically by `mbariml cluster` via `db.ensure_column`, so it only "
        "exists once clustering has run — which is why it's absent from the "
        "table below unless you've clustered.",
    ),
}

RUN_INFO_DOCS: dict[str, tuple[str, str]] = {
    "model_path": ("ingest", "The model that produced this database's detections."),
    "detected_at": ("ingest", "When that ingest run happened."),
}


def _columns(conn, table: str) -> list[tuple[str, str, bool]]:
    """(name, type, nullable) straight from the live database."""
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [(r[1], r[2], not r[3]) for r in rows]


def _table_section(conn, table: str, docs: dict[str, tuple[str, str]], intro: str) -> str:
    lines = [f"## `{table}`", "", intro, "", "| Column | Type | Null? | Written by | Meaning |", "|---|---|---|---|---|"]
    for name, sql_type, nullable in _columns(conn, table):
        if name not in docs:
            raise SystemExit(
                f"ERROR: column '{table}.{name}' has no description in "
                f"{Path(__file__).name}. Add one so this page stays complete."
            )
        written_by, meaning = docs[name]
        lines.append(f"| `{name}` | `{sql_type}` | {'yes' if nullable else 'no'} | {written_by} | {meaning} |")
    return "\n".join(lines)


def render() -> str:
    conn = duckdb.connect(":memory:")
    conn.execute(db.CURATION_SCHEMA_SQL)

    header = f"""<!-- Generated by scripts/{Path(__file__).name} -- do not edit by hand.
     Regenerate after changing CURATION_SCHEMA_SQL in src/mbariml/db.py. -->

# Database schema

Every `mbariml` command reads and writes this one schema, which is what lets
any command's output feed any other (see [README.md](../README.md)). It's a
DuckDB file, so anything here is queryable directly:

```bash
mbariml query results/yolo_predictions.duckdb \\
    "SELECT new_label, COUNT(*) FROM predictions GROUP BY 1 ORDER BY 2 DESC"
```

One row is **one ROI** — a single detection, with its crop stored inline. In
video tracking mode that means one row per *track*, not per frame.
"""

    predictions = _table_section(
        conn, "predictions", PREDICTIONS_DOCS,
        "The main table. Columns are `NULL` until whichever command owns them "
        "runs — a freshly-ingested database has no `embedding` or `new_label` "
        "yet, and the video columns stay `NULL` for image-derived rows.",
    )
    run_info = _table_section(
        conn, "run_info", RUN_INFO_DOCS,
        "Provenance for the ingest run, so later commands (the `*.id` export, "
        "for one) can report which model produced these detections without the "
        "caller retyping it. Replaced by each ingest run.",
    )

    dynamic = """## Columns added later

`evoc_clust` (`INTEGER`) is added by `mbariml cluster` via
`db.ensure_column`, so it exists only after clustering has run. `verified`
and the five video columns were likewise added to existing databases with
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, which is why opening an older
database with a newer mbariml migrates it in place rather than failing.
"""

    return "\n\n".join([header.strip(), predictions, run_info, dynamic.strip()]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if the committed page is out of date, without writing.")
    args = parser.parse_args()

    rendered = render()
    if args.check:
        current = OUTPUT_PATH.read_text() if OUTPUT_PATH.exists() else ""
        if current != rendered:
            sys.exit(f"{OUTPUT_PATH.relative_to(REPO_ROOT)} is stale -- re-run {Path(__file__).name}")
        print(f"{OUTPUT_PATH.relative_to(REPO_ROOT)} is up to date.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered)
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
