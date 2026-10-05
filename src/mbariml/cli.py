"""Unified CLI for the mbariml pipeline.

The pipeline is five stages (as named in the paper), not a numbered list of
steps (the numbering was dropped in v0.11.0 -- it had been renumbered three
times as commands merged and moved, and every renumbering meant touching
every docstring and both docs, for a sequence that was never actually linear):

    Import   import yolo | import voc        existing labeled dataset -> database
    Generate infer images | infer video      pixels + detections -> database
    Enrich   embed | cluster | refine        add embeddings and grouping
    Curate   review | remap-labels           human review and relabeling
    Export   export {voc,yolo,id,html,stats} | query

Every command reads and writes the SAME database schema (see ``mbariml.db``),
so any command's output is usable by any other that needs what it has. That's
what lets you start anywhere: import an existing training set, or point
``infer video`` at new footage, and go straight to ``review`` on the result,
or run ``embed``/``cluster`` over a database built by any Import or Generate
command.

``mbariml run`` chains the scriptable part of that (ingest -> embed -> cluster
-> export) for convenience, over images or video, and can start or stop
at any step of the chain via ``--from``/``--to``. The interactive review GUI, ad
hoc queries, label remapping, and stats aren't in the chain -- they're not
batch operations, or they don't belong in the middle of one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mbariml.logging_utils import get_logger
from mbariml.steps import (
    cluster as cluster_step,
    embed as embed_step,
    export_html,
    export_ids,
    export_voc,
    export_yolo,
    import_voc,
    import_yolo,
    infer_images,
    infer_video,
    query as query_step,
    refine as refine_step,
    remap_labels as remap_labels_step,
    sam3 as sam3_step,
    stats as stats_step,
)

logger = get_logger(__name__)

app = typer.Typer(help="mbariml: detection -> embedding -> clustering -> curation pipeline for imagery and video.")

# Import: `mbariml import {yolo,voc}` brings an existing labeled dataset in
# as ordinary rows (ROI crops and all), the reverse of the matching `export`
# subcommands. Registered first, as the first stage. See mbariml.import_common.
import_app = typer.Typer(help="Import an existing labeled dataset (YOLO or Pascal VOC) into a curation database.")
import_app.command("yolo")(import_yolo.import_yolo)
import_app.command("voc")(import_voc.import_voc)
app.add_typer(import_app, name="import")

# Generate: `mbariml infer {images,video}`. These were three separate commands
# before v0.11.0 -- `detect` and `infer-images` did the same job with
# different defaults (see infer_images.py's docstring for the merge), and
# video had no home at all.
infer_app = typer.Typer(help="Run a model over imagery or video, storing detections + ROI crops in a database.")
infer_app.command("images")(infer_images.infer_images)
infer_app.command("video")(infer_video.infer_video)
app.add_typer(infer_app, name="infer")

# Enrich / Curate / Export.
app.command("embed")(embed_step.embed)
app.command("cluster")(cluster_step.cluster)
app.command("refine")(refine_step.refine)
app.command("remap-labels")(remap_labels_step.remap_labels)
app.command("query")(query_step.query)

# `mbariml export {voc,yolo,id,html,stats}` -- one downstream annotation
# format/gallery/summary per subcommand. voc/yolo also each write their own
# image_manifest.csv/copy_images.py so the matching source images can be
# pulled later (see mbariml.export_common).
export_app = typer.Typer(help="Export curated labels to a downstream annotation format, an HTML gallery, or summary stats.")
export_app.command("voc")(export_voc.export_voc)
export_app.command("yolo")(export_yolo.export_yolo)
export_app.command("id")(export_ids.export_ids)
export_app.command("html")(export_html.generate_html)
# `stats` lives here, not at the top level: it answers "what is in this
# database" using the very same selection rule the annotation exports use
# (mbariml.db.curated_where), so its numbers are a preview of what they will
# write rather than a separate kind of output. It was briefly registered in
# both places; one name only, so there is no question which is canonical.
export_app.command("stats")(stats_step.stats)
app.add_typer(export_app, name="export")


@app.command("review")
def review(
    database_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    label: str = typer.Option(None, help="Filter ROIs by label"),
    page_size: int = typer.Option(500, help="Number of ROIs to show per page."),
    sam3_model: str = typer.Option(
        None,
        "--sam3-model",
        help="Path to SAM3 weights (sam3.pt) to enable 'Add ROI with SAM3' and 'Tighten Boxes'. "
        "Optional; defaults to $MBARIML_SAM3_MODEL, else the copy `mbariml sam3 download` fetched. "
        "Also needs Ultralytics' CLIP: pip install -e \".\\[sam3]\". `mbariml sam3 check` says what's missing.",
    ),
) -> None:
    """Launch the interactive ROI review/labeling GUI."""
    # Imported lazily: PySide6 is only needed for this one interactive command.
    from mbariml.steps.review import review as _review

    _review(database_path, label, page_size, sam3_model)


# Setup, not a stage: `mbariml sam3 {download,check}` fetches and checks the
# optional SAM3 model review can use. See mbariml.gui.sam3_service.
sam3_app = typer.Typer(help="Download and check the optional SAM3 model the review GUI can use.")
sam3_app.command("download")(sam3_step.download)
sam3_app.command("check")(sam3_step.check)
app.add_typer(sam3_app, name="sam3")


# The scriptable chain, in order. "Steps", not "stages": the stages are the
# pipeline's five (Import, Generate, Enrich, Curate, Export -- see the module
# docstring), and these chain steps cut across them. Named rather than
# numbered so adding or merging a command never renumbers anything again.
_CHAIN_STEPS = ["ingest", "embed", "cluster", "export"]


@app.command("run")
def run(
    model_path: str = typer.Argument(..., help="Path to the YOLO model (used by the ingest step)."),
    input_path: str = typer.Argument(..., help="Directory of images, or a video file/directory (see --media)."),
    output_dir: str = typer.Argument(..., help="Directory for the database and all step outputs."),
    media: str = typer.Option("images", help="What INPUT_PATH holds: 'images' or 'video'."),
    from_step: str = typer.Option(
        "ingest", "--from", help=f"First step to run: one of {_CHAIN_STEPS}."
    ),
    to_step: str = typer.Option(
        "export", "--to", help=f"Last step to run (inclusive): one of {_CHAIN_STEPS}."
    ),
    limit: Optional[int] = typer.Option(None, help="Limit passed through to steps that support it, for a quick test run."),
    random_sample: bool = typer.Option(
        False, "--random/--no-random", help="With --limit, sample images randomly at ingest instead of taking the first N in sorted order."
    ),
    seed: Optional[int] = typer.Option(
        None, help="Random seed, shared by --random image sampling and EVoC clustering, for a reproducible run."
    ),
    approx_n_clusters: Optional[int] = typer.Option(18, help="Passed through to cluster; see `mbariml cluster --help`."),
    noise_level: float = typer.Option(0.2, help="Passed through to cluster; see `mbariml cluster --help`."),
) -> None:
    """Run the scriptable chain (ingest -> embed -> cluster -> export: voc + html).

    Every step reads/writes OUTPUT_DIR/yolo_predictions.duckdb, so re-running
    with --from past ingest resumes against whatever is already in that
    database -- nothing upstream is re-run or overwritten.
    """
    if from_step not in _CHAIN_STEPS or to_step not in _CHAIN_STEPS:
        raise typer.BadParameter(f"--from/--to must each be one of {_CHAIN_STEPS}")
    if _CHAIN_STEPS.index(from_step) > _CHAIN_STEPS.index(to_step):
        raise typer.BadParameter("--from must come at or before --to")
    if media not in ("images", "video"):
        raise typer.BadParameter("--media must be 'images' or 'video'")

    output_dir_path = Path(output_dir)
    db_path = output_dir_path / "yolo_predictions.duckdb"

    begin, end = _CHAIN_STEPS.index(from_step), _CHAIN_STEPS.index(to_step)
    steps = _CHAIN_STEPS[begin : end + 1]
    logger.info("Running steps: %s", " -> ".join(steps))

    # NOTE: these step functions are decorated with @app.command(), so their
    # non-required parameters default to raw typer.Option(...)/typer.Argument(...)
    # sentinel objects -- those only get resolved to real values when Click
    # parses a CLI invocation. Calling the functions directly (as we do here)
    # means every parameter besides positional required ones must be passed
    # explicitly with a concrete value, or the sentinel object itself would
    # leak through as e.g. `conf=<typer.models.OptionInfo ...>` and break the
    # underlying library call.
    for step in steps:
        logger.info("=== %s ===", step)
        if step == "ingest" and media == "images":
            infer_images.infer_images(
                model_path, input_path, str(output_dir_path), preset="curate",
                limit=limit, random_sample=random_sample, seed=seed,
                batch_size=50, conf=None, iou=None, max_det=500, imgsz=None,
                device="auto", save_annotated=None, export_csv=False,
            )
        elif step == "ingest":
            infer_video.infer_video(
                model_path, input_path, str(output_dir_path), mode="track",
                stride=infer_video.DEFAULT_STRIDE, tracker=infer_video.DEFAULT_TRACKER,
                track_roi=infer_video.DEFAULT_TRACK_ROI_POLICY, min_track_length=1,
                preset="curate", limit=limit, batch_size=16,
                conf=None, iou=None, max_det=500, imgsz=None, device="auto",
            )
        elif step == "embed":
            embed_step.embed(str(db_path), limit=limit, batch_size=32, decode_workers=0, flush_size=2000, force=False)
        elif step == "cluster":
            # Only the two most impactful EVoC knobs are exposed here for
            # convenience; base_min_cluster_size/n_neighbors/min_samples keep
            # cluster's own tuned defaults -- run `mbariml cluster` directly
            # for full control over every option.
            cluster_step.cluster(
                str(db_path), limit=limit, off=False,
                approx_n_clusters=approx_n_clusters, noise_level=noise_level,
                base_min_cluster_size=2, n_neighbors=40, min_samples=5, seed=seed,
                naming="auto", label_source="original",
            )
        elif step == "export":
            export_voc.export_voc(str(db_path), str(output_dir_path))
            export_html.generate_html(str(db_path), str(output_dir_path / "html"), items_per_page=250)

    logger.info("Pipeline run complete: %s", db_path)


if __name__ == "__main__":
    app()
