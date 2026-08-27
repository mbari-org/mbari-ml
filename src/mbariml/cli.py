"""Unified CLI for the mbariml pipeline.

Every step is available as its own subcommand (``mbariml detect``,
``mbariml infer-images``, etc.) so you can run -- or re-run -- any single
step on its own, pointed at whatever database/directory you already have.
That's the main way to "start at any step": e.g. to run inference on a new
set of images (step 9) you don't need to touch anything upstream:

    mbariml infer-images runs/best.pt /data/new_survey/ /data/new_survey_results/

``detect``/``embed``/``cluster``/``refine`` build curated *training* data;
``infer-images`` (and, eventually, a video+tracking counterpart) instead
runs an already-trained model over new data -- a different job, hence the
separate name rather than folding it into ``detect``.

``mbariml run`` additionally chains the scriptable steps of the curation
pipeline (1 detect -> 2 embed -> 3 cluster -> 6 export voc -> 7 html) for
convenience, and can start or stop anywhere in that chain via
``--from-step``/``--to-step``. Step 5 (the interactive review GUI) and step 8
(ad hoc queries) are not part of that chain since they aren't scriptable/
batch operations; step 9 (inference) and step 10 (label remapping) are
intentionally excluded from ``run`` too, since they aren't links in the same
database's chain -- run them directly with their own subcommand instead.

``mbariml export`` groups every downstream annotation-format export under
one subcommand (``export voc``, ``export yolo``, ``export id``) rather than
a flat ``export-voc``/``export-ids`` per format (pre-v0.8.0); ``mbariml
stats`` is a separate, read-only command for label counts and per-image
detection stats, not an export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mbariml.logging_utils import get_logger
from mbariml.steps import (
    step1_detect,
    step2_embed,
    step3_cluster_evoc,
    step4_cluster_refine,
    step6_export_voc,
    step7_generate_html,
    step8_query,
    step9_inference,
    step10_remap_labels,
    step_backfill_sharpness,
    step_export_ids,
    step_export_yolo,
    step_stats,
)

logger = get_logger(__name__)

app = typer.Typer(help="mbariml: YOLO detection -> embedding -> clustering -> curation pipeline.")

app.command("detect")(step1_detect.detect)
app.command("embed")(step2_embed.embed)
app.command("cluster")(step3_cluster_evoc.cluster)
app.command("refine")(step4_cluster_refine.refine)
app.command("html")(step7_generate_html.generate_html)
app.command("query")(step8_query.query)
app.command("infer-images")(step9_inference.infer)
app.command("remap-labels")(step10_remap_labels.remap_labels)
app.command("backfill-sharpness")(step_backfill_sharpness.backfill_sharpness)
app.command("stats")(step_stats.stats)

# `mbariml export {voc,yolo,id}` -- one downstream annotation format per
# subcommand, each writing its own image_manifest.csv/copy_images.py (voc,
# yolo) so the matching source images can be pulled later (see
# mbariml.export_common). Replaces the old flat `export-voc`/`export-ids`
# commands (v0.8.0) now that there's a third format (yolo) to group with them.
export_app = typer.Typer(help="Export curated labels to a specific downstream annotation format.")
export_app.command("voc")(step6_export_voc.export_voc)
export_app.command("yolo")(step_export_yolo.export_yolo)
export_app.command("id")(step_export_ids.export_ids)
app.add_typer(export_app, name="export")


@app.command("review")
def review(
    database_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    label: str = typer.Option(None, help="Filter ROIs by label"),
    page_size: int = typer.Option(500, help="Number of ROIs to show per page."),
) -> None:
    """Launch the interactive ROI review/labeling GUI (step 5)."""
    # Imported lazily: PySide6 is only needed for this one interactive command.
    from mbariml.steps.step5_roi_editor_gui import review as _review

    _review(database_path, label, page_size)


# The scriptable curation chain, in order. Each entry is (step number, label).
_CHAIN_STEPS = [(1, "detect"), (2, "embed"), (3, "cluster"), (6, "export voc"), (7, "html")]


@app.command("run")
def run(
    model_path: str = typer.Argument(..., help="Path to the YOLO model (used by the detect step)."),
    image_dir: str = typer.Argument(..., help="Directory containing input images."),
    output_dir: str = typer.Argument(..., help="Directory for the database and all step outputs."),
    from_step: int = typer.Option(1, help="First step to run: one of 1, 2, 3, 6, 7."),
    to_step: int = typer.Option(7, help="Last step to run (inclusive): one of 1, 2, 3, 6, 7."),
    limit: Optional[int] = typer.Option(None, help="Limit passed through to steps that support it, for a quick test run."),
    random_sample: bool = typer.Option(
        False, "--random/--no-random", help="With --limit, sample images randomly (step 1) instead of taking the first N in sorted order."
    ),
    seed: Optional[int] = typer.Option(
        None, help="Random seed, shared by --random image sampling (step 1) and EVoC clustering (step 3), for a reproducible run."
    ),
    approx_n_clusters: Optional[int] = typer.Option(18, help="Passed through to step 3; see `mbariml cluster --help`."),
    noise_level: float = typer.Option(0.2, help="Passed through to step 3; see `mbariml cluster --help`."),
) -> None:
    """Run the scriptable curation chain (detect -> embed -> cluster -> export voc -> html).

    Every step reads/writes OUTPUT_DIR/yolo_predictions.duckdb, so re-running
    with --from-step > 1 resumes against whatever is already in that database
    -- nothing upstream is re-run or overwritten.
    """
    valid_steps = {n for n, _ in _CHAIN_STEPS}
    if from_step not in valid_steps or to_step not in valid_steps:
        raise typer.BadParameter(f"--from-step/--to-step must each be one of {sorted(valid_steps)}")
    if from_step > to_step:
        raise typer.BadParameter("--from-step must be <= --to-step")

    output_dir_path = Path(output_dir)
    db_path = output_dir_path / "yolo_predictions.duckdb"

    steps_to_run = [(n, name) for n, name in _CHAIN_STEPS if from_step <= n <= to_step]
    logger.info("Running steps: %s", ", ".join(f"{n}:{name}" for n, name in steps_to_run))

    # NOTE: these step functions are decorated with @app.command(), so their
    # non-required parameters default to raw typer.Option(...)/typer.Argument(...)
    # sentinel objects -- those only get resolved to real values when Click
    # parses a CLI invocation. Calling the functions directly (as we do here)
    # means every parameter besides positional required ones must be passed
    # explicitly with a concrete value, or the sentinel object itself would
    # leak through as e.g. `conf=<typer.models.OptionInfo ...>` and break the
    # underlying library call.
    for step_num, name in steps_to_run:
        logger.info("=== Step %d: %s ===", step_num, name)
        if name == "detect":
            step1_detect.detect(
                model_path, image_dir, str(output_dir_path), limit=limit,
                random_sample=random_sample, seed=seed,
                conf=0.005, iou=0.05, max_det=500, imgsz=1952, device="auto",
            )
        elif name == "embed":
            step2_embed.embed(str(db_path), limit=limit, batch_size=32, decode_workers=0, flush_size=2000, force=False)
        elif name == "cluster":
            # Only the two most impactful EVoC knobs are exposed here for
            # convenience; base_min_cluster_size/n_neighbors/min_samples keep
            # step 3's own tuned defaults -- run `mbariml cluster` directly
            # for full control over every option.
            step3_cluster_evoc.cluster(
                str(db_path), limit=limit, off=False,
                approx_n_clusters=approx_n_clusters, noise_level=noise_level,
                base_min_cluster_size=2, n_neighbors=40, min_samples=5, seed=seed,
            )
        elif name == "export voc":
            step6_export_voc.export_voc(str(db_path), str(output_dir_path))
        elif name == "html":
            step7_generate_html.generate_html(str(db_path), str(output_dir_path / "html"), items_per_page=250)

    logger.info("Pipeline run complete: %s", db_path)


if __name__ == "__main__":
    app()
