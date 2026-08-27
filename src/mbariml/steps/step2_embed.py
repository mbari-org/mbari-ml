"""Step 2: generate embeddings for the ROI blobs stored by step 1.

Uses DINOv3 (ViT-Large/16, general-purpose ``lvd1689m`` weights) -- swapped
in from DINOv2 since it was found to be substantially more accurate in
testing. Preprocessing is built from ``timm.data.resolve_data_config`` for
whatever model is actually loaded, instead of hand-picked constants: the
previous code resized to a fixed 518x518 and normalized with CLIP's mean/std
regardless of which backbone was loaded, which happened to be harmless for
that specific DINOv2 tag (turned out not to use a random classifier head --
verified separately) but was fragile and would have quietly produced wrong
embeddings for a model with different expected preprocessing.

Bug fixed here (this was why a real run took an hour to reach 12%): ROIs
used to be embedded one at a time -- decode, preprocess, transfer to the
GPU, forward pass, transfer back, single-row DB UPDATE, repeat. Each of
those round trips has fixed overhead (Python-level dispatch, an MPS/CUDA
kernel launch, a host<->device transfer) that dominates when the batch size
is 1: the GPU spends most of its time idle waiting for the next single-image
dispatch instead of doing throughput work. ROIs are now decoded and
preprocessed into a single stacked batch tensor, run through the model in
one forward pass per batch, and written back with one executemany per batch
-- the GPU actually stays busy. Use --batch-size to tune it (larger uses
more device memory; shrink it if you hit an out-of-memory error).

Second bottleneck fixed here: decoding + preprocessing each ROI (JPEG
decode, color convert, PIL resize/normalize) is CPU-bound work, but it was
done in a single-threaded Python loop -- on a machine with dozens of CPU
cores, that pins one core while the rest sit idle and the GPU waits on it.
It's now farmed out across a thread pool (--decode-workers, default up to
16 cores): cv2 and PIL release the GIL for the bulk of their C-level work,
so this genuinely parallelizes across cores instead of just adding Python
overhead. Only the model's actual forward pass runs on the main thread
(GPU work must not be parallelized across threads like this).

Third, and biggest in practice, bottleneck fixed here: writing each batch's
results with its own ``UPDATE ... WHERE id = ?`` executemany call. Measured
directly (see the version-history discussion this fix came from): model
throughput on this pipeline is a rock-stable ~80 items/sec on real hardware,
but per-batch DB write time grew from ~1s to ~15s over just 45 batches on a
run that never got faster again -- DuckDB is a columnar/OLAP engine and is
*documented* to be dramatically slower at many small row-by-row UPDATEs
(each one carries MVCC row-versioning overhead) than at doing the same work
as one bulk UPDATE. New embeddings are now staged into a temp table and
applied with a single ``UPDATE ... FROM`` every ``--flush-size`` rows
(default 2000) instead of once per batch -- this was the actual root cause
of "embeddings are taking forever", not the model or the GPU at all.

The original script also loaded the (large) embedding model at *import
time*, as a module-level side effect. That meant simply importing the file
-- e.g. to register it as a CLI subcommand -- downloaded/loaded a model
whether or not you were actually about to use it. It's now loaded lazily,
once, on first use.

IMPORTANT: embeddings from different backbones are not comparable. If you
switch ``EMBEDDING_MODEL_NAME`` (or re-run this after a previous run used a
different model), re-embed the *whole* database with ``--force`` -- do not
let a database end up with a mix of embeddings from two different models,
since clustering/similarity would silently compare vectors from different
spaces as if they were the same.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import typer
from tqdm import tqdm

from mbariml import db
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Generate embeddings for ROIs stored as blobs in the database.")
logger = get_logger(__name__)

EMBEDDING_MODEL_NAME = "vit_large_patch16_dinov3.lvd1689m"


@lru_cache(maxsize=1)
def _load_embedding_model():
    import timm
    import torch
    from timm.data import create_transform, resolve_data_config

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS backend for embeddings.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA backend for embeddings.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU backend for embeddings (no GPU available) -- this will be slow.")

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = timm.create_model(EMBEDDING_MODEL_NAME, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)

    # Build the exact preprocessing this model was trained with, rather than
    # hand-picked constants that only happen to be right for one model.
    preprocess = create_transform(**resolve_data_config({}, model=model), is_training=False)
    return model, preprocess, device


def _decode_roi(roi_blob: bytes) -> np.ndarray:
    roi_array = np.frombuffer(roi_blob, dtype=np.uint8)
    roi = cv2.imdecode(roi_array, cv2.IMREAD_COLOR)
    if roi is None:
        raise ValueError("cv2.imdecode returned None -- corrupt or unsupported ROI blob")
    return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)


def _decode_and_preprocess_safe(roi_blob: bytes):
    """Decode one ROI and run it through the model's preprocessing transform,
    catching (rather than raising) any failure so one bad blob can't blow up
    the whole batch via executor.map. Pure CPU work with no GPU/model-state
    access besides the (already-loaded, cached) transform object -- safe to
    run from a worker thread. Returns (tensor, None) or (None, exception)."""
    from PIL import Image

    try:
        _, preprocess, _ = _load_embedding_model()
        return preprocess(Image.fromarray(_decode_roi(roi_blob))), None
    except Exception as exc:
        return None, exc


def _prepare_batch(chunk: list[tuple], executor: ThreadPoolExecutor) -> tuple[list, list, int]:
    """Decode + preprocess a batch's ROI blobs in parallel across `executor`'s
    threads. Returns (tensors, ids, num_failed), with per-item failures
    logged and excluded rather than aborting the whole batch."""
    ids = [row[0] for row in chunk]
    blobs = [row[1] for row in chunk]

    tensors = []
    ok_ids = []
    failed = 0
    for roi_id, (tensor, exc) in zip(ids, executor.map(_decode_and_preprocess_safe, blobs)):
        if exc is not None:
            failed += 1
            logger.error("Error decoding/preprocessing ROI id=%s; leaving embedding NULL", roi_id, exc_info=exc)
            continue
        tensors.append(tensor)
        ok_ids.append(roi_id)
    return tensors, ok_ids, failed


def _run_forward_pass(tensors: list) -> np.ndarray:
    """Run one forward pass over an already-preprocessed batch. Must be
    called from the main thread only -- GPU/MPS work isn't thread-parallel
    like the CPU preprocessing above."""
    import torch

    model, _, device = _load_embedding_model()
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        return model(batch).cpu().numpy()


def _flush_pending(conn, pending: list[tuple]) -> None:
    """Apply accumulated (id, embedding) pairs with ONE bulk UPDATE instead
    of one UPDATE per row/batch -- see the module docstring for why this
    matters (a lot) on DuckDB specifically."""
    db.bulk_update(conn, "predictions", "id", "INTEGER", {"embedding": "FLOAT[]"}, pending)


@app.command()
def embed(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database (from step 1)."),
    limit: Optional[int] = typer.Option(None, help="Limit the number of ROIs processed, for testing."),
    batch_size: int = typer.Option(
        32, help="Number of ROIs embedded per forward pass. Bigger is faster on a GPU/MPS "
        "up to a point; shrink it if you hit an out-of-memory error."
    ),
    decode_workers: int = typer.Option(
        0, help="Threads used to decode/preprocess ROIs in parallel (CPU-bound work, "
        "independent of the GPU forward pass). 0 (default) auto-picks min(16, cpu_count)."
    ),
    flush_size: int = typer.Option(
        2000, help="Apply this many newly-computed embeddings to the database per bulk UPDATE. "
        "Larger means fewer, bigger writes (much faster on DuckDB); smaller means less work "
        "lost if the run is interrupted between flushes."
    ),
    force: bool = typer.Option(
        False, help="Recompute embeddings even for ROIs that already have one. "
        "Required after changing EMBEDDING_MODEL_NAME -- never mix embeddings from two different models in one database."
    ),
) -> None:
    """Generate embeddings for ROIs stored as blobs in the database and update the database."""
    if decode_workers <= 0:
        decode_workers = min(16, os.cpu_count() or 4)

    rows: list[tuple]
    with db.connect(db_path) as conn:
        query = "SELECT id, roi FROM predictions" if force else "SELECT id, roi FROM predictions WHERE embedding IS NULL"
        query += " ORDER BY id"  # sorted writes are much faster on DuckDB than unsorted ones
        rows = conn.execute(query).fetchall()
        if limit:
            rows = rows[:limit]

        if not rows:
            logger.info("No ROIs without an embedding found in %s; nothing to do (use --force to recompute anyway).", db_path)
            return

        logger.info(
            "Embedding %d ROI(s) from %s using %s (batch size %d, %d decode workers, flush every %d)",
            len(rows), db_path, EMBEDDING_MODEL_NAME, batch_size, decode_workers, flush_size,
        )
        _load_embedding_model()  # load once on the main thread before any worker touches the cache

        succeeded = 0
        failed = 0
        pending: list[tuple] = []
        with ThreadPoolExecutor(max_workers=decode_workers) as executor:
            with tqdm(total=len(rows), desc="Generating embeddings") as progress:
                for i in range(0, len(rows), batch_size):
                    chunk = rows[i:i + batch_size]
                    tensors, ok_ids, n_failed = _prepare_batch(chunk, executor)
                    failed += n_failed

                    if tensors:
                        try:
                            embeddings = _run_forward_pass(tensors)
                            pending.extend((ok_ids[j], embeddings[j].tolist()) for j in range(len(ok_ids)))
                            succeeded += len(ok_ids)
                        except Exception:
                            failed += len(ok_ids)
                            logger.exception("Error embedding batch starting at ROI id=%s; leaving embeddings NULL", ok_ids[0])

                    progress.update(len(chunk))

                    if len(pending) >= flush_size:
                        _flush_pending(conn, pending)
                        pending = []

            _flush_pending(conn, pending)  # apply whatever's left after the last full flush

    logger.info("Done: %d embedded, %d failed", succeeded, failed)
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
