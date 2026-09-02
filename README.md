# mbari-ml pipeline

A chain of steps that go from raw survey imagery to a curated, labeled
DuckDB database: **detect → embed → cluster → refine → review → export
(voc/yolo/id/html) → query → remap**, plus a standalone **infer-images**
step for running an already-trained model on a new batch of images, and a
**stats** step for label counts and per-image detection stats.

## Setup

```bash
pip install -e .
```

This installs the `mbariml` command. (`requirements.txt` is also kept, fixed,
for anyone who just wants `pip install -r requirements.txt` without an
editable install — see "What changed" below for why it needed fixing.)

## Running a single step ("start at any step")

Every step reads/writes the **same database schema** and just operates on
whatever database you point it at — there's no hidden state, and no step
needs any specific earlier step to have run, only a database that already
has what that step needs (e.g. `embed` needs ROI blobs, `cluster` needs
embeddings). In particular, **step 8 (inference on a new set of images) is a
standalone entry point** — it doesn't need any earlier step's output to run:

```bash
mbariml infer-images runs/train/best.pt /data/new_survey/ /data/new_survey_results/
```

...and because it writes the same schema as every other step, you can
immediately continue from its output into anything downstream — review it,
embed it, cluster it, export it — without ever running `detect`:

```bash
mbariml review /data/new_survey_results/yolo_predictions.duckdb
mbariml export html /data/new_survey_results/yolo_predictions.duckdb /data/new_survey_results/html
mbariml embed /data/new_survey_results/yolo_predictions.duckdb
```

(This wasn't always true — see "What changed" below; step 8 used to write a
separate, lighter schema that none of those commands could read.)

All subcommands:

| Command | Step | What it does |
|---|---|---|
| `mbariml detect`        | 1  | YOLO-detect + extract ROI crops into a fresh curation database |
| `mbariml embed`         | 2  | Compute an embedding for every ROI |
| `mbariml cluster`       | 3  | Cluster embeddings with EVoC, tag dominant labels, export review grids |
| `mbariml refine`        | 4  | Re-cluster one label's ROIs into finer sub-clusters |
| `mbariml review`        | 5  | Interactive GUI for labeling/deleting ROIs |
| `mbariml export voc`    | 6  | Export curated labels as Pascal VOC XML (see below) |
| `mbariml export yolo`   | 6  | Export curated labels as YOLO-format label files + names.txt (see below) |
| `mbariml export id`     | 6  | Write a `*.id` sidecar file next to each source image with a curated identification (see below) |
| `mbariml export html`   | 6  | Paginated HTML gallery of images + crops |
| `mbariml query`         | 7  | Run ad hoc SQL against a database |
| `mbariml infer-images`  | 8  | Run a trained model over new images (standalone) |
| `mbariml remap-labels`  | 9  | Bulk-rename `new_label` values from a CSV |
| *(reserved)*            | 10 | `infer-videos` -- video+tracking counterpart to `infer-images`; not built yet |
| `mbariml stats`         | 11 | Label counts, boxes-per-image stats, and an optional image × label count matrix CSV (see below) |

`mbariml export {voc,yolo,id,html}` groups every downstream annotation
format *and* the HTML gallery under one subcommand, as step 6 (before
v0.8.0 these were flat `export-voc`/`export-ids` commands with no YOLO
format at all; `html` was its own separate step 7 before v0.9.0 folded it
in here instead). `backfill-sharpness` (a one-time migration utility for
databases created before sharpness scoring existed) was removed in v0.9.0 —
every database has gotten a real sharpness score at write time (`detect`/
`infer-images`) since v0.7.0.

Note: `export voc`/`export yolo`/`export html` don't take an `image_dir`
argument (removed in v0.3.0, before the `export` group existed) — images are
located via the path recorded at detection time, which also fixes a real bug
(see "What changed" below).

Run `mbariml <command> --help` for each step's full option list (`mbariml
export --help` for the export group itself).

### Using the review GUI (step 5) well

`mbariml review DB_PATH` labels/deletes ROIs one page at a time:

- **Click** a thumbnail to select it, **Shift-click** to select a range,
  **Ctrl-click** to add to the selection.
- Type a label into the relabel field (free text, or pick from the dropdown
  of labels already in use) and press **Enter**, or click **Label**, to
  apply it to the current selection.
- **Delete** removes the current selection (asks for confirmation first —
  it's permanent).
- **Escape** clears the selection.
- **Right-click** a thumbnail to sort *every page* by embedding similarity
  to that ROI (most similar first), across the whole database (respecting
  the current `--label` filter, if any). Requires `mbariml embed` to have run
  first. Changing the Sort dropdown exits similarity mode and returns to
  normal sorting.
- The status line under the buttons always shows the current page, how many
  ROIs are shown, and how many are selected, so a keypress never surprises you.
- **Add New ROI** (green button, top of the controls panel): for a detection
  YOLO missed entirely. Click it, drag a box on the full-image panel, type a
  label in the prompt that pops up, and repeat -- it stays armed for
  drawing as many boxes as you need, on the currently shown image or any
  other you select next, until you click the button again or press
  **Escape**. Each finished box is inserted immediately (confidence fixed at
  `1.0`, `verified` set -- a human drew and named it, there's nothing left
  to review) and shows up in the detail view right away; the grid picks it
  up shortly after via the normal page refresh. Its embedding is computed
  right after, in the background (the status line says so), using the exact
  same DINOv3 model/preprocessing `mbariml embed` uses -- directly
  comparable to every other embedding in the database, so a new box is
  immediately usable by similarity search/clustering, not stuck with
  `embedding IS NULL` until someone remembers to run `mbariml embed`
  (if that background step ever fails, the ROI itself is still saved and
  `mbariml embed` remains a safe fallback -- it only ever fills in rows
  that are still NULL). An empty/cancelled label prompt discards that box --
  nothing is written. Wheel-zoom, dragging an *existing* box, and everything
  else on the panel work exactly as before; only a drag on empty image
  background behaves differently while armed.

Labeling and deleting no longer rebuild the entire page of thumbnails (the
original did, on every single click, which is why review used to feel slow) —
only the affected thumbnails' captions update immediately. Sorting and
paging still do a full rebuild, since the visible set of ROIs actually
changes then.

**Sort by Sharpness now actually means something.** It used to sort by a
column that was hardcoded to `0.0` for every row -- a no-op. `mbariml
detect`/`infer-images` now compute a real blur score (Laplacian variance)
per ROI as they extract it, so every database built since v0.7.0 has real
values to sort by. (A `backfill-sharpness` utility used to exist for
databases built *before* that change, recomputing the score after the fact
from each row's already-stored ROI crop; it was removed in v0.9.0 once every
current write path covered it at the source. If you're still holding a
pre-v0.7.0 database that needs backfilling, ask for that utility back rather
than assuming a migration path still exists.)

Sorting by sharpness (ascending) then surfaces the blurriest ROIs first —
useful for quickly finding and deleting unusable crops.

### Embeddings (DINOv3)

`mbariml embed` uses DINOv3 (ViT-Large/16, `vit_large_patch16_dinov3.lvd1689m`
general-purpose weights) — swapped in from DINOv2 for accuracy. Embeddings
from different backbones are **not comparable**: if you ever change
`EMBEDDING_MODEL_NAME` in `src/mbariml/steps/step2_embed.py`, or a database
already has embeddings from a different model, re-embed the whole thing with
`--force` so nothing ends up mixing two different embedding spaces (which
would silently corrupt both clustering and the GUI's similarity sort):

```bash
mbariml embed /data/survey_results/yolo_predictions.duckdb --force
```

**Embedding is batched** (`--batch-size`, default 32) — ROIs are decoded and
run through the model together in one forward pass per batch, instead of one
at a time. One-at-a-time was the original approach and made a GPU/MPS sit
mostly idle waiting on serialized dispatches instead of doing throughput
work. Increase `--batch-size` for more throughput up to your device's
memory limit; decrease it if you hit an out-of-memory error.

**Decoding/preprocessing is parallelized** (`--decode-workers`, default
`min(16, cpu_count)`) — this CPU-bound work (JPEG decode, resize, normalize)
used to run in a single-threaded Python loop, pinning one core while dozens
sat idle on a many-core machine and leaving the GPU waiting on it.

**Database writes are batched** (`--flush-size`, default 2000) — this was
the *dominant* real bottleneck, much bigger than either point above. New
embeddings used to be written with one `UPDATE ... WHERE id = ?` per batch.
Measured directly on real hardware: model throughput was a rock-stable ~80
items/sec, but per-batch DB write time grew from ~1s to ~15s over just 45
batches and kept climbing — DuckDB is a columnar/OLAP engine and is
[documented](https://github.com/duckdb/duckdb/discussions/3492) to be
dramatically slower at many small row-by-row UPDATEs (each carries MVCC
row-versioning overhead) than at one bulk UPDATE. New embeddings are now
staged into a temp table and applied with a single `UPDATE ... FROM` every
`--flush-size` rows instead of once per batch. Verified end-to-end: a run
that degraded from 45 it/s to 6.5 it/s (and was still falling) over 3000
ROIs became a flat ~70-75 it/s for the same 3000 ROIs after this fix — no
degradation at all, ~6x faster overall on top of removing an actively
worsening trend that would have made a large run take dramatically longer
than a naive per-item estimate suggests.

It's safe to interrupt and resume any time: `mbariml embed` only processes
rows with `embedding IS NULL`, so re-running after a `Ctrl-C` (or a crash)
picks up right where the last completed flush left off.

If a long embedding run is still far slower than expected after all of the
above, with a confirmed MPS/CUDA device, rule out something else competing
for the GPU or unified memory (check Activity Monitor / `nvidia-smi` while
it runs) before assuming it's this code.

**On MLX**: [`mlx-image`](https://github.com/riccardomusmeci/mlx-image) does
have a DINOv3 implementation for Apple Silicon (weights converted from the
same timm/HuggingFace source, hosted at
[huggingface.co/mlx-vision](https://huggingface.co/mlx-vision)), so it's a
real option if you want to explore it further. It wasn't adopted here: no
published benchmark showed it meaningfully outperforming a well-optimized
PyTorch/MPS pipeline for this kind of batched-inference workload (MLX's
biggest advantages are for autoregressive/LLM-style workloads, not a single
forward pass per batch), and — as the numbers above show — the actual
bottleneck was never the model or the framework at all. Porting to MLX would
not have fixed a DuckDB write pattern.

### Clustering (step 3) -- and why DuckDB VSS/ANN wouldn't help

`mbariml cluster` used to be able to take *hours* on a large database, with
no visible progress. Directly measured cause: **not** the clustering math.
`evoc.EVoC.fit_predict()` isn't brute-force KNN -- it has its own
JIT-compiled approximate nearest-neighbor search built in already (similar
in spirit to DuckDB's VSS/HNSW extension), and clusters 50,000 embeddings in
~2.6 seconds, scaling roughly linearly (300k+ points finishes in well under
a minute). Adding an ANN index wouldn't touch this cost at all -- there's no
repeated similarity-search query here for an index to accelerate, just one
batch clustering call that's already fast.

The actual cost was the same DuckDB small-UPDATE problem as `embed` (see
above), except worse: writing cluster results touches the *indexed*
`new_label` column. Measured directly: 100,000 rows via one `UPDATE` per
row took 38.6 seconds and was still trending worse; the same 100,000 rows
via a staged bulk `UPDATE ... FROM` (`mbariml.db.bulk_update`) took 6.0
seconds. `cluster` now uses this, logs progress at each phase (fetching,
building the embedding matrix, clustering, writing results) so a long run
is never silent, and step 4 (`refine`) got the identical fix for the same
reason.

**A second, even more fundamental finding while chasing this down**:
DuckDB's Python driver commits (and fsyncs to disk) after *every individual
statement* by default when writing to a file-backed database -- even
within a single `executemany()` call. Measured directly: an identical
2000-row `INSERT` took **11.46 seconds** without an explicit transaction
around it, and **0.82 seconds** wrapped in one (`BEGIN TRANSACTION` /
`COMMIT`) -- a ~14x difference from transaction-wrapping alone, independent
of row count or which column is touched. This affected every step that
writes many rows directly to the persistent `predictions` table:
`mbariml.db.fast_executemany` (a drop-in replacement for
`conn.executemany`) now wraps every such call in steps 1, 2, 3, 4, 5, and 8
in an explicit transaction. Steps 1 and 8 already committed incrementally
per image/batch for resumability -- that's unchanged; this fix is about
what happens *inside* each of those commits, not how often they happen.

### Exporting to YOLO format, and pulling the matching images

`mbariml export yolo DB_PATH OUTPUT_DIR` writes curated labels (`new_label`,
excluding `noise`) as YOLO-format label files — one `labels/<name>.txt` per
image, each line `class_id x_center y_center width height` normalized
against that image's actual pixel dimensions — plus `names.txt` (class index
→ label, in the order the label files use):

```bash
mbariml export yolo /data/survey_results/yolo_predictions.duckdb /data/survey_results/yolo_out/
```

It deliberately doesn't copy the source images into an `images/` folder
itself (they already exist on the survey volume this ran against). Instead,
`export yolo` and `export voc` both also write `image_manifest.csv` (every
distinct source image referenced, mapped to a collision-safe destination
filename) and a standalone `copy_images.py` next to it. Run that script
later — from this machine or any other that can see the recorded source
paths — to actually pull the matching images down, e.g. to Desktop or
straight into an `images/` directory next to the label files:

```bash
python3 /data/survey_results/yolo_out/copy_images.py
python3 /data/survey_results/yolo_out/copy_images.py --dest /data/survey_results/yolo_out/images
```

`copy_images.py` is stdlib-only (`argparse`/`csv`/`shutil`/`pathlib`) and
doesn't import `mbariml` — it's meant to be portable, not tied to this repo
being installed wherever it eventually runs.

### Exporting `*.id` identification files

`mbariml export id DB_PATH` writes a `<image_stem>.id` sidecar file next to
every source image that has at least one curated identification (`new_label`
set, excluding `noise`) — wherever that image actually lives on disk, so it
naturally follows a nested mission directory structure:

```bash
mbariml export id /data/survey_results/yolo_predictions.duckdb
```

Each file has a header (generator + version, the user who ran the export,
the model that produced the detections — recorded automatically from
`mbariml detect`, or override with `--model`, the source image, and the
identification count) followed by one line per identification, as a
4-vertex polygon (top-left, top-right, bottom-right, bottom-left) with pixel
coordinates filled in and `lon,lat,depth` left as `0.0` placeholders, meant
to be filled in later by a separate navigation-merge process:

```
# mbariml identification file
# generator: mbariml v0.10.0
# generated_by: lonny
# generated_at: 2026-08-19T17:36:28Z
# model: /path/to/best.pt
# source_image: 1619554491865857.png
# count: 2
#
# index label confidence  vertices(TL,TR,BR,BL as px_x,px_y,lon,lat,depth)
0 Muusoctopus 0.8740  120,45,0.0,0.0,0.0  180,45,0.0,0.0,0.0  180,90,0.0,0.0,0.0  120,90,0.0,0.0,0.0
1 Actiniaria 0.6110  40,200,0.0,0.0,0.0  95,200,0.0,0.0,0.0  95,260,0.0,0.0,0.0  40,260,0.0,0.0,0.0
```

### Label counts and per-image detection stats

`mbariml stats DB_PATH` prints two tables to the console: label counts (with
percent of total) and boxes-per-image summary stats (avg/min/median/max,
across every image with at least one detection). Both use
`COALESCE(new_label, label)` as the effective label — the same per-row
fallback `export html` uses — so this is useful both before curation (nothing but
the raw YOLO `label` yet) and after (`new_label` set). `noise` is included
by default (useful while curating, to see how much of the database is still
noise/unlabeled); pass `--exclude-noise` once you want real-identification
counts only:

```bash
mbariml stats /data/survey_results/yolo_predictions.duckdb
mbariml stats /data/survey_results/yolo_predictions.duckdb --exclude-noise --top 20
```

Pass `--output-dir` to also write `label_by_image_matrix.csv` — an image ×
label count matrix (rows are images, columns are labels, values are box
counts) for granular, per-concept-per-image analysis, e.g. loading straight
into pandas/R for ecological statistics (per-image richness, per-label
frequency-of-occurrence across images, etc.):

```bash
mbariml stats /data/survey_results/yolo_predictions.duckdb --output-dir /data/survey_results/stats/
```

## Running the whole curation chain

```bash
mbariml run best.pt /data/survey_images/ /data/survey_results/
```

This chains detect → embed → cluster → export (voc + html) against
`/data/survey_results/yolo_predictions.duckdb`. Use `--from-step`/`--to-step`
(values from `1, 2, 3, 6`) to run only part of it — e.g. to resume after
already reviewing in the GUI and just re-export:

```bash
mbariml run best.pt /data/survey_images/ /data/survey_results/ --from-step 6 --to-step 6
```

Step 6 always runs both `export voc` and `export html` together when
reached — they were separate steps (6 and 7) before v0.9.0 folded `html`
into `export`; if you only want one of the two, run it directly
(`mbariml export voc ...` or `mbariml export html ...`) instead of via `run`.

Step 5 (interactive review) and step 7 (queries) aren't part of `run` since
they're not batch operations. Step 8 (`infer-images`) and step 9
(`remap-labels`) also aren't, since they don't chain against the same
database — run them directly. `export yolo`/`export id` and `stats` (step
11) aren't part of the chain either — run them directly, any time, against
whatever database `run` (or any individual step) already produced.

## What changed from the original scripts

**"Add New ROI" in the review GUI (v0.10.0)**: previously the only way to
fix a missed detection was to run `detect`/`infer-images` again with a lower
`--conf`, or add it out-of-band and re-import -- there was no way to just
draw the box YOLO should have found. The review GUI (step 5) now has a
green "Add New ROI" button that arms drawing mode on the full-image panel:
drag a box, type its label in the prompt, and repeat for as many as needed
before clicking the button again or pressing Escape. Implemented as
`_DrawableViewBox` in `detail_view.py` (a `pyqtgraph.ViewBox` subclass that
reuses pyqtgraph's own built-in `RectMode` scale-box mechanics for the
rubber-band visual and coordinate math, but reports the finished rect
instead of zooming to it) plus `annotation_service.insert_roi` (new row:
`confidence` fixed at `1.0`, `class_id` left `NULL`, `new_label` and `label`
both set to the typed value, `verified` set immediately -- a human drew and
named it, there's nothing left to review). Its embedding is computed right
after, off the GUI thread (`MainWindow._embed_new_roi_worker`, applied via
`annotation_service.set_embedding` once it finishes), using
`mbariml.steps.step2_embed.embed_roi_bgr` -- a new single-item entry point
into the exact same model/preprocessing `mbariml embed`'s batched pipeline
uses (verified directly: byte-for-byte identical output to running the same
crop through the batched path), so a hand-drawn box's embedding is
guaranteed comparable to every other embedding in the database, not a
second, potentially-drifted implementation. Not adapted from vars-gridview;
new here.

**`export`/step-numbering restructuring (v0.9.0)**: `mbariml html` (step 7)
is now `mbariml export html`, grouped into step 6 alongside `export
voc`/`export yolo`/`export id` -- one step for "produce some downstream
output from curated data" instead of two. Steps after it shifted down by one
to fill the gap: `query` is now step 7, `infer-images` step 8,
`remap-labels` step 9 (a step 10, `infer-videos`, is reserved for a future
video+tracking counterpart to `infer-images` -- not built). `mbariml stats`
is now numbered step 11. `backfill-sharpness` was removed -- see "Using the
review GUI (step 5) well" above for why it's no longer needed. `mbariml
run`'s scriptable chain is now `1 detect -> 2 embed -> 3 cluster -> 6
export` (`--from-step`/`--to-step` values `1, 2, 3, 6`); reaching step 6
still runs both `export voc` and `export html`, same as the old chain
running through the old steps 6 and 7.

**`export`/`stats` restructuring (v0.8.0)**: the flat `export-voc` and
`export-ids` commands are now grouped as `mbariml export voc` and `mbariml
export id`, alongside a new `mbariml export yolo` (YOLO-format label files +
`names.txt`) — one downstream annotation format per subcommand instead of a
flat command per format, now that there are three. `export voc`/`export
yolo` also each write `image_manifest.csv` + a standalone `copy_images.py`
so the matching source images can be pulled down later (e.g. to Desktop)
without walking the mission's own nested directory structure by hand — see
"Exporting to YOLO format, and pulling the matching images" above. New
`mbariml stats` command for label counts, boxes-per-image summary stats, and
an optional image × label count matrix CSV for ecological analysis — see
"Label counts and per-image detection stats" above.

**Clustering (step 3) could take hours with no visible progress, and
DuckDB's Python driver fsyncs per statement by default (v0.7.0 fix)**: see
"Clustering (step 3)" above for the full story -- in short, the clustering
math was never the problem (evoc already has its own fast approximate
nearest-neighbor search; DuckDB VSS/ANN would not have helped), the problem
was the same small-UPDATE pattern as `embed`'s, made worse by touching an
indexed column, fixed with `mbariml.db.bulk_update`. Chasing that down
surfaced a more fundamental issue: DuckDB's Python driver commits (and
fsyncs to disk) after *every statement* by default when writing to a
file-backed database, even inside one `executemany()` call -- measured at a
~14x cost for an identical INSERT with vs. without an explicit transaction
around it. `mbariml.db.fast_executemany` fixes this everywhere the pipeline
writes many rows to the persistent table at once (steps 1, 2, 3, 4, 5, 9).

**Embedding was slow, and getting progressively slower the longer it ran
(v0.5.0/v0.6.0 fixes)**: on a real run against 327,045 ROIs on a Mac Studio
M3 Ultra with a confirmed MPS device, throughput was still crawling after an
hour. Three compounding causes, found by isolating and timing each stage
separately rather than guessing:
1. ROIs were embedded one at a time (decode, preprocess, transfer to GPU,
   forward pass, transfer back, single-row DB write, repeat) -- fixed by
   batching (`--batch-size`, default 32): a whole batch is stacked into one
   tensor and run through the model in a single forward pass.
2. Decode/preprocess (JPEG decode, resize, normalize) is CPU-bound work that
   ran in a single-threaded Python loop, pinning one core while dozens sat
   idle -- fixed by parallelizing it across a thread pool
   (`--decode-workers`, default up to 16 cores).
3. **The actual dominant bottleneck**: each batch's new embeddings were
   written with their own `UPDATE ... WHERE id = ?` executemany call.
   Measured directly: model throughput was a rock-stable ~80 items/sec, but
   per-batch DB write time grew from ~1s to ~15s over just 45 batches and
   kept climbing -- DuckDB is a columnar/OLAP engine, documented to be
   dramatically slower at many small row-by-row UPDATEs (MVCC row-versioning
   overhead) than at one bulk UPDATE. Fixed by staging new embeddings into a
   temp table and applying them with a single `UPDATE ... FROM` every
   `--flush-size` rows (default 2000) instead of once per batch. Verified: a
   run that degraded from 45 it/s to 6.5 it/s (still falling) over 3000 ROIs
   became a flat ~70-75 it/s for the same 3000 ROIs with no degradation at
   all after this fix. An earlier attempt at fixing the degradation with
   periodic `torch.mps.empty_cache()` calls was tested and found to have no
   effect (the cause was never GPU memory) -- removed once the real cause
   was isolated. See "Embeddings (DINOv3)" above for the full numbers,
   including why an MLX port was investigated and not adopted.

**Step 8's database used a different schema than every other step**, missing
`roi_index`, `roi` (the actual crop blob), `embedding`, and `new_label`.
That meant `mbariml review`, `cluster`, `refine`, `export voc`, and
`remap-labels` would all fail outright against step 8's output -- only the
HTML gallery step and `query` happened to work. Fixed by having step 8 write
the same curation schema as every other step (cropping each ROI directly
from Ultralytics' already-loaded image, not by re-reading files), so its
output is immediately usable by anything downstream. Verified by running
`infer` then `embed`, `review` (headless), and the HTML gallery step against
the same database. While fixing this, also caught and fixed a related gap
it exposed: the HTML gallery step picked which label column to use
(`new_label` vs raw `label`) once per *database*, based on whether the
column existed -- but now every database has a `new_label` column, so a
database fresh out of `infer`/`detect` (not yet curated, `new_label` NULL on
every row) would show "None" as every crop's caption instead of the model's
actual prediction. Fixed to fall back per *row* (`COALESCE(new_label,
label)`) instead.

**Cross-dive image collision bug (export voc, export html)**: both used to
group detections by *bare filename* and reconstruct each image's path as
`image_dir / image_name`. For a mission with nested per-dive subdirectories,
two images with the same filename in different dives (e.g.
`dive01/img_0001.jpg` and `dive02/img_0001.jpg`) collided into one key, and
both got silently mapped onto whichever single flat path happened to exist —
merging detections from different dives onto the wrong image, or exporting
one dive's identifications under the other's filename. Fixed by grouping on
the full `image_path` already recorded at detection time (which is also why
`export voc`/`export html` no longer need a separate `image_dir` argument),
and by disambiguating output filenames with their parent directory name.
Verified with two images sharing a filename in different subdirectories.

**The step-8 bug ("code seemed to run, but no results saved, no db
generated")**: `9_inference.py` (the original pre-refactor script; this is
step 8 in the current numbering) used to buffer every detection row from
*every* image batch in memory and write them to the database in a single
`executemany()` call **after the entire run finished**, outside any
try/except. YOLO would visibly process every image and even save annotated
copies to disk, but if anything went wrong in that one final insert — or the
process was interrupted before reaching it — nothing ever reached the
database. It now writes each batch's rows immediately after that batch
finishes, so progress is durable as the run proceeds, and:

- bad `--model-path`/input directories now raise immediately with a clear
  message instead of failing silently or deep inside Ultralytics;
- `--device` defaults to `auto` (was hardcoded to `mps`, which errors or
  silently produces nothing on a machine without Apple Silicon);
- the database connection is always properly closed (via a context manager),
  which is required for DuckDB to guarantee writes are flushed;
- the run always ends with an explicit summary: images processed, rows
  written, and where — so "nothing happened" is no longer possible to miss.

**Other real bugs fixed along the way:**
- `requirements.txt` used to contain a dump of raw `import` statements
  copied from the GUI script (e.g. `from PySide6.QtCore import Qt`), not
  actual package names — `pip install -r requirements.txt` would have failed
  outright, silently leaving packages like `duckdb`/`evoc` never installed.
- Step 3's `--limit` option ran `DELETE FROM predictions WHERE rowid NOT IN
  (...)` — passing `--limit` for a quick test permanently deleted every row
  beyond N. `--limit` now only limits what's read for clustering.
- Step 4 needed the original full-frame images to build its review grids,
  but its CLI never exposed an `--image-dir` option — it hardcoded
  `Path(".")`, so every ROI silently failed to render unless you happened to
  run it from inside the image directory. It now reads the ROI crop already
  stored in the database instead of re-reading images from disk, which
  removes the need for that argument entirely.
- The GUI's label filter was interpolated directly into SQL and broke on
  labels containing a quote; it's now parameterized. Its database connection
  is now closed on exit too.
- Every step's hardcoded `device="mps"` is now resolved against actual
  hardware (`mbariml.yolo_utils.resolve_device`), with a clear error if you
  explicitly request a device that isn't available.
- Exceptions used to be caught and printed as a single line, discarding the
  traceback (`print(f"Error ...: {e}")`). Everything now goes through
  `logging`, and unexpected exceptions are logged with `logger.exception(...)`
  so the traceback is never lost in the console output.
- The review GUI's "Sort by Sharpness" was sorting by a column hardcoded to
  `0.0` for every row — a no-op. Step 1 now computes a real per-ROI blur
  score (see "Using the review GUI (step 5) well" above). While reworking
  that screen, labeling/deleting were also changed to update only the
  affected thumbnails instead of rebuilding the entire ~500-thumbnail page on
  every click, instead of the full rebuild the original did on every action.
- The embedding backbone switched from DINOv2 to DINOv3 (see "Embeddings
  (DINOv3)" above) for accuracy. While there, preprocessing switched from
  hand-picked constants (a fixed 518x518 resize + CLIP's mean/std, regardless
  of which model was actually loaded) to `timm.data.resolve_data_config`,
  which builds the exact preprocessing pipeline for whatever model is
  loaded, instead of silently being wrong for a model with different
  expected input stats.
- `mbariml run`'s internal calls to each step's function (bypassing Click,
  which normally resolves `typer.Option(...)` defaults to real values) have
  to pass every non-required parameter explicitly, or the raw sentinel
  object leaks through and gets treated as truthy. Adding `--force` to
  `mbariml embed` without updating `run`'s call site would have made every
  `mbariml run` silently re-embed the entire database on every invocation;
  caught and fixed before it shipped.

**Structure**: shared logic (DB schema/connection handling, image
directory scanning, YOLO model loading/device selection) now lives in
`src/mbariml/` instead of being copy-pasted across each numbered script;
each step's implementation is in `src/mbariml/steps/`. `8_query.py` was
previously a notebook-style script with a hardcoded absolute database path
and an uncommented query that nulled out every label — it's now a small,
safe, reusable `mbariml query DB_PATH "SELECT ..."` command.

The original numbered scripts (`1_generate_detections.py`, etc.) have been
removed in favor of the `mbariml` CLI.
