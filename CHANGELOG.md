# Changelog

What changed, newest first. The deep-dive explanations of *how* things work
(embeddings, clustering, the export formats, the review GUI) live in
[README.md](README.md); this file records what changed and why.

Most entries below carry measurements rather than adjectives — several of
these were performance or correctness bugs that were only found by timing
each stage separately, and the numbers are kept so a future regression is
recognizable.

---

## 0.21.0 — exports are named after the source image, not `<parent>_<image>`

Every export that writes one file per source image prefixed it with the
image's parent directory, so an export of `PROSILICA_L` imagery produced
`PROSILICA_L_1784391225718813.id`, `PROSILICA_L_...txt`, `PROSILICA_L_...xml`
and so on. The prefix is noise when filenames are already unique — survey
imagery is named by timestamp, and neither of this deployment's databases has
a single colliding filename across 1,095 images.

Files are now named after the source image itself: `1784391225718813.id`.

The prefix existed for a real reason, though, and that reason still holds:
flattening a nested mission tree into one output directory is exactly when
`dive01/img_0001.jpg` and `dive02/img_0001.jpg` become the same output file,
and the second silently overwrites the first — an entire image's annotations
gone with no error. So naming is now decided for the export as a *whole*
rather than per file: unique filenames are used bare, and only the names that
genuinely clash fall back to `<parent_dir>_<stem>`, with a warning naming
them. One unlucky pair in a 10,000-image survey no longer prefixes the other
9,998. If two images resolve to the same name even after disambiguation, the
export fails rather than overwriting.

This also consolidates the rule. `export voc` and `export html` each built
the prefix inline with their own f-string rather than calling the shared
helper, so "how are output files named" had three implementations and could
drift. There is now one `export_stem_map`, computed once per export and
threaded through every artifact — label files, XML, sidecars, gallery images
and crops, the manifest, the split lists, the stats matrix. That threading is
load-bearing for `export yolo`: the split lists deliberately see a narrower
path set than the label files (images missing from disk are dropped), so a
map recomputed per artifact is precisely how `images/X.jpg` would stop lining
up with `labels/X.txt`.

Verified on a 995-image export: 995 label files, 995 split entries, 995
manifest rows, 0 mismatches between any pair of them, 0 names carrying a
prefix. Then verified the fallback by rewriting two images to share a
filename across two dives: 993 bare names, 2 prefixed (`dive01_CLASH.id`,
`dive02_CLASH.id`), warning emitted.

---

## 0.20.0 — .id rows are flat CSV, and lon/lat/depth are stated once

Each identification is now one plain CSV row:

```
# index,label,confidence,center_x,center_y,lon,lat,depth,tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y
0,Crinoidea,0.9463,1499,482,0.0,0.0,0.0,1470,454,1528,454,1528,510,1470,510
```

Two changes from 0.19.0's TAB-delimited, point-grouped rows.

**Every value is comma-separated**, so `csv.reader`, pandas or a spreadsheet
reads the rows with no custom splitting, and the last comment line is a
usable header. The tab delimiter existed only to survive labels containing
spaces; commas do that too, since no taxon name contains one. Rows are
written with `csv.writer`, which quotes only when it must — a future
`Nudibranchia, sp. A` comes out as `"Nudibranchia, sp. A"` instead of
silently adding a column and shifting every coordinate after it.

**`lon,lat,depth` appear once, beside the center**, instead of once per
vertex. They describe where the observation *is*, and an observation has one
position; the old layout shipped five identical `0.0,0.0,0.0` triples per row
for a single unknown. Rows are a third shorter (80 chars vs ~120 on this
survey).

This matches what actually happens downstream: a separate process populates
lat/lon/depth per center point, so one triple per observation is the right
shape. Per-vertex geolocation, which the original format reserved space for
and nothing ever filled, would now be a format change rather than filling in
existing columns.

The header names those three as the only columns that process should change
(0-based 5, 6, 7), and tells it to rewrite with a CSV-aware writer so a
quoted label stays quoted. Checked by simulating the merge end to end:
lon/lat/depth replaced on every row, re-read, 16 fields intact and all
geometry byte-identical.

Verified across all 35,492 rows of a real export: every row parses to exactly
16 fields, `center` is the integer midpoint of its corners, the four corners
form a rectangle, every corner lies within `[0,W]x[0,H]`, every center within
`[0,W-1]x[0,H-1]`, and all 436 rows whose label contains a space round-trip
intact.

---

## 0.19.0 — .id rows are TAB-delimited, and the header explains itself

**The data rows were ambiguous.** Fields were space-separated, but taxon
names routinely contain spaces, so `index, label, confidence, *corners =
row.split()` yielded `label="marine"`, `confidence="organism"` — wrong, and
wrong silently. Measured on one real survey export: **436 of 35,492
identifications** across 7 taxa (`marine organism`, `Heteropolypus
ritteri`, `Spectrunculus grandis`, `Geological feature`, `LRJ Complex`,
`Saccocalyx pedunculatus`, `Bathyalcyon robustum`). Since the entire purpose
of these files is that a navigation process re-parses them later to fill in
lon/lat/depth, that is exactly where it would have surfaced.

Rows are now TAB-delimited. A tab cannot occur inside a taxon name, so
`row.split('\t')` is unambiguous for every label with no quoting or
escaping. Verified by re-parsing all 35,492 rows in all 995 files of a real
export: 0 malformed, 51 distinct labels recovered intact.

**The header now explains the format instead of compressing it.** It was one
line:

```
# index label confidence  vertices(TL,TR,BR,BL as px_x,px_y,lon,lat,depth)
```

which describes a nested structure — four corners, each itself five values —
inside a single parenthesis, so it reads as one flat list of nine things and
leaves the reader to work out where a corner ends. It is now a field-by-field
legend naming each column, the corner order, the five values within a corner,
their units, and the parse. Whoever writes the navigation merge reads this to
learn the format; nine comment lines per file is a fair price.

**Each row now carries the observation's position as a `center` point**, in
front of the four corners and in the identical five-value encoding, so the
navigation merge fills in its lon/lat/depth like any other point. One pixel
position is what most consumers actually want; the corners remain for
anything needing the extent.

The center is the integer midpoint of the *rounded* TL/BR corners printed
beside it, not a separately-rounded midpoint of the underlying floats. Those
two disagree by a pixel on 222 of this survey's 35,492 rows — either is
defensible alone, but a file whose stated center contradicts its own corners
recreates the "two consumers compute different centers" problem the field
exists to remove. Integer `//` rather than `round()` on the midpoint, since
Python's round-half-to-even would make an odd span's tie-break depend on
coordinate parity. Verified across all 35,492 rows: 0 disagreements.

**The header records `image_width`/`image_height`.** A consumer can now
bound-check a coordinate, or place it in the frame, without opening the
imagery. Read via PIL's lazy `Image.open`, which parses the header and
stops: 6.3 ms/image against cv2.imread's 43.9 ms on this survey's 1936x1456
TIFFs, so ~6s rather than ~44s across a 995-image export. This is the only
reason `export id` touches the imagery at all. An image that has moved or
won't open writes `unknown` and logs a count; its identifications are
unaffected, since all box geometry comes from the database.

Adding the dimensions immediately caught something: **884 coordinates sat
outside the frame** — all of them exactly `x=1936` or `y=1456`, only ever on
the right/bottom corners, from 422 boxes touching the right edge and 13 the
bottom. Not a data bug. Corner coordinates are box *edges*, spanning
`[0,W]x[0,H]`, not pixel indices spanning `[0,W-1]x[0,H-1]`: a box flush
against the right side legitimately has `px_x` one past the last column, and
that is precisely what makes `width = right - left` exact — the same
arithmetic `export yolo` already relies on. So the legend now states the
real rule rather than declaring 884 good coordinates invalid, and notes that
`center`, being a true pixel, does stay within `0..W-1`/`0..H-1`. Verified
across all 35,492 rows: 0 corners outside `[0,W]x[0,H]`, 0 centers outside
`[0,W-1]x[0,H-1]`.

The point legend spells out that `px_x`/`px_y` are a *pair* — column from
the left edge, row from the top, both 0-based — rather than "pixel
coordinates", which read ambiguously enough that the first question asked of
the new format was whether a pixel has a single unique number. (It does not.)

**`source_image` records the full path**, not the basename. With
`--output-dir` the sidecars no longer sit beside their imagery, so a
basename alone would not say which dive an identification came from — and a
survey has many directories holding an image of the same name.

---

## 0.18.0 — `stats` is an export subcommand only, and `export id` can emit raw detections

**`mbariml stats` is gone; use `mbariml export stats`.** 0.17.0 registered it
in both places to avoid breaking muscle memory, which left two names for one
command and no clear answer about which was canonical. One name now. Every
reference in the README, cheat sheet and SCHEMA.md was updated, and the
cheat sheet's captured `--help` block was regenerated from the live CLI
rather than hand-edited — it had drifted to v0.15.0 and was missing
`--include-unverified`, `--yaml-name` and `--output-dir` entirely.

**`export id` takes `--include-unverified`.** The sidecars are an
identification product, not only a training input — sometimes what's wanted
is everything the detector found, not only what a human confirmed. Default
is unchanged (verified only, matching `export yolo`/`voc`), and the log line
now says which population it wrote. `export yolo` and `export voc`
deliberately still have no such flag: unverified boxes in a training set is
the failure 0.16.0 exists to prevent.

---

## 0.17.0 — `export stats`, `export id --output-dir`, and a version that had drifted

**`stats` is now also `mbariml export stats`.** It reports the same
population the annotation exports write (`mbariml.db.curated_where`), so it
belongs with them — its numbers are a preview of what a training set will
hold, not a separate kind of thing. Registered in both places rather than
moved: `mbariml stats` is what the docs, the cheat sheet and muscle memory
all say, and breaking that to relocate a command earns nothing. Same
callback, so the two cannot diverge.

**`export id` takes `--output-dir`.** It wrote every sidecar next to its own
source image, which is the right default but the only option — no good on a
read-only survey volume, or when handing the identifications off without the
imagery. Unset, behaviour is unchanged. Set, the files are collected in one
directory and named by `disambiguated_stem` (`<parent>_<stem>.id`), because
flattening a nested mission tree is exactly the case where two dives'
identically-named images would silently overwrite each other's
identifications. Each file's header still records the original image name,
so provenance survives the flattening.

**`__version__` was three releases stale.** It was a second, hand-maintained
copy of the version in `src/mbariml/__init__.py`, reading `0.13.0` against a
`pyproject.toml` that said `0.16.1`. Not decorative: `export id` stamps it
into every sidecar as `# generator: mbariml vX.Y.Z`, so every .id file
written since 0.13.0 named a version that had not produced it. It now comes
from the installed package metadata (`importlib.metadata.version`), which
cannot drift from `pyproject.toml`. Note this reads what was *installed*, so
an editable checkout reports the version as of its last `pip install -e .` —
re-run that after a version bump, as the cheat sheet already advises for any
`pyproject.toml` change.

---

## 0.16.1 — a mistyped database path created an empty database

`db.connect()` passes its path straight to `duckdb.connect()`, which
**creates** a database at whatever path it is handed. Every read-only
command used it, so one mistyped character was enough to make a new, empty
database and then fail four frames deep with:

```
CatalogException: Catalog Error: Table with name predictions does not exist!
```

That message points at the schema, which is not the problem. Reproduced from
a real invocation: `yolo_predictions.duckdbb` (a doubled `b`) left a 12 KB
phantom database sitting next to the real 1 GB one, where it would have been
easy to later mistake for a failed export rather than a typo.

`connect(must_exist=True)` now checks first and is used by all ten commands
that read an existing database (`export {yolo,voc,id,html}`, `stats`,
`query`, `embed`, `cluster`, `refine`, `remap-labels`). The two ingest
commands still create, which is their job. The error names the typo and
looks in the same directory for what was probably meant:

```
Error: Invalid value: No such database: .../yolo_predictions.duckdbb.
Did you mean yolo_predictions.duckdb?
```

It is raised as `typer.BadParameter` specifically — checked against click
8.5.0, a bare `click.UsageError`, a bare `ClickException` and even a
`BadParameter` *subclass* all print a full rich traceback instead, and a
stack dump for a typo buries the one line that says what is wrong. Exits 2,
touches nothing on disk, and needs no reinstall (it rides the existing
console-script entry point rather than changing it).

`require_verified_column` also reports a database with no `predictions`
table as not being an mbariml database, rather than letting the raw
CatalogException through.

---

## 0.16.0 — exports dropped the localizations you verified but didn't rename

`export yolo`, `export voc` and `export id` selected rows with
`new_label IS NOT NULL AND new_label != 'noise'`. That filter means "boxes
whose name I retyped", not "boxes I confirmed": the review GUI's Verify
button sets `verified = 1` and leaves `new_label` NULL, and only relabelling
writes it (`gui/annotation_service.py`). Every ROI where the detector was
already right and the reviewer simply confirmed it was invisible to all
three exports.

Measured on a real 35,492-row survey database, every row of it verified:

| | boxes | images | classes |
|---|---|---|---|
| exported before | 2,805 | 896 | 50 |
| exported after | **35,492** | 995 | 51 |

92% of the reviewer's work was being discarded. The class count barely
moved, which is what made this hard to spot from the outside: nearly every
taxon was present in the export, just represented by a tiny and badly
unrepresentative fraction of its boxes. Per class the loss was
lopsided, because it fell hardest on the classes the detector got *right*
most often: Ophiuroidea exported 86 of 11,251 verified boxes,
Hexactinellida 35 of 3,601, Muusoctopus 11 of 2,063.

Worse than the omission: 29,917 of the dropped boxes sat on images that
*were* in the export. Those images got a label file listing only the
handful of retyped boxes, so every other animal in the frame had no label
line — and YOLO reads an unlabeled object as background. The export was
actively teaching the model that 10,149 Ophiuroidea and 3,242
Hexactinellida were empty seafloor, which is why training on it went
backwards rather than merely plateauing.

`export voc` had the same bug wearing a disguise: `WHERE new_label !=
'noise'` looks permissive, but SQL three-valued logic makes
`NULL != 'noise'` evaluate to NULL rather than TRUE, so it dropped the
identical rows.

The rule is now defined once, in `mbariml.db` (`EFFECTIVE_LABEL_SQL` and
`curated_where()`), and used by every consumer:

| In the database | Exported? | Name used |
|---|---|---|
| verified, name unchanged | yes | the original detector `label` |
| verified, name updated | yes | `new_label` |
| not verified | no | — |

It lives in one place because it previously did not: `stats`, `cluster
--label-source new` and `export html` already used `COALESCE(new_label,
label)` — `stats` even documented it as "the effective label" — while the
three dataset exports had drifted onto the bare `new_label` filter. Nothing
connected them, so `stats` would report 35,492 localizations across 51
classes while `export yolo` beside it wrote 2,805, with no indication the
two numbers meant different things.

Also changed:

- `stats` and `export html` now default to the verified-only population, so
  their output matches what the dataset exports write. Both gained
  `--include-unverified` to restore summarizing/browsing raw detector
  output on a database that has not been reviewed yet — their other
  documented use, which a hard filter would have broken silently.
- `stats` says which population it counted in its table headers
  (`verified only` / `verified + unverified`), and warns rather than
  printing an empty table when a database has no verified rows.
- The dataset exports fail with an explanatory error on a database
  predating the `verified` column, instead of writing a well-formed empty
  dataset — the exact silent failure this release exists to end.
- `export voc`'s `new_names.txt` now applies the same rule as the XML files
  beside it (it was computing its label list from a different query) and is
  sorted, so re-exporting is byte-stable.
- `cluster` is deliberately unchanged: it still groups and relabels
  unverified rows, since naming un-reviewed ROIs is the point of it.

### `export yolo` now writes the dataset YAML

Previously it wrote `names.txt` and left assembling the Ultralytics config
to whoever ran the training, which meant hand-maintaining a `names:` list
against a file that changes every time the database is re-reviewed — and
this release changes the class list on every existing database, so that
hand-maintenance was about to go wrong quietly.

`<output_dir name>.yaml` is written alongside the splits, in the same format
the existing MBARI training configs use (`train`/`val`/`test`, `nc`,
`names`). `nc` and `names` come from the same in-memory list that assigned
the class indices in `labels/` and was written to `names.txt`, not from a
second query, so no reordering can get between them: resolving every label
file's class index through the YAML's `names` reproduces the database's
per-class counts exactly, which is checked directly.

Its split paths are relative and it writes no `path:` key, so Ultralytics
resolves them against the YAML's own directory — naming the YAML at training
time is all that's needed, and the dataset directory works unchanged whether
it's read from `/Volumes/M3_ML/...` on macOS or `/mnt/M3_ML/...` on the
Linux trainer. Same reasoning as the `./images/<file>` lines already inside
the split files. Checked against Ultralytics 8.4.154 via
`check_det_dataset`, from an unrelated working directory and again after
copying the folder to a different path.

An empty split is omitted from the YAML rather than written as a key
pointing at an empty file: with `--split-ratios '85 15 0'` there is no test
set, and a `test:` line promising one would fail at evaluation, long after
the export looked fine. An empty *val* split warns, since Ultralytics needs
one to train. `--yaml-name` overrides the filename, `--no-yaml` skips it.

---

## 0.15.0 — choose which label names a cluster

`cluster` names each cluster after the most common label among its members, and
it read the raw detector class (`label`) to do it, unconditionally. On a
database that has already been through review that is precisely the wrong
column: a single-class detector puts `object` in it for every row, and the
human decisions live in `new_label`, invisible to the naming step. So
re-clustering a curated database discarded the curation twice over — the bulk
UPDATE overwrites `new_label`, and the replacement names were voted on from a
column the reviewer never touched.

The new `--label-source` says which label to use:

- `original` (default, unchanged behaviour) — the raw detector class.
- `new` — the curated `new_label`, falling back to the raw class for
  un-reviewed rows, the same convention `stats` and `export html` already use.

Measured on a 300-ROI database curated into three taxa and clustered into three
groups: the default returns `object_1`/`object_2`/`object_3`, while
`--label-source new` returns `Muusoctopus`/`Sponge_sp_A`/`Coral_bamboo`. With a
third of the rows' reviews removed, that third's cluster falls back to the raw
`object` while the two reviewed clusters keep their curated names.

Both sources are wrapped in `COALESCE(..., 'unlabeled')` so the vote can never
elect NULL and name a cluster `None_1` — reachable in `original` mode too, for
a row whose detector class was never recorded.

Two things worth stating plainly, since neither is obvious from the command and
both cost real work to discover:

- **Clustering overwrites `new_label` for every embedded row, verified rows
  included, whichever source is chosen.** It does not merge and there is no
  undo, and `verified` is not cleared either, so afterwards those rows still
  look human-reviewed while carrying a machine-generated label. Work on a copy
  of the database. Copy it into its own directory — `cluster` writes
  `roi_grids/` next to the database, so a same-directory copy lands grids on
  top of the originals. The copy carries the embeddings, so `embed` never
  re-runs and each clustering iteration takes seconds.
- `--label-source new` reads the very column clustering writes back to, so
  running it twice in a row feeds the first run's generated names back in as if
  they were human decisions. It is for the first pass over a reviewed database,
  not for repeated re-runs.

Also fixed: `mbariml run` calls `cluster()` directly as a Python function
rather than through Typer, so every parameter must be passed explicitly — an
omitted one arrives as a `typer.OptionInfo` object instead of its default. The
new parameter is passed explicitly there, which the chain's cluster stage was
re-run end to end to confirm.

---

## 0.14.0 — YOLO train/val/test splits, and a visible similarity-search scope

### `export yolo` now writes a dataset, not just labels

`export yolo` wrote `labels/` and `names.txt`, which is most of a YOLO
dataset but not a trainable one — the image lists a training config actually
points at still had to be produced by hand. It now also writes `train.txt`,
`val.txt` and `test.txt`, one `./images/<file>` path per line.

Relative paths deliberately: the dataset directory only refers to itself, so
it can be zipped and moved to a training box without rewriting anything,
which absolute paths recorded on the exporting machine could not survive.
The filenames are the same collision-safe `<parent_dir>_<stem>` names
`copy_images.py` copies to and `labels/` is keyed by, so `images/X.jpg` ↔
`labels/X.txt` pairs up by construction — the pairing YOLO resolves by
swapping `/images/` for `/labels/`. Export plus copy script now leaves a
directory that trains as-is:

```bash
mbariml export yolo predictions.duckdb dataset/
python3 dataset/copy_images.py --dest dataset/images
```

`--split-ratios "85 10 5"` (default), `--split-seed 42` (default),
`--test-images-file` to pin a fixed benchmark set into test across every
export, `--no-splits` to skip. Ratios that don't sum to 100 are rejected
rather than rescaled — a typo'd `"80 10 5"` is a miscount, not a request for
a dataset missing 5% of its data.

Seeded by default so re-exporting after relabelling a handful of ROIs
reproduces the same split. An unseeded shuffle would silently reshuffle what
was held out on every export, making any two models trained across one
incomparable.

Splitting is per **image**, never per box: two crops of one frame on opposite
sides of the train/val boundary leak the same background, lighting and often
the same individual animal across the split, which inflates validation scores
on benthic transect imagery where consecutive frames already overlap heavily.

Also fixed alongside: images recorded in the database but missing from disk
were listed in `image_manifest.csv` despite no label file being written for
them. They're now excluded from both the manifest and the splits — a split
line pointing at an image that was never copied surfaces much later as a
training-time error.

### Similarity search now states its own scope

The right-click similarity search always ranked the entire matching ROI set,
every page of it — a full-table scan and a numpy matmul over every stored
embedding, not a re-ordering of the visible page. But nothing in the GUI said
so, and one situation makes it look otherwise: `compute_similarity_order` can
only rank rows that *have* an embedding, and a partial, interrupted, or
`--limit`ed `mbariml embed` leaves the rest out silently. A search ranking
3,001 of 20,000 ROIs and one ranking all 3,001 there were displayed
identically, as a page count — which reads exactly like "it only sorted what
I was looking at".

The status line now reports the pool outright: *"sorted by similarity to ROI
#812 (all 8,412 matching ROIs)"*, or *"(3,001 of 8,412 matching ROIs — 5,411
not embedded yet)"*, with a matching log warning naming the fix. The new
`query_service.count_similarity_pool` counts the same pool the ranking uses
minus its embedding requirement, sharing one WHERE-clause builder with
`compute_similarity_order` so the two can't drift into describing different
populations — the denominator is only worth showing if it's truthful.

---

## 0.13.0 — cluster naming for single-class detectors

`cluster` labelled each cluster with the dominant original label of its
members. With a single-class detector that is the same string for every
cluster, so the clustering was immediately thrown away: every row came back
labelled `object`, and since the review GUI filters and sorts on `new_label`,
the groups became indistinguishable. Only `evoc_clust` still held the
structure.

The new `--naming` option defaults to `auto`, which keeps the dominant label
but appends an index whenever one label wins more than one cluster. A
single-class run now produces `object_1`, `object_2`, … Verified on a 147-ROI
run: six clusters that previously collapsed to a single `object` label come
back as `object_1`–`object_6` with 34/35/12/16/12/26 ROIs.

This helps multi-class runs too. On the same ROIs with the detector's real
labels, `Actiniaria` and `Ceriantharia` each won one cluster and keep their
bare names, while four distinct sponge clusters become `Hexactinellida_1`–`_4`
instead of being flattened together. `dominant` restores the old behaviour and
`indexed` always suffixes. The suffixing matches what `refine` already did for
sub-clusters, and numbering follows cluster id so re-runs with the same seed are
stable.

---

## 0.12.0 — brightness/contrast for the review grid

Two sliders below the tile-size slider adjust the ROI thumbnails across the
whole mosaic. Faint animals against sediment, and low-contrast crops from
deep footage, are considerably easier to identify stretched than at native
exposure — and previously the only way to get a better look was to open the
source image outside the tool.

Contrast pivots around mid-grey rather than around black: `convertScaleAbs`'s
plain `alpha * pixel + beta` scales about zero, so raising contrast would
also wash the tile brighter and the two controls would fight each other.
Folding `128 * (1 - alpha)` into beta keeps mid-grey fixed, so contrast
stretches the range about the middle while brightness alone shifts it.

Applied per tile in `RectWidget.getpic`, *after* the thumbnail resize, so the
work happens on a ~120×120 image rather than the full-resolution crop; slider
movement is debounced (70 ms) so dragging across a 500-tile page coalesces
into one re-render rather than 30. Strictly view-only — the stored `roi` blob
is never touched, so nothing exported changes. A **Reset** button returns
both to neutral, and the setting carries across paging and sorting.

---

## 0.11.0 — video ingest, `infer` merge, phases instead of step numbers

### Added

**`mbariml infer video`** — video ingest, in two modes.

`--mode track` (default) keeps **one ROI per tracked object**, which is what
you want for building curation/training data from video: an animal in view
for 300 frames is one observation, not 300 training examples. Measured on a
real 8-second benthic clip: 633 detections collapsed to 4 tracks → 4 ROIs.
Tracking is necessarily two passes, because a track's representative frame
can't be chosen until the track has ended — pass 1 tracks the whole video
recording *metadata only* (memory is O(open tracks), not O(video)); pass 2
sweeps forward once, extracting just the chosen frames. Pass 2 sweeps rather
than seeks deliberately: `CAP_PROP_POS_FRAMES` is unreliable on long-GOP
encodings and lands on the nearest keyframe, which would silently pair a
detection's box with the wrong pixels.

`--track-roi` chooses which frame represents a track, defaulting to
`best-conf-central` — the most confident frame of the track's *middle third*.
A track's first and last frames are when the animal is entering or leaving
view (clipped at the frame edge, occluded, motion-blurred), and plain
max-confidence picks exactly those. `sharpest-central`, `best-conf`, and
`center` are also available; short tracks fall back automatically.

`--tracker` takes any Ultralytics tracker config — default `tracktrack.yaml`,
or a path to your own YAML. Note that **track count is governed by the
tracker's own thresholds** (`track_high_thresh`, `new_track_thresh`), not by
`--conf`: the same clip gave 1 track with stock settings and 4 with lowered
ones.

`--mode stride` skips all of that and samples every Nth frame as an
independent image.

Chosen frames are extracted to real JPEGs under `OUTPUT_DIR/frames/`, with
`image_path` pointing at those files, so **video rows are indistinguishable
from image rows to every downstream command** — review, embed, cluster, all
four exports, html, and stats work unchanged, with no video-aware code
anywhere else in the pipeline. The alternative (store the video path plus a
frame number, resolve lazily) would have meant teaching five separate
consumers to decode video, and the exports would have had to materialize
frames anyway. Provenance is kept in new nullable columns — `video_path`,
`frame_number`, `frame_time_s`, `track_id`, `track_length`. Frames with no
detections are never written.

**"Open Video" button in the review GUI** — for video-derived ROIs, opens the
source footage at the exact moment of that detection, from the row's
`video_path` + `frame_time_s`. Tries IINA, then mpv, then VLC (all of which
honor a start position), falling back to the default browser with a `#t=`
media fragment. Disabled for image-derived ROIs, so the button state itself
answers "did this come from video?".

### Changed

**`detect` and `infer-images` merged into `mbariml infer images`.** They were
~90% redundant: same schema, same ROI cropping, same provenance record,
differing only in batching, annotated-image saving, and a 16× gap in default
confidence — flags and defaults, not architecture. Keeping two copies meant
every fix had to be made twice. The intent distinction survives as
`--preset curate` (conf 0.005, imgsz 1952, no annotated images — mine
everything, then cluster/review and discard the noise) and `--preset predict`
(conf 0.08, imgsz 992, saves annotated images). `curate` is the default
deliberately: an over-permissive threshold is recoverable by filtering later,
while a too-strict one silently drops detections you can't get back without a
full re-run. Any individual option overrides its preset.

**Numbered steps replaced by four phases** — Ingest, Enrich, Curate, Emit.
The numbering had been reshuffled three times as commands merged and moved,
each pass rippling through every docstring and both documents, for a sequence
that was never actually linear. Step modules lost their numeric filename
prefixes to match (`step8_inference.py` → `infer_images.py`, and so on), and
`mbariml run` now takes named stages (`--from export`) instead of
`--from-step 6`. `mbariml run` also gained `--media images|video`.

### Fixed

**Ingest recorded `image_path` verbatim**, so a relative input directory
produced relative paths that stopped resolving the moment `review` or an
export ran from a different working directory — the whole "images are located
via the path recorded at ingest" contract depends on those paths being
resolvable later. Both ingest commands now store absolute paths.

**Ingest always numbered rows from 0**, so a second run into an existing
database collided on the UNIQUE `id` index instead of appending. Ids now
continue from the database's own counter (`db.next_free_id`), which is what
makes one database holding several videos — or images and video together —
actually work. Verified: 5 image rows + 4 video rows in one database, ids 0–4
and 5–8, with `stats` aggregating across both.

---

## 0.10.0 — "Add New ROI" in the review GUI

Previously the only way to fix a missed detection was to re-run ingest with a
lower `--conf`, or add it out-of-band and re-import — there was no way to just
draw the box the model should have found.

The review GUI now has a green "Add New ROI" button that arms drawing mode on
the full-image panel: drag a box, type its label, and repeat for as many as
needed before clicking the button again or pressing Escape. Implemented as
`_DrawableViewBox` in `detail_view.py` (a `pyqtgraph.ViewBox` subclass that
reuses pyqtgraph's own built-in `RectMode` scale-box mechanics for the
rubber-band visual and coordinate math, but reports the finished rectangle
instead of zooming to it), plus `annotation_service.insert_roi` (new row:
`confidence` fixed at `1.0`, `class_id` left `NULL`, `new_label` and `label`
both set to the typed value, `verified` set immediately — a human drew and
named it, so there's nothing left to review).

The embedding is computed right after, off the GUI thread
(`MainWindow._embed_new_roi_worker`, applied via
`annotation_service.set_embedding`), using `embed.embed_roi_bgr` — a
single-item entry point into the exact same model and preprocessing the
batched `mbariml embed` pipeline uses. Verified byte-for-byte identical to
running the same crop through the batched path, so a hand-drawn box's
embedding is guaranteed comparable to every other embedding in the database
rather than a second, potentially drifting implementation.

Not adapted from vars-gridview; new here.

---

## 0.9.0 — `html` folded into `export`, `backfill-sharpness` removed

`mbariml html` became `mbariml export html`, joining `export voc`/`yolo`/`id`
— one step for "produce some downstream output from curated data" instead of
two.

`backfill-sharpness` was removed. It existed to compute sharpness scores for
databases created before ingest did so automatically; every write path has
computed a real Laplacian-variance blur score since 0.7.0, so it no longer
had a database to migrate.

---

## 0.8.0 — `export` group, YOLO export, `stats`

The flat `export-voc` and `export-ids` commands became `mbariml export voc`
and `mbariml export id`, alongside a new `mbariml export yolo` (YOLO-format
label files + `names.txt`) — one downstream annotation format per subcommand,
now that there were three.

`export voc`/`export yolo` also each write `image_manifest.csv` plus a
standalone, stdlib-only `copy_images.py`, so the matching source images can
be pulled down later (e.g. to a laptop's Desktop) without walking the
mission's nested directory structure by hand.

New `mbariml stats`: label counts, boxes-per-image summary statistics, and an
optional image × label count matrix CSV for ecological analysis.

---

## 0.7.0 — clustering could take hours; DuckDB fsyncs per statement

Clustering could run for *hours* on a large database with no visible
progress. The clustering math was never the problem: `evoc.fit_predict()`
has its own JIT-compiled approximate nearest-neighbor search and clusters
50,000 embeddings in ~2.6 seconds, scaling roughly linearly — so DuckDB's
VSS/HNSW extension would not have helped, as there's no repeated
similarity-search query for an index to accelerate. The real cost was writing
results back with one `UPDATE ... WHERE id = ?` per row against the *indexed*
`new_label` column. Measured at 100,000 rows: 38.6 seconds via the per-row
pattern (still trending worse) versus ~6 seconds via a staged bulk
`UPDATE ... FROM` (`mbariml.db.bulk_update`).

Chasing that down surfaced something more fundamental: **DuckDB's Python
driver commits (and fsyncs to disk) after every individual statement** by
default when writing to a file-backed database, even inside a single
`executemany()` call. Measured: an identical 2000-row INSERT took 11.46s
without an explicit transaction and 0.82s wrapped in one — a ~14× difference
from transaction-wrapping alone. `mbariml.db.fast_executemany` now wraps
every bulk write in the pipeline.

---

## 0.5.0 / 0.6.0 — embedding was slow, and got slower the longer it ran

On a real run against 327,045 ROIs on a Mac Studio M3 Ultra with a confirmed
MPS device, throughput was still crawling after an hour. Three compounding
causes, found by isolating and timing each stage rather than guessing:

1. ROIs were embedded one at a time (decode, preprocess, transfer to GPU,
   forward pass, transfer back, single-row DB write, repeat) — fixed by
   batching (`--batch-size`, default 32).
2. Decode/preprocess (JPEG decode, resize, normalize) is CPU-bound work that
   ran in a single-threaded Python loop, pinning one core while dozens sat
   idle — fixed by parallelizing across a thread pool (`--decode-workers`,
   default up to 16 cores).
3. **The dominant bottleneck**: each batch's embeddings were written with
   their own `UPDATE ... WHERE id = ?` executemany call. Model throughput was
   a rock-stable ~80 items/sec, but per-batch DB write time grew from ~1s to
   ~15s over just 45 batches and kept climbing. Fixed by staging embeddings
   into a temp table and applying them with a single `UPDATE ... FROM` every
   `--flush-size` rows (default 2000). Verified: a run that degraded from 45
   it/s to 6.5 it/s (still falling) over 3000 ROIs became a flat ~70–75 it/s
   for the same 3000 ROIs, with no degradation at all.

An earlier attempt to fix the degradation with periodic
`torch.mps.empty_cache()` calls was tested and found to have no effect — the
cause was never GPU memory — and was removed once the real cause was
isolated. See the README's "Embeddings (DINOv3)" section for the full
numbers, including why an MLX port was investigated and not adopted.

The embedding backbone also switched from DINOv2 to DINOv3 for accuracy.
Preprocessing switched from hand-picked constants (a fixed 518×518 resize
plus CLIP's mean/std, regardless of which backbone was loaded) to
`timm.data.resolve_data_config`, which builds the exact pipeline for whatever
model is actually loaded.

---

## Earlier — replacing the original numbered scripts

**Image inference wrote a different schema than every other step**, missing
`roi_index`, `roi` (the crop blob), `embedding`, and `new_label`. That meant
`review`, `cluster`, `refine`, `export voc`, and `remap-labels` all failed
outright against its output — only the HTML gallery and `query` happened to
work. Fixed by having it write the same curation schema as everything else,
cropping each ROI directly from Ultralytics' already-loaded image rather than
re-reading files. While fixing it, a related gap surfaced: the HTML gallery
picked its label column (`new_label` vs raw `label`) once per *database*
based on whether the column existed — but every database has a `new_label`
column now, so a freshly-ingested, not-yet-curated database showed "None" as
every caption instead of the model's actual prediction. Fixed to fall back
per *row* (`COALESCE(new_label, label)`).

**"Code seemed to run, but no results saved, no db generated."** The original
`9_inference.py` buffered every detection row from every batch in memory and
wrote them in a single `executemany()` call *after the entire run finished*,
outside any try/except. YOLO would visibly process every image and even save
annotated copies, but if anything went wrong in that one final insert — or
the process was interrupted before reaching it — nothing ever reached the
database. Rows are now written after every batch, so progress is durable as
the run proceeds. Alongside that:

- bad model paths and input directories raise immediately with a clear
  message instead of failing silently or deep inside Ultralytics;
- `--device` defaults to `auto` (it was hardcoded to `mps`, which errors or
  silently produces nothing on a machine without Apple Silicon);
- the database connection is always closed via a context manager, which
  DuckDB requires to guarantee writes are flushed;
- every run ends with an explicit summary — images processed, rows written,
  and where — so "nothing happened" is impossible to miss.

**Cross-dive image collision (VOC export, HTML gallery).** Both grouped
detections by *bare filename* and reconstructed each path as
`image_dir / image_name`. For a mission with nested per-dive subdirectories,
two images sharing a filename (`dive01/img_0001.jpg` and
`dive02/img_0001.jpg`) collided into one key, and both were silently mapped
onto whichever flat path happened to exist — merging detections from
different dives onto the wrong image. Fixed by grouping on the full
`image_path` recorded at ingest (which is also why these commands no longer
need a separate `image_dir` argument) and disambiguating output filenames
with their parent directory name.

**Other real bugs fixed along the way:**

- `requirements.txt` contained a dump of raw `import` statements copied from
  the GUI script (e.g. `from PySide6.QtCore import Qt`), not package names —
  `pip install -r requirements.txt` would have failed outright, silently
  leaving `duckdb`/`evoc` uninstalled.
- Clustering's `--limit` ran `DELETE FROM predictions WHERE rowid NOT IN
  (...)` — passing `--limit` for a quick test **permanently deleted** every
  row beyond N. It now only limits what's read.
- Refine needed the original full-frame images to build review grids, but its
  CLI never exposed an `--image-dir` option — it hardcoded `Path(".")`, so
  every ROI silently failed to render unless you happened to run it from
  inside the image directory. It now reads the ROI crop already stored in the
  database, removing the need for that argument entirely.
- The GUI's label filter was interpolated directly into SQL and broke on
  labels containing a quote; it's now parameterized. Its database connection
  is closed on exit too.
- Hardcoded `device="mps"` is now resolved against actual hardware
  (`mbariml.yolo_utils.resolve_device`), with a clear error if you explicitly
  request a device that isn't available.
- Exceptions were caught and printed as a single line, discarding the
  traceback (`print(f"Error ...: {e}")`). Everything goes through `logging`
  now, with `logger.exception(...)` for unexpected failures.
- The review GUI's "Sort by Sharpness" sorted by a column hardcoded to `0.0`
  for every row — a no-op. Ingest now computes a real per-ROI blur score.
  While reworking that screen, labeling and deleting were changed to update
  only the affected thumbnails instead of rebuilding the entire
  ~500-thumbnail page on every action.
- `mbariml run`'s internal calls to each command's function bypass Click,
  which normally resolves `typer.Option(...)` defaults to real values, so
  every non-required parameter must be passed explicitly or the raw sentinel
  object leaks through and is treated as truthy. Adding `--force` to
  `mbariml embed` without updating `run`'s call site would have made every
  `mbariml run` silently re-embed the entire database on every invocation;
  caught before it shipped.
- A taxa-by-image CSV export interpolated image filenames into SQL while
  escaping only the column alias, not the string literal — any filename
  containing a single quote broke the query with a syntax error.
- `remap-labels` applied each rename as its own sequential UPDATE, so a
  changes file that swapped two labels (`A,B` then `B,A`) renamed every A to
  B and then every B — including the just-renamed As — back to A, silently
  collapsing both into one label. Every pair is now applied in one pass
  against the original state.

**Structure**: shared logic (DB schema/connection handling, image directory
scanning, YOLO model loading and device selection) lives in `src/mbariml/`
instead of being copy-pasted across each numbered script; each command's
implementation is in `src/mbariml/steps/`. `8_query.py` was a notebook-style
script with a hardcoded absolute database path and an uncommented query that
nulled out every label — it's now a small, safe, reusable
`mbariml query DB_PATH "SELECT ..."`.

The original numbered scripts (`1_generate_detections.py`, etc.) have been
removed in favor of the `mbariml` CLI.
