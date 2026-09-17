"""Export curated labels to YOLO-format label files, plus a names file.

Writes one ``labels/<disambiguated_stem>.txt`` per source image with at
least one curated identification -- every VERIFIED localization, named
``new_label`` where the reviewer retyped it and the original detector
``label`` where they confirmed it unchanged, excluding ``noise`` (see
``mbariml.db.curated_where``, the same rule every export uses), each line
``class_id x_center y_center width height`` normalized to [0, 1] against
that image's actual pixel dimensions (read from the image itself, same as
``export voc``), plus ``names.txt`` (one label per line, in class-id order --
the file Ultralytics training configs expect for ``names:``).

Also writes the Ultralytics dataset YAML (``<output_dir>.yaml`` by default):
split paths, ``nc``, and the ``names`` list. Its ``nc``/``names`` are taken
from the same in-memory list that assigned the class indices in ``labels/``
and was written to ``names.txt``, so the three can never disagree about
which class index is which taxon. Its split paths are relative, like
everything else here, so the directory is portable; an empty split is
omitted rather than advertised as a key pointing at an empty file.

Also writes ``train.txt``/``val.txt``/``test.txt`` -- the image lists
Ultralytics/darknet training configs point at -- holding ``./images/<file>``
RELATIVE paths, one per line. Relative deliberately: a dataset directory
that only refers to itself can be moved, zipped, or copied to a training box
without rewriting every path, which absolute paths recorded on this machine
could not survive.

Deliberately does not also write copies of the source images into an
``images/`` directory here -- label files are cheap text, but copying every
image would duplicate the same JPEGs already sitting on the survey volume
this ran against. Pairs with ``image_manifest.csv``/``copy_images.py``
(also written here, see ``mbariml.export_common``) to actually assemble a
self-contained, trainable ``images/`` + ``labels/`` dataset directory:

    mbariml export yolo predictions.duckdb dataset/
    python3 dataset/copy_images.py --dest dataset/images

Every filename in the split files is the same collision-safe
``disambiguated_stem`` name ``copy_images.py`` copies to and ``labels/``
is keyed by, so ``images/<name>.jpg`` <-> ``labels/<name>.txt`` lines up
by construction -- which is exactly the pairing YOLO resolves by swapping
``/images/`` for ``/labels/`` in these paths.

Images are located via the path recorded at detection time, the same as
``export voc``/``export html`` -- no separate image_dir argument needed.

One of the Emit-phase exports, alongside ``export voc``/``export id``/
``export html``.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import typer
from tqdm import tqdm

from mbariml import db
from mbariml.export_common import write_image_manifest_and_script
from mbariml.image_naming import disambiguated_stem
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Export curated labels to YOLO-format label files, plus a names file.")
logger = get_logger(__name__)


def _fetch_curated(conn) -> list[tuple]:
    """Every VERIFIED localization, carrying its effective label.

    Selection and naming are both ``mbariml.db``'s shared rule: a row is in
    the dataset because a human verified it, and it is named ``new_label``
    if they retyped it or the original detector ``label`` if they confirmed
    it as-is. This used to filter on ``new_label IS NOT NULL``, which is
    only the retyped ones -- see EFFECTIVE_LABEL_SQL's comment for what that
    cost a real survey database.
    """
    return conn.execute(
        f"""
        SELECT image_path, {db.EFFECTIVE_LABEL_SQL}, x_min, y_min, x_max, y_max
        FROM predictions
        {db.curated_where()}
        """
    ).fetchall()


def _export_labels(conn, output_dir: Path) -> tuple[int, list[str], list[str]]:
    """Returns (files_written, sorted distinct names, exported image_path strings).

    The returned paths are only those a label file was actually written for
    -- images recorded in the database but no longer on disk are left out.
    They feed both the image manifest and the train/val/test splits, and an
    image with no label file and no file to copy has no business in either:
    in a split file it is a line pointing at nothing, which surfaces much
    later as a training-time error about a missing image.
    """
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows = _fetch_curated(conn)
    names = sorted({new_label for _, new_label, *_ in rows})
    class_index = {name: i for i, name in enumerate(names)}

    grouped: dict[str, list[tuple]] = {}
    for image_path, new_label, x_min, y_min, x_max, y_max in rows:
        grouped.setdefault(image_path, []).append((new_label, x_min, y_min, x_max, y_max))

    written = 0
    missing = 0
    exported_paths: list[str] = []
    for image_path_str, boxes in tqdm(grouped.items(), desc="Exporting to YOLO"):
        image_path = Path(image_path_str)
        if not image_path.exists():
            missing += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            missing += 1
            continue
        height, width = image.shape[:2]

        lines = []
        for new_label, x_min, y_min, x_max, y_max in boxes:
            cx = ((x_min + x_max) / 2) / width
            cy = ((y_min + y_max) / 2) / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height
            lines.append(f"{class_index[new_label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path = labels_dir / f"{disambiguated_stem(image_path)}.txt"
        label_path.write_text("\n".join(lines) + "\n")
        written += 1
        exported_paths.append(image_path_str)

    if missing:
        logger.warning("%d image(s) recorded in the database could not be found on disk and were skipped "
                        "(their labels were not written -- normalizing boxes needs actual image dimensions).", missing)
    return written, names, exported_paths


SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT_RATIOS = "85 10 5"

# What each split-file line is prefixed with. YOLO resolves an image's label
# file by swapping "/images/" for "/labels/" in its path, so this prefix is
# what makes `images/<name>.jpg` <-> `labels/<name>.txt` resolve, and it must
# stay in sync with the `images/` directory copy_images.py is pointed at.
SPLIT_PATH_PREFIX = "./images"


def _parse_split_ratios(split_ratios: str) -> tuple[int, int, int]:
    """Parse "85 10 5" (commas tolerated) into (train, val, test) percentages.

    Insists they sum to exactly 100 rather than normalizing whatever is
    given: a typo'd "80 10 5" almost certainly means a miscounted split, not
    a request for a 95%-of-the-data dataset, and silently rescaling it would
    hand back a train/val/test division nobody asked for.
    """
    parts = split_ratios.replace(",", " ").split()
    if len(parts) != 3:
        raise typer.BadParameter(
            f"Expected three values (train val test), e.g. '{DEFAULT_SPLIT_RATIOS}'; got {split_ratios!r}."
        )
    try:
        train_ratio, val_ratio, test_ratio = (int(part) for part in parts)
    except ValueError:
        raise typer.BadParameter(f"Split ratios must be whole percentages; got {split_ratios!r}.")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise typer.BadParameter(f"Split ratios cannot be negative; got {split_ratios!r}.")
    if train_ratio + val_ratio + test_ratio != 100:
        raise typer.BadParameter(
            f"Split ratios must sum to 100; {split_ratios!r} sums to {train_ratio + val_ratio + test_ratio}."
        )
    return train_ratio, val_ratio, test_ratio


def _read_pinned_test_names(pinned_file: Path) -> set[str]:
    """Filenames to force into the test split, one per line.

    Matched leniently -- against the exported ``<parent>_<stem><suffix>``
    name, the original filename, or the full recorded source path, with or
    without an extension. The list is typically written by hand or pasted
    from a previous split file, and which of those spellings someone has to
    hand is not worth making them care about. Blank lines and ``#`` comments
    are ignored.
    """
    names: set[str] = set()
    for line in pinned_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(line)
    return names


def _matches_pinned(image_path: Path, dest_filename: str, pinned: set[str]) -> bool:
    candidates = {
        dest_filename,
        Path(dest_filename).stem,
        image_path.name,
        image_path.stem,
        str(image_path),
        f"{SPLIT_PATH_PREFIX}/{dest_filename}",
    }
    return bool(candidates & pinned)


def _write_splits(
    image_paths: list[str],
    output_dir: Path,
    *,
    split_ratios: str,
    seed: int,
    pinned_test_file: Path | None,
) -> dict[str, int]:
    """Write train.txt/val.txt/test.txt of ``./images/<file>`` relative paths.

    Splits at the IMAGE level, never at the box level: two crops of the same
    frame landing on opposite sides of the train/val boundary leak the exact
    background, lighting and often the same individual animal across the
    split, which quietly inflates validation scores on benthic transect
    imagery where consecutive frames overlap heavily anyway.

    Seeded by default, so the same database, ratios and seed reproduce the
    same split -- re-exporting after relabelling a handful of ROIs should not
    silently reshuffle which images were held out, or every model trained
    before and after becomes incomparable.
    """
    train_ratio, val_ratio, test_ratio = _parse_split_ratios(split_ratios)

    # Sorted before shuffling: dict insertion order here follows whatever
    # order DuckDB returned rows in, which is not guaranteed stable, and
    # seeding a shuffle of an unstable order reproduces nothing.
    entries = sorted(
        ((Path(p), f"{disambiguated_stem(Path(p))}{Path(p).suffix}") for p in set(image_paths)),
        key=lambda entry: entry[1],
    )

    pinned = _read_pinned_test_names(pinned_test_file) if pinned_test_file else set()
    pinned_entries = [e for e in entries if _matches_pinned(e[0], e[1], pinned)]
    # Partitioned on the dest filename (unique by construction) rather than
    # `e not in pinned_entries`, which is a list scan per entry -- O(images x
    # pinned), and a survey-scale export against a few thousand pinned names
    # is exactly when that bites.
    pinned_filenames = {dest for _, dest in pinned_entries}
    remaining = [e for e in entries if e[1] not in pinned_filenames]

    if pinned and not pinned_entries:
        logger.warning(
            "None of the %d name(s) in %s matched an exported image -- the test split is "
            "an ordinary random sample.", len(pinned), pinned_test_file,
        )

    random.Random(seed).shuffle(remaining)

    total = len(entries)
    n_test = max(len(pinned_entries), int(total * test_ratio / 100))
    n_val = int(total * val_ratio / 100)

    test_entries = pinned_entries + remaining[: n_test - len(pinned_entries)]
    rest = remaining[n_test - len(pinned_entries) :]
    val_entries = rest[:n_val]
    # Train takes the remainder rather than its own computed count, so
    # integer truncation above can never silently drop images from the
    # dataset entirely -- every exported image lands in exactly one split.
    train_entries = rest[n_val:]

    counts = {}
    for split_name, split_entries in zip(SPLIT_NAMES, (train_entries, val_entries, test_entries)):
        split_path = output_dir / f"{split_name}.txt"
        lines = [f"{SPLIT_PATH_PREFIX}/{dest_filename}" for _, dest_filename in split_entries]
        split_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[split_name] = len(split_entries)

    logger.info(
        "Wrote train/val/test splits to %s -- %d train / %d val / %d test (requested %d/%d/%d%%, seed %d%s)",
        output_dir, counts["train"], counts["val"], counts["test"],
        train_ratio, val_ratio, test_ratio, seed,
        f", {len(pinned_entries)} pinned to test" if pinned_entries else "",
    )
    return counts


def _quote_yaml(name: str) -> str:
    """Single-quoted YAML scalar. A literal apostrophe is doubled, which is
    how YAML escapes one inside single quotes -- without this a taxon like
    ``Cuvier's beaked whale`` would close the quote early and produce a file
    that silently fails to parse at training time."""
    return "'" + name.replace("'", "''") + "'"


def _write_dataset_yaml(
    names: list[str], output_dir: Path, *, yaml_name: str, split_counts: dict[str, int]
) -> Path:
    """Write the Ultralytics dataset YAML (train/val/test paths, nc, names).

    ``nc`` and ``names`` are derived from the very same ``names`` list that
    assigned the class indices in the label files and was written to
    names.txt -- not recomputed from the database -- so the three cannot
    disagree about what class 17 is. That is the whole failure this guards
    against: a YAML whose name order differs from the indices in labels/
    trains every class against the wrong name and looks entirely normal
    while doing it.

    Split paths are written RELATIVE (bare ``train.txt``), with no ``path:``
    key. Ultralytics resolves them against the YAML's own directory when
    ``path`` is absent, so naming the YAML at training time is genuinely all
    that is needed and the dataset directory works unchanged wherever it is
    mounted -- exported under /Volumes/M3_ML on macOS, read from /mnt/M3_ML
    on the Linux trainer, with nothing to rewrite in between. Same reasoning
    as the ``./images/<file>`` lines inside the split files themselves,
    which Ultralytics likewise resolves relative to the split file.

    An EMPTY split is left out entirely rather than written as a key
    pointing at an empty file: with ratios like '85 15 0' there is no test
    set, and a ``test:`` line promising one would fail at the moment someone
    ran evaluation against it, long after the export looked fine.
    """
    lines = ["# train and val data"]
    lines += [f"{split}: {split}.txt" for split in SPLIT_NAMES if split_counts.get(split, 0) > 0]
    lines += ["", "# number of classes", f"nc: {len(names)}", "", "# class names"]

    if names:
        quoted = [_quote_yaml(name) for name in names]
        # Matches the reference layout: the list opens on the `names:` line
        # and every subsequent entry sits on its own line, closing on the
        # last one. Valid YAML flow sequence, and it keeps a 50-line class
        # list reviewable as a diff.
        body = ",\n".join(quoted)
        lines.append(f"names: [{body}]")
    else:
        lines.append("names: []")

    yaml_path = output_dir / yaml_name
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def _write_names(names: list[str], output_dir: Path) -> Path:
    names_path = output_dir / "names.txt"
    names_path.write_text("\n".join(names) + ("\n" if names else ""))
    return names_path


@app.command()
def export_yolo(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    output_dir: str = typer.Argument(
        ...,
        help="Directory to write labels/, names.txt, train/val/test.txt, and the image manifest/copy script into.",
    ),
    splits: bool = typer.Option(
        True, "--splits/--no-splits", help="Also write train.txt/val.txt/test.txt image lists."
    ),
    split_ratios: str = typer.Option(
        DEFAULT_SPLIT_RATIOS,
        help="Train/val/test percentages, e.g. '85 10 5'. Must sum to 100.",
    ),
    split_seed: int = typer.Option(
        42,
        help="Seed for the split shuffle. The same database + ratios + seed always produce the same "
             "split, so re-exporting after relabelling doesn't reshuffle what was held out.",
    ),
    test_images_file: str = typer.Option(
        None,
        help="Optional file of image filenames (one per line) to force into the test split -- e.g. a "
             "fixed benchmark set you want held out of training across every export.",
    ),
    dataset_yaml: bool = typer.Option(
        True, "--yaml/--no-yaml",
        help="Also write the Ultralytics dataset YAML (train/val/test paths, nc, names). "
             "Requires --splits, since it points at the split files.",
    ),
    yaml_name: str = typer.Option(
        None,
        help="Filename for the dataset YAML. Defaults to '<output_dir name>.yaml'.",
    ),
) -> None:
    """Export curated labels to YOLO-format label files, plus a names file.

    Images are located via the path recorded at detection time -- no
    separate image_dir argument needed (same convention as `export voc`).

    Writes a complete, trainable dataset skeleton: labels/, names.txt,
    train/val/test.txt (relative `./images/...` paths), the Ultralytics
    dataset YAML, plus image_manifest.csv and copy_images.py to fetch the
    matching images:

        mbariml export yolo predictions.duckdb dataset/
        python3 dataset/copy_images.py --dest dataset/images

    The YAML's `nc`/`names` come from the same list that assigned the class
    indices in labels/, so they cannot disagree. Its split paths are
    relative to the YAML itself, so naming it at training time is all that
    is needed wherever the dataset is mounted.
    """
    output_dir_path = Path(output_dir)

    pinned_test_file = Path(test_images_file) if test_images_file else None
    if pinned_test_file and not pinned_test_file.exists():
        raise typer.BadParameter(f"--test-images-file not found: {pinned_test_file}")
    # Parsed before any work so a bad ratio string fails immediately rather
    # than after writing a directory of label files.
    if splits:
        _parse_split_ratios(split_ratios)

    with db.connect(db_path) as conn:
        db.require_verified_column(conn, db_path)
        written, names, image_paths = _export_labels(conn, output_dir_path)

    names_path = _write_names(names, output_dir_path)
    write_image_manifest_and_script(image_paths, output_dir_path, export_name="yolo")

    logger.info("Wrote %d YOLO label file(s) to %s", written, output_dir_path / "labels")
    logger.info("Wrote %d distinct label(s) to %s", len(names), names_path)

    split_counts: dict[str, int] = {}
    if splits:
        split_counts = _write_splits(
            image_paths,
            output_dir_path,
            split_ratios=split_ratios,
            seed=split_seed,
            pinned_test_file=pinned_test_file,
        )

    if dataset_yaml and splits:
        if not split_counts.get("val"):
            logger.warning(
                "The val split is empty, so the dataset YAML has no `val:` key -- Ultralytics "
                "needs one to train. Give val a non-zero share in --split-ratios."
            )
        yaml_path = _write_dataset_yaml(
            names,
            output_dir_path,
            yaml_name=yaml_name or f"{output_dir_path.resolve().name}.yaml",
            split_counts=split_counts,
        )
        listed = ", ".join(s for s in SPLIT_NAMES if split_counts.get(s, 0) > 0)
        logger.info("Wrote dataset YAML (nc: %d, splits: %s) to %s", len(names), listed, yaml_path)
    elif dataset_yaml:
        logger.warning(
            "Skipped the dataset YAML: it points at train/val/test.txt, which --no-splits did "
            "not write. Re-run with --splits, or pass --no-yaml to silence this."
        )


if __name__ == "__main__":
    app()
