"""Shared machinery for ``mbariml import {yolo,voc}``.

The two import formats differ only in how a box is read off disk; everything
after that -- finding the image a label file belongs to, cropping each box as
an ROI, computing sharpness, and inserting curation-schema rows -- is the same
and lives here. Rows go in through ``infer_images.insert_rows``, the very
insert every image-ingest row uses, so an imported localization is
indistinguishable from a detected one downstream: review can edit or delete
it, ``embed``/``cluster`` pick it up, and every export emits it.

Imported boxes are VERIFIED by default. An existing training set is, almost
always, human-labeled ground truth -- and every export selects only verified
rows (``mbariml.db.curated_where``), so importing them unverified would make
a dataset that round-trips straight back out as nothing. ``--unverified`` is
there for importing someone else's model predictions, which deserve review.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2

from mbariml import db
from mbariml.image_quality import compute_sharpness
from mbariml.images import collect_images
from mbariml.logging_utils import get_logger
from mbariml.steps.infer_images import insert_rows

logger = get_logger(__name__)

# Images per database flush. Same reasoning as infer_images committing after
# every batch: an interrupted import keeps everything up to the last flush.
FLUSH_EVERY = 200


@dataclass
class ImportedBox:
    """One annotation, in absolute pixel coordinates of its image."""

    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_id: Optional[int] = None
    confidence: Optional[float] = None


@dataclass
class ImportedImage:
    """One annotation file, resolved (or not) to its image on disk.

    ``image_path`` is None when no matching image could be found. ``boxes``
    holds boxes in pixel coordinates; ``normalized`` boxes (YOLO) are scaled
    against the image's real dimensions once it has been read.
    """

    annotation_path: Path
    image_path: Optional[Path]
    boxes: list[ImportedBox]
    normalized: bool = False
    # (width, height) the annotation claims, VOC only -- checked against the
    # real image so a box drawn on a resized copy doesn't import silently wrong.
    declared_size: Optional[tuple[int, int]] = None


class ImageIndex:
    """Every image under a directory, findable by the ways annotation files
    refer to their images.

    Built once up front (one recursive scan) rather than probing the
    filesystem per annotation file. Lookups prefer the most specific match --
    the same relative path -- and only fall back to a bare filename or stem
    when that is unambiguous: two dives' ``img_0001.jpg`` must never be
    silently paired with each other's labels.
    """

    def __init__(self, images_dir: str | Path):
        self.root = Path(images_dir).resolve()
        self.paths = [p.resolve() for p in collect_images(self.root)]
        self._by_rel_stem: dict[str, Path] = {}
        self._by_name: dict[str, list[Path]] = {}
        self._by_stem: dict[str, list[Path]] = {}
        for path in self.paths:
            rel = path.relative_to(self.root)
            self._by_rel_stem[rel.with_suffix("").as_posix()] = path
            self._by_name.setdefault(path.name, []).append(path)
            self._by_stem.setdefault(path.stem, []).append(path)
        # Annotation files left unpaired because only an ambiguous
        # filename/stem match was available.
        self.ambiguous = 0

    def find(self, rel_stem: str, filename: Optional[str] = None, folder: Optional[str] = None) -> Optional[Path]:
        """The image for an annotation file at ``rel_stem`` (its path relative
        to the annotation root, without suffix), optionally naming the image
        file it describes and that file's parent directory (VOC's
        ``<filename>``/``<folder>``)."""
        candidate_lists = []
        if filename:
            exact = self._by_rel_stem.get((Path(rel_stem).parent / Path(filename).stem).as_posix())
            if exact is not None and exact.name == filename:
                return exact
            by_name = self._by_name.get(filename) or []
            # <folder> breaks a filename tie: `export voc` flattens
            # train/a.jpg and val/a.jpg to train_a.xml/val_a.xml, both
            # saying <filename>a.jpg</filename>, with <folder> telling them apart.
            if folder and len(by_name) > 1:
                in_folder = [p for p in by_name if p.parent.name == folder]
                if in_folder:
                    by_name = in_folder
            candidate_lists.append(by_name)
        exact = self._by_rel_stem.get(rel_stem)
        if exact is not None:
            return exact
        candidate_lists.append(self._by_stem.get(Path(rel_stem).name))

        saw_ambiguous = False
        for candidates in candidate_lists:
            if candidates and len(candidates) == 1:
                return candidates[0]
            saw_ambiguous |= bool(candidates)
        if saw_ambiguous:
            self.ambiguous += 1
        return None


def _to_pixel_box(box: ImportedBox, width: int, height: int, normalized: bool) -> Optional[tuple[int, int, int, int]]:
    """Integer pixel box clamped to the image, or None if nothing is left.

    Integer like ``infer images`` stores, so the stored box and the cropped
    ROI describe exactly the same pixels.
    """
    x_min, y_min, x_max, y_max = box.x_min, box.y_min, box.x_max, box.y_max
    if normalized:
        x_min, x_max = x_min * width, x_max * width
        y_min, y_max = y_min * height, y_max * height
    x_min, x_max = sorted((x_min, x_max))
    y_min, y_max = sorted((y_min, y_max))
    x_min = max(0, min(width, round(x_min)))
    x_max = max(0, min(width, round(x_max)))
    y_min = max(0, min(height, round(y_min)))
    y_max = max(0, min(height, round(y_max)))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def import_annotations(
    images: Iterable[ImportedImage],
    output_dir: str | Path,
    *,
    source_description: str,
    verified: bool = True,
    skip_existing: bool = True,
    names: Optional[list[str]] = None,
) -> Path:
    """Crop and insert every box in ``images`` into
    OUTPUT_DIR/yolo_predictions.duckdb (created if needed, appended to if
    not). Returns the database path."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    db_path = output_dir_path / "yolo_predictions.duckdb"

    counts = dict(
        images=0, boxes=0, no_image=0, unreadable=0, already_present=0,
        degenerate=0, size_mismatch=0, empty=0,
    )
    labels_seen: set[str] = set()

    with db.init_curation_db(db_path) as conn:
        existing = {row[0] for row in conn.execute("SELECT DISTINCT image_path FROM predictions").fetchall()}
        next_id = db.next_free_id(conn)
        pending: list[tuple] = []
        pending_images = 0

        for item in images:
            if item.image_path is None:
                counts["no_image"] += 1
                logger.warning("No image found for %s; skipped", item.annotation_path)
                continue
            if not item.boxes:
                counts["empty"] += 1
                continue
            image_path_str = str(item.image_path)
            if skip_existing and image_path_str in existing:
                counts["already_present"] += 1
                continue

            image = cv2.imread(image_path_str, cv2.IMREAD_COLOR)
            if image is None:
                counts["unreadable"] += 1
                logger.warning("Could not read image %s; skipped", item.image_path)
                continue
            height, width = image.shape[:2]

            if item.declared_size and item.declared_size != (width, height) and all(item.declared_size):
                counts["size_mismatch"] += 1
                logger.warning(
                    "%s says the image is %dx%d but %s is %dx%d -- boxes imported as written, check them in review",
                    item.annotation_path, *item.declared_size, item.image_path.name, width, height,
                )

            image_rows = 0
            for box in item.boxes:
                pixel_box = _to_pixel_box(box, width, height, item.normalized)
                if pixel_box is None:
                    counts["degenerate"] += 1
                    continue
                x_min, y_min, x_max, y_max = pixel_box
                roi = image[y_min:y_max, x_min:x_max]
                ok, roi_encoded = cv2.imencode(".jpg", roi)
                if not ok:
                    counts["degenerate"] += 1
                    continue
                pending.append((
                    next_id, item.image_path.name, image_path_str, next_id,
                    float(x_min), float(y_min), float(x_max), float(y_max),
                    box.class_id,
                    1.0 if box.confidence is None else float(box.confidence),
                    box.label,
                    None,  # embedding, filled in by `mbariml embed`
                    None,  # new_label: the imported name IS the original label
                    roi_encoded.tobytes(),
                    compute_sharpness(roi),
                    1 if verified else 0,
                ))
                next_id += 1
                image_rows += 1
                labels_seen.add(box.label)

            if image_rows:
                existing.add(image_path_str)
                counts["images"] += 1
                counts["boxes"] += image_rows
                pending_images += 1
            if pending_images >= FLUSH_EVERY:
                insert_rows(conn, pending)
                logger.info("Imported %d box(es) from %d image(s) so far", counts["boxes"], counts["images"])
                pending, pending_images = [], 0

        insert_rows(conn, pending)

        # Only fill in provenance where there is none: overwriting a real
        # model path recorded by `infer` would make `export id` attribute that
        # model's detections to this import.
        if not db.row_count(conn, "run_info"):
            conn.execute("INSERT INTO run_info VALUES (?, CURRENT_TIMESTAMP)", (source_description,))
        final_count = db.row_count(conn)

    # names.txt only if absent -- `infer` writes the model's class list there,
    # and an import appended to that database must not replace it with its own.
    names_path = output_dir_path / "names.txt"
    if not names_path.exists():
        names_to_write = names if names is not None else sorted(labels_seen)
        names_path.write_text("\n".join(names_to_write) + ("\n" if names_to_write else ""))

    logger.info(
        "Imported %d box(es) from %d image(s), %d distinct label(s), %s; database now holds %d row(s): %s",
        counts["boxes"], counts["images"], len(labels_seen),
        "verified" if verified else "unverified", final_count, db_path,
    )
    if counts["empty"]:
        logger.info("%d annotation file(s) had no boxes (background images -- nothing to store).", counts["empty"])
    if counts["already_present"]:
        logger.warning(
            "%d image(s) already in the database were skipped (pass --no-skip-existing to import their "
            "boxes anyway, which will duplicate any already there).", counts["already_present"],
        )
    if counts["no_image"]:
        logger.warning("%d annotation file(s) had no matching image and were skipped.", counts["no_image"])
    if counts["unreadable"]:
        logger.warning("%d image(s) could not be read and were skipped.", counts["unreadable"])
    if counts["degenerate"]:
        logger.warning("%d box(es) had zero area (after clipping to the image) and were skipped.", counts["degenerate"])
    if counts["size_mismatch"]:
        logger.warning("%d annotation file(s) declared a different image size than the image on disk.",
                       counts["size_mismatch"])
    if counts["boxes"]:
        logger.info(
            "Ready to continue with: mbariml embed %s | mbariml review %s",
            db_path, db_path,
        )
    return db_path
