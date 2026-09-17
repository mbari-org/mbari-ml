"""Export *.id identification sidecar files into the mission directory structure.

For every source image that has at least one curated identification (every
verified localization, named ``new_label`` where the reviewer retyped it and
the original detector ``label`` where they confirmed it unchanged, excluding
``noise`` -- ``mbariml.db.curated_where``, the same rule as ``export
voc``/``export yolo``), writes a ``<image_stem>.id`` file *next to that
image* (wherever it actually lives on disk -- this walks whatever path was
recorded in ``image_path`` at detection time, so it naturally follows the
mission's own directory structure without needing a separate ``image_dir``
argument).

``--include-unverified`` also writes the raw, un-reviewed detections,
named with the detector's own label -- off by default so the sidecars agree
with what ``export yolo``/``voc`` write.

``--output-dir`` collects them into one directory instead, for a read-only
survey volume or a handoff without the imagery. Names there are
``disambiguated_stem``-based, since flattening a nested mission tree is
precisely when two dives' same-named images would overwrite each other.

Each bounding box is written as a 4-vertex polygon (top-left, top-right,
bottom-right, bottom-left), with pixel coordinates filled in immediately and
lon/lat/depth left as ``0.0`` placeholders -- per-vertex geolocation is
expected to be appended later by a separate process once navigation data is
available, by re-parsing/updating these same files.

One of the Emit-phase exports, alongside ``export voc``/``export yolo``/
``export html``, but not part of ``mbariml run``'s scriptable chain -- run
it directly whenever you want fresh sidecar files for a curated database.
"""

from __future__ import annotations

import csv
import getpass
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from mbariml import __version__
from mbariml import db
from mbariml.image_naming import disambiguated_stem
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Export *.id identification sidecar files next to each source image.")
logger = get_logger(__name__)


def _resolve_model_description(conn, model_override: Optional[str]) -> str:
    if model_override:
        return model_override
    row = conn.execute("SELECT model_path FROM run_info ORDER BY detected_at DESC LIMIT 1").fetchone()
    return row[0] if row and row[0] else "unknown"


def _image_size(image_path: Path) -> Optional[tuple[int, int]]:
    """``(width, height)`` of an image, or None if it can't be read.

    PIL's ``Image.open`` parses the header and stops, so this costs a header
    read rather than a full decode -- measured on this survey's 1936x1456
    TIFFs at 6.3 ms/image against cv2.imread's 43.9 ms, i.e. ~6s rather than
    ~44s across a 995-image export. That matters because this is the only
    reason `export id` touches the imagery at all; everything else it writes
    comes from the database.

    Returns None rather than raising for an image that has moved or won't
    open: the sidecar is still worth writing (all of its pixel geometry comes
    from the database, not the file), it just can't state the frame size.
    """
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None


COLUMNS = [
    "index", "label", "confidence",
    "center_x", "center_y", "lon", "lat", "depth",
    "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y",
]


def _row_values(
    index: int, label: str, confidence: float,
    x_min: float, y_min: float, x_max: float, y_max: float,
) -> list:
    """One identification as the flat list of values named by ``COLUMNS``.

    The box's float coordinates are rounded to whole pixels ONCE here, and
    the center is then the integer midpoint of those same rounded corners --
    not a separately-rounded midpoint of the original floats. Those two
    differ by a pixel surprisingly often: on a real 35,492-identification
    export, 222 rows. Either answer is defensible alone, but a file whose
    stated center disagrees with the midpoint of the corners printed beside
    it is the exact "two consumers compute different centers" problem the
    center exists to remove -- so anyone who derives it from the corners
    gets the value written here.

    Integer ``//`` rather than ``round()`` on the midpoint, so an odd span
    resolves the same way every time. Python's ``round()`` is
    round-half-to-even, which would make the tie-break depend on the
    coordinate's parity.

    lon/lat/depth appear ONCE, next to the center, rather than once per
    vertex. They describe where the observation is, and an observation has
    one position; carrying four more copies meant every row shipped five
    identical ``0.0,0.0,0.0`` triples for a single unknown.
    """
    left, top = int(round(x_min)), int(round(y_min))
    right, bottom = int(round(x_max)), int(round(y_max))
    return [
        index, label, f"{confidence:.4f}",
        (left + right) // 2, (top + bottom) // 2,  # center_x, center_y
        0.0, 0.0, 0.0,                             # lon, lat, depth -- placeholders
        left, top,       # TL
        right, top,      # TR
        right, bottom,   # BR
        left, bottom,    # BL
    ]


def _build_id_file_content(
    image_path: str, model_desc: str, username: str, generated_at: str,
    detections: list[tuple], image_size: Optional[tuple[int, int]] = None,
) -> str:
    """One .id file's full text: commented header, then one row per identification.

    The header carries a field-by-field legend rather than the single
    ``vertices(TL,TR,BR,BL as px_x,px_y,lon,lat,depth)`` line it used to.
    That line was trying to describe a nested structure -- four corners, each
    itself five values -- in one parenthesis, so it read as one flat list of
    nine things and left the reader to guess where a corner ended. Whoever
    writes the navigation merge reads this header to find out what the
    columns are; spelling it out costs nine comment lines once per file.

    ``source_image`` is the full recorded path, not just the basename. With
    --output-dir the sidecars no longer sit beside their imagery, so the
    basename alone would not say which dive an identification came from --
    and a survey has many directories holding an image of the same name.
    """
    # Corner coordinates are box EDGES, not pixel indices: a box flush against
    # the right side of the frame has px_x == image_width, one PAST the last
    # column. That is not an error and must not be "corrected" -- it is what
    # makes width == right - left exact (the same arithmetic `export yolo`
    # uses), and on this survey 422 boxes touch the right edge and 13 the
    # bottom. Saying "px_x is 0..W-1" here would have declared 884 perfectly
    # good coordinates out of range, so the legend states the real rule.
    if image_size:
        width, height = image_size
        bounds_lines = [
            f"#                Corner coordinates are box EDGES: for this {width}x{height} image",
            f"#                tl_x/tr_x/br_x/bl_x span 0..{width} and the _y values 0..{height},",
            f"#                so a box flush against the right side has tr_x = {width} -- one",
            f"#                past the last column ({width - 1}). That is what makes the width",
            f"#                arithmetic exact, and must not be 'corrected' to {width - 1}.",
            f"#                center_x/center_y are true pixels and stay within 0..{width - 1} / 0..{height - 1}.",
        ]
    else:
        bounds_lines = [
            "#                Corner coordinates are box EDGES, so they may sit one past the",
            "#                last column/row -- that is what makes the width arithmetic exact.",
            "#                center_x/center_y are true pixels, always inside the frame.",
            "#                The frame size could not be read, so image_width/image_height",
            "#                above say 'unknown'.",
        ]

    lines = [
        "# mbariml identification file",
        f"# generator: mbariml v{__version__}",
        f"# generated_by: {username}",
        f"# generated_at: {generated_at}",
        f"# model: {model_desc}",
        f"# source_image: {image_path}",
        f"# image_width: {image_size[0] if image_size else 'unknown'}",
        f"# image_height: {image_size[1] if image_size else 'unknown'}",
        f"# count: {len(detections)}",
        "#",
        "# One identification per row below, with these fields:",
        "#   index        0-based position of this identification within this file",
        "#   label        taxon name (may contain spaces)",
        "#   confidence   detector confidence, 0.0-1.0; 1.0000 means a human drew the box",
        "#   center_x     the observation's position: the box's center pixel, which",
        "#   center_y     is exactly the integer midpoint of the tl/br corners below",
        "#   lon lat      geolocation OF THAT CENTER POINT (see below)",
        "#   depth",
        "#   tl_* tr_*    the same box as four corners, in this order: top-left,",
        "#   br_* bl_*    top-right, bottom-right, bottom-left. The box covers",
        "#                columns tl_x .. tr_x-1 and rows tl_y .. bl_y-1; its",
        "#                width is tr_x - tl_x and its height is bl_y - tl_y.",
        "#",
        "# Every value is comma-separated -- the rows below are plain CSV, and the",
        "# line just above the first one names the columns. Labels are written with",
        "# standard CSV quoting, so a label containing a comma would be quoted.",
        "#",
        "#   *_x          COLUMN in the source image, counted from the left edge",
        "#   *_y          ROW in the source image, counted from the top edge",
        "#                Both are 0-based. A pixel is addressed by this PAIR --",
        "#                there is no single pixel number.",
        *bounds_lines,
        "#   lon,lat      decimal degrees; written as 0.0 placeholders here",
        "#   depth        meters, positive down; written as a 0.0 placeholder here",
        "#",
        "# lon/lat/depth describe where the OBSERVATION is -- they belong to the",
        "# center point, not to each corner -- and are filled in later from",
        "# navigation data by re-parsing and rewriting these same files.",
        "#",
        "# " + ",".join(COLUMNS),
    ]
    # csv.writer, not ",".join: taxon names carry spaces, and while none in
    # this survey's 51 labels contains a comma, one that did would silently
    # add a column and shift every coordinate after it. The writer quotes
    # only when it has to, so with ordinary labels the output is byte-for-byte
    # what a plain join would produce, and a comma in a future label degrades
    # to standard CSV quoting instead of corrupting the row.
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for i, (label, confidence, x_min, y_min, x_max, y_max) in enumerate(detections):
        writer.writerow(_row_values(i, label, confidence, x_min, y_min, x_max, y_max))
    return "\n".join(lines) + "\n" + buffer.getvalue()


@app.command()
def export_ids(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    output_dir: Optional[str] = typer.Option(
        None,
        help="Write the .id files into this directory instead of next to each source image. "
             "Filenames are disambiguated by parent directory, so two dives sharing an image "
             "name don't collide. Default (unset) writes each sidecar beside its own image.",
    ),
    include_unverified: bool = typer.Option(
        False, "--include-unverified/--verified-only",
        help="Also write identifications nobody has verified yet, named with the raw detector "
             "label. Off by default, so the sidecars match what `export yolo`/`voc` write; turn "
             "it on to emit everything the detector found. [default: verified-only]",
    ),
    model: Optional[str] = typer.Option(
        None, help="Model identifier to record in each file's header. Defaults to what "
        "`mbariml detect` recorded for this database, or 'unknown' if that's not available."
    ),
) -> None:
    """Write a *.id sidecar file for each source image with at least one curated identification.

    Writes each file next to its own source image by default. Pass
    --output-dir to collect them all in one directory instead -- e.g. when
    the survey volume is read-only, or when the sidecars are being handed
    off somewhere without the imagery.
    """
    with db.connect(db_path, must_exist=True) as conn:
        model_desc = _resolve_model_description(conn, model)

        if not include_unverified:
            db.require_verified_column(conn, db_path)
        rows = conn.execute(
            f"""
            SELECT image_path, {db.EFFECTIVE_LABEL_SQL}, confidence, x_min, y_min, x_max, y_max
            FROM predictions
            {db.curated_where(require_verified=not include_unverified)}
            """
        ).fetchall()

    if not rows:
        logger.warning(
            "No identifications found (no %s, non-'noise' localizations); no .id files written.%s",
            "verified" if not include_unverified else "usable",
            "" if include_unverified else
            " Verify some ROIs in `mbariml review` first, or pass --include-unverified to write "
            "the raw detections.",
        )
        return

    grouped: dict[str, list[tuple]] = {}
    for image_path, label, confidence, x_min, y_min, x_max, y_max in rows:
        grouped.setdefault(image_path, []).append((label, confidence, x_min, y_min, x_max, y_max))

    username = getpass.getuser()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    output_dir_path = Path(output_dir) if output_dir else None
    if output_dir_path:
        output_dir_path.mkdir(parents=True, exist_ok=True)

    written, failed, unsized = 0, 0, 0
    for image_path, detections in tqdm(grouped.items(), desc="Writing .id files"):
        image_path_obj = Path(image_path)
        if output_dir_path:
            # disambiguated_stem, not the bare stem: flattening a mission's
            # nested per-dive directories into one output directory is
            # exactly when two dives' identically-named images collide, and
            # the second would silently overwrite the first's identifications.
            # Same naming every other export uses for the same reason.
            id_path = output_dir_path / f"{disambiguated_stem(image_path_obj)}.id"
        else:
            id_path = image_path_obj.with_suffix(".id")
        image_size = _image_size(image_path_obj)
        if image_size is None:
            unsized += 1
        try:
            id_path.write_text(
                _build_id_file_content(
                    str(image_path_obj), model_desc, username, generated_at, detections, image_size
                )
            )
            written += 1
        except OSError:
            failed += 1
            logger.exception("Failed to write %s", id_path)

    destination = str(output_dir_path) if output_dir_path else "next to each source image"
    logger.info("Wrote %d .id file(s) to %s (%s; model: %s, user: %s).",
                written, destination,
                "verified + unverified" if include_unverified else "verified only",
                model_desc, username)
    if unsized:
        logger.warning(
            "%d image(s) could not be opened to read their frame size, so those sidecars say "
            "image_width/image_height: unknown. Their identifications are unaffected -- all box "
            "geometry comes from the database, not the image.", unsized,
        )
    if failed:
        logger.error("%d .id file(s) failed to write -- see tracebacks above.", failed)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
