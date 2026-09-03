"""`export html`: generate a paginated HTML gallery of images and their
labeled crops -- a quick visual QA pass in a browser.

One of the Emit-phase exports, alongside `export voc`/`export yolo`/
`export id`.
"""

from __future__ import annotations

import html
import os
from collections import defaultdict
from pathlib import Path

import cv2
import typer
from tqdm import tqdm

from mbariml import db
from mbariml.logging_utils import get_logger

app = typer.Typer(help="Generate a paginated HTML gallery of images and their labeled crops.")
logger = get_logger(__name__)

CROP_THUMB_SIZE = 60


def _label_expr(conn) -> str:
    """Prefer the curated new_label, falling back to the raw YOLO label --
    per row, not per database. Every database has a new_label column now
    (every ingest command writes the same curation schema), so a database
    fresh out of `mbariml infer images`/`infer video` -- not yet reviewed -- has
    new_label NULL on every row; without this fallback, an HTML gallery
    generated before curation would show "None" instead of the model's
    actual prediction for every single crop."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info('predictions')").fetchall()}
    return "COALESCE(new_label, label)" if "new_label" in columns else "label"


def _process_images_and_crops(conn, output_dir: Path) -> list[dict]:
    """Bug fixed here: this used to group by bare image_name and reconstruct
    the path as image_dir/image_name. For a mission with nested per-dive
    subdirectories, two images with the same filename in different dives
    collided into one key and both got mapped onto whichever single flat
    path happened to exist, silently merging both dives' crops onto one
    image. Grouping is now by the full image_path already recorded at
    detection time, and output filenames are disambiguated by parent
    directory name so two dives' same-named images don't collide there too.
    """
    label_expr = _label_expr(conn)
    rows = conn.execute(
        f"SELECT image_path, x_min, y_min, x_max, y_max, {label_expr} AS label FROM predictions"
    ).fetchall()

    images_dir = output_dir / "images"
    crops_dir = output_dir / "crops"
    images_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)

    seen_crops: set[tuple] = set()
    image_data: dict[str, dict] = {}

    for image_path_str, crops in tqdm(grouped.items(), desc="Processing images and crops"):
        image_path = Path(image_path_str)
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Could not read image (skipping): %s", image_path)
            continue

        unique_stem = f"{image_path.parent.name}_{image_path.stem}"
        output_image_path = images_dir / f"{unique_stem}.jpg"
        annotated = image.copy()

        for _path, x_min, y_min, x_max, y_max, label in crops:
            crop_key = (image_path_str, x_min, y_min, x_max, y_max, label)
            if crop_key in seen_crops:
                continue

            x_min_i, y_min_i, x_max_i, y_max_i = int(x_min), int(y_min), int(x_max), int(y_max)
            cv2.rectangle(annotated, (x_min_i, y_min_i), (x_max_i, y_max_i), (0, 255, 0), 2)
            cv2.putText(annotated, str(label), (x_min_i, y_min_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            crop = image[y_min_i:y_max_i, x_min_i:x_max_i]
            if crop.size == 0:
                logger.warning("Empty crop for %s; skipping", crop_key)
                continue
            crop = cv2.resize(crop, (CROP_THUMB_SIZE, CROP_THUMB_SIZE))

            crop_output_path = crops_dir / f"{unique_stem}_{label}_{x_min_i}_{y_min_i}_{x_max_i}_{y_max_i}.jpg"
            cv2.imwrite(str(crop_output_path), crop)

            entry = image_data.setdefault(image_path_str, {
                "full_image": os.path.relpath(output_image_path, output_dir),
                "image_name": image_path.name,
                "crops": [],
            })
            entry["crops"].append({
                "label": label,
                "crop_path": os.path.relpath(crop_output_path, output_dir),
            })
            seen_crops.add(crop_key)

        cv2.imwrite(str(output_image_path), annotated)

    return list(image_data.values())


def _render_page(page_data: list[dict], page: int, total_pages: int, folder_name: str) -> str:
    rows_html = []
    for row in page_data:
        crops_html = "".join(
            f'<div class="crop"><img src="{html.escape(c["crop_path"])}" alt="Crop">'
            f'<div>{html.escape(str(c["label"]))}</div></div>'
            for c in row["crops"]
        )
        rows_html.append(f'''
            <div class="image-row">
                <div class="main-image">
                    <img src="{html.escape(row["full_image"])}" alt="{html.escape(row["image_name"])}" />
                    <div>{html.escape(row["image_name"])}</div>
                </div>
                <div class="crops-grid">{crops_html}</div>
            </div>
        ''')

    nav_links = []
    if page > 1:
        nav_links.append(f'<a href="{folder_name}_page_{page - 1}.html">Previous</a>')
    for i in range(1, total_pages + 1):
        nav_links.append(f'<span class="active">{i}</span>' if i == page else f'<a href="{folder_name}_page_{i}.html">{i}</a>')
    if page < total_pages:
        nav_links.append(f'<a href="{folder_name}_page_{page + 1}.html">Next</a>')
    navigation_html = " | ".join(nav_links)

    return f'''
    <html>
      <head>
        <title>{html.escape(folder_name)} - Page {page}</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 20px; }}
          .image-row {{ display: flex; margin-bottom: 40px; padding: 10px; border: 1px solid #ddd; }}
          .main-image img {{ max-height: 375px; margin-right: 20px; }}
          .crops-grid {{ display: flex; gap: 10px; flex-wrap: wrap; }}
          .crop img {{ width: {CROP_THUMB_SIZE}px; height: {CROP_THUMB_SIZE}px; }}
          .crop div {{ text-align: center; font-size: 12px; margin-top: 5px; }}
          .pagination {{ text-align: center; margin: 20px 0; }}
          .pagination a {{ margin: 0 5px; text-decoration: none; color: blue; }}
          .pagination a:hover {{ text-decoration: underline; }}
          .pagination .active {{ font-weight: bold; text-decoration: underline; }}
        </style>
      </head>
      <body>
        <h1>{html.escape(folder_name)} - Page {page}</h1>
        <div class="pagination">{navigation_html}</div>
        <div id="image-container">{"".join(rows_html)}</div>
        <div class="pagination">{navigation_html}</div>
      </body>
    </html>
    '''


def _save_html_pages(data: list[dict], output_dir: Path, folder_name: str, items_per_page: int) -> None:
    total_pages = max(1, (len(data) + items_per_page - 1) // items_per_page)
    for page in tqdm(range(1, total_pages + 1), desc="Generating HTML pages"):
        start = (page - 1) * items_per_page
        page_html = _render_page(data[start:start + items_per_page], page, total_pages, folder_name)
        output_file = output_dir / f"{folder_name}_page_{page}.html"
        output_file.write_text(page_html)
    logger.info("Wrote %d HTML page(s) to %s", total_pages, output_dir)


@app.command()
def generate_html(
    db_path: str = typer.Argument(..., help="Path to the DuckDB database."),
    output_dir: str = typer.Argument(..., help="Directory to save the generated HTML files."),
    items_per_page: int = typer.Option(250, help="Number of items per HTML page."),
) -> None:
    """Generate a paginated HTML gallery of images and their labeled crops.

    Images are located via the path recorded at detection time -- no
    separate image directory argument needed (removed in v0.3.0; it was a
    source of the image-name-collision bug described above).
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    folder_name = output_dir_path.name

    with db.connect(db_path) as conn:
        data = _process_images_and_crops(conn, output_dir_path)

    if not data:
        logger.warning("No images could be processed; no HTML pages were generated.")
        return

    _save_html_pages(data, output_dir_path, folder_name, items_per_page)


if __name__ == "__main__":
    app()
