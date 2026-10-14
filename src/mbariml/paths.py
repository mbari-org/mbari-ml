"""Finding a source image again after its folder has moved.

``image_path`` is recorded absolute at import/inference time, so `review` and
the exports work from any directory. The cost is that moving or renaming the
project folder (or copying it to another machine) breaks every path at once,
even though the frames are usually still right beside the database: `infer
video` writes them to ``<output_dir>/frames``, next to the ``.duckdb``. Before
this, review drew its tiles from the stored crops and so *looked* fine, while
the full frame, box edits, SAM3 and colour correction silently had no image,
and the exports skipped every frame as missing.

``ImageLocator`` keeps a path that still exists as is. For one that doesn't,
it looks under the database's folder for the path's trailing components:
``/old/place/greek_video_test/frames/x.jpg`` is tried as ``<db dir>/old/
place/greek_video_test/frames/x.jpg``, then ``<db dir>/place/...``, and so
on down to ``<db dir>/frames/x.jpg`` and ``<db dir>/x.jpg``. The longest
match wins, so two dives' same-named frames in sibling folders stay apart.
Each stored directory is searched once and the answer remembered, so a moved
folder of thousands of frames costs a few stat calls, not a search per image.
Nothing is written back to the database.
"""

from __future__ import annotations

import threading
from pathlib import Path

from mbariml.logging_utils import get_logger

logger = get_logger(__name__)


class ImageLocator:
    def __init__(self, db_path: str | Path):
        self.root = Path(db_path).expanduser().resolve().parent
        self._moved: dict[Path, Path | None] = {}  # stored directory -> where it is now
        self._lock = threading.Lock()

    def locate(self, stored: str) -> str:
        """*stored* if it exists or can't be found, else where it is now."""
        if not stored:
            return stored
        path = Path(stored)
        if path.is_file():
            return stored
        with self._lock:
            if path.parent not in self._moved:
                self._moved[path.parent] = self._search(path)
            new_dir = self._moved[path.parent]
        if new_dir is None:
            return stored
        candidate = new_dir / path.name
        return str(candidate) if candidate.is_file() else stored

    def _search(self, path: Path) -> Path | None:
        dirs = path.parent.parts[1:] if path.is_absolute() else path.parent.parts
        for keep in range(len(dirs), -1, -1):
            new_dir = self.root.joinpath(*dirs[len(dirs) - keep:])
            if (new_dir / path.name).is_file():
                logger.info("Images recorded in %s were found in %s (moved since import)",
                            path.parent, new_dir)
                return new_dir
        return None


def locate_rows(rows: list[tuple], db_path: str | Path) -> list[tuple]:
    """*rows* with their first column (``image_path``) located, for the exports."""
    locator = ImageLocator(db_path)
    return [(locator.locate(row[0]), *row[1:]) for row in rows]
