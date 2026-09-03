"""Shared video decoding, frame extraction, and "open the source footage" helpers.

Used by ``mbariml infer video`` (both stride and tracking modes) and by the
review GUI's "Open Video" button.

Two decoding rules this module exists to enforce in one place:

1. **Never seek; sweep.** ``cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, n)`` is
   unreliable on long-GOP encodings (H.264/HEVC survey footage very much
   included) -- it lands on the nearest keyframe, so the frame you get back
   is not necessarily the frame you asked for, silently pairing a detection's
   box with the wrong pixels. :func:`extract_frames` instead reads forward
   from the start and picks frames off as it passes them, which is exact by
   construction.

2. **Grab, then retrieve only what you want.** ``cap.grab()`` advances the
   decoder without paying for full decode + color conversion;
   ``cap.retrieve()`` finishes the job. Sweeping a video for a sparse set of
   frames (one per track, say) is dramatically cheaper when the frames you're
   skipping only cost a ``grab()`` -- so both the sparse sweep and the strided
   reader below skip with ``grab()`` and only ``retrieve()`` frames they
   actually return.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2

from mbariml.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"})

# Fallback when a container reports a nonsense frame rate (0, NaN, or absurdly
# high). Only affects the frame <-> timestamp math, never which frames are
# read, so a wrong guess costs you a slightly-off "Open Video" seek position,
# not wrong pixels.
FALLBACK_FPS = 30.0


@dataclass
class VideoInfo:
    """What a container reports about itself."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe_video(video_path: str | Path) -> VideoInfo:
    """Read a video's fps/frame count/dimensions, raising if it can't be opened.

    ``frame_count`` comes from the container's own metadata and is occasionally
    wrong (a truncated or still-being-written file, some variable-frame-rate
    encodings). It's used for progress reporting and stride planning only --
    never to decide when to stop reading, which is always driven by
    ``cap.grab()`` actually returning False.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video (unsupported codec, or not a video): {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not fps or fps != fps or fps > 1000:  # 0, NaN, or nonsense
            logger.warning("Video %s reports fps=%r; assuming %.1f for timestamps.", video_path, fps, FALLBACK_FPS)
            fps = FALLBACK_FPS
        return VideoInfo(
            path=video_path,
            fps=fps,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        cap.release()


def collect_videos(
    input_path: str | Path, extensions: Iterable[str] = DEFAULT_VIDEO_EXTENSIONS
) -> list[Path]:
    """Every video under *input_path* (or just it, if it's a single file).

    Mirrors ``mbariml.images.collect_images``: raises rather than returning an
    empty list, so a typo'd path fails immediately instead of producing a run
    that "worked" and did nothing.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {input_path}")

    exts = {e.lower() for e in extensions}
    if input_path.is_file():
        if input_path.suffix.lower() not in exts:
            raise ValueError(f"Not a recognized video file ({sorted(exts)}): {input_path}")
        return [input_path]

    videos = sorted(f for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() in exts)
    if not videos:
        raise FileNotFoundError(
            f"No videos with extensions {sorted(exts)} found under {input_path} (searched recursively)."
        )
    return videos


def iter_strided_frames(video_path: str | Path, stride: int) -> Iterator[tuple[int, "cv2.typing.MatLike"]]:
    """Yield ``(frame_number, frame_bgr)`` for every *stride*-th frame.

    Skipped frames cost only a ``grab()`` (no decode/convert) -- see the module
    docstring. ``stride=1`` yields every frame.
    """
    stride = max(1, int(stride))
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frame_number = 0
        while True:
            if not cap.grab():
                return
            if frame_number % stride == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield frame_number, frame
            frame_number += 1
    finally:
        cap.release()


def extract_frames(
    video_path: str | Path, frame_numbers: Iterable[int]
) -> Iterator[tuple[int, "cv2.typing.MatLike"]]:
    """Yield ``(frame_number, frame_bgr)`` for exactly *frame_numbers*, in order.

    One forward sweep, no seeking (see the module docstring for why). Stops as
    soon as the last requested frame has been handed back, so asking for a
    handful of early frames doesn't decode the whole file. Frames that turn out
    not to exist (a requested number past the end of a truncated file) are
    simply not yielded -- the caller sees a short result rather than an error.
    """
    wanted = sorted({int(n) for n in frame_numbers})
    if not wanted:
        return
    wanted_set = set(wanted)
    last_wanted = wanted[-1]

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frame_number = 0
        while frame_number <= last_wanted:
            if not cap.grab():
                return
            if frame_number in wanted_set:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield frame_number, frame
                else:
                    logger.warning("Could not decode frame %d of %s; skipping", frame_number, video_path)
            frame_number += 1
    finally:
        cap.release()


def frame_time_seconds(frame_number: int, fps: float) -> float:
    """Presentation timestamp of *frame_number*, in seconds."""
    return frame_number / fps if fps else 0.0


def frame_stem(video_path: str | Path, frame_number: int) -> str:
    """``<video_stem>_frame_<n zero-padded>`` -- the name an extracted frame is
    saved under.

    Zero-padded to 6 digits so extracted frames sort correctly in a file
    browser, and prefixed with the video's own stem so frames from different
    videos never collide in one output directory (the same problem
    ``mbariml.image_naming.disambiguated_stem`` solves for source images).
    """
    return f"{Path(video_path).stem}_frame_{int(frame_number):06d}"


def save_frame(frame_bgr, output_path: str | Path, jpeg_quality: int = 95) -> bool:
    """Write one decoded frame to disk as a JPEG. Returns whether it worked."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        logger.warning("Failed to write frame to %s", output_path)
    return ok


# -- "Open Video" (review GUI) ------------------------------------------------

# Players that can be told to start at a timestamp, best first. IINA is
# checked inside its app bundle too: it ships `iina-cli`, but installing that
# on PATH is a separate opt-in step most people never do.
_IINA_BUNDLED_CLI = "/Applications/IINA.app/Contents/MacOS/iina-cli"


def _iina_command(path: Path, seconds: float) -> list[str] | None:
    cli = shutil.which("iina-cli") or (_IINA_BUNDLED_CLI if os.path.exists(_IINA_BUNDLED_CLI) else None)
    if cli is None:
        return None
    # --mpv-resume-playback=no matters: without it, IINA restores wherever you
    # last left this file and silently ignores the start position we asked for.
    return [cli, f"--mpv-start={seconds:.3f}", "--mpv-resume-playback=no", str(path)]


def _mpv_command(path: Path, seconds: float) -> list[str] | None:
    mpv = shutil.which("mpv")
    return [mpv, f"--start={seconds:.3f}", str(path)] if mpv else None


def _vlc_command(path: Path, seconds: float) -> list[str] | None:
    vlc = shutil.which("vlc")
    if vlc is None and sys.platform == "darwin" and os.path.exists("/Applications/VLC.app/Contents/MacOS/VLC"):
        vlc = "/Applications/VLC.app/Contents/MacOS/VLC"
    return [vlc, f"--start-time={seconds:.3f}", str(path)] if vlc else None


def open_video_at(video_path: str | Path, seconds: float) -> str:
    """Open *video_path* in a video player, positioned at *seconds*.

    Tries the players that can actually honor a start position first (IINA,
    mpv, VLC), then falls back to the default browser with a ``#t=`` media
    fragment -- which Safari/Chrome honor for codecs they can play, and which
    at least opens the right file everywhere else.

    Returns a short description of what was launched, for the caller to show
    in a status line. Raises ``RuntimeError`` only if nothing at all could be
    launched.
    """
    path = Path(video_path)
    if not path.exists():
        raise RuntimeError(f"Video not found on disk: {path}")
    seconds = max(0.0, float(seconds))

    for name, build in (("IINA", _iina_command), ("mpv", _mpv_command), ("VLC", _vlc_command)):
        command = build(path, seconds)
        if command is None:
            continue
        try:
            # Detached: the player outlives this call, and a player that exits
            # non-zero later must not surface as an exception here.
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("Opened %s at %.1fs in %s", path.name, seconds, name)
            return f"{name} at {seconds:.1f}s"
        except OSError:
            logger.exception("Failed to launch %s; trying the next player", name)

    url = path.resolve().as_uri() + f"#t={seconds:.3f}"
    if webbrowser.open(url):
        logger.info("Opened %s at %.1fs in the default browser", path.name, seconds)
        return f"browser at {seconds:.1f}s"

    raise RuntimeError(
        f"No video player available to open {path}. Install IINA, mpv, or VLC, "
        "or open the file manually."
    )


__all__ = [
    "VideoInfo",
    "DEFAULT_VIDEO_EXTENSIONS",
    "probe_video",
    "collect_videos",
    "iter_strided_frames",
    "extract_frames",
    "frame_time_seconds",
    "frame_stem",
    "save_frame",
    "open_video_at",
]
