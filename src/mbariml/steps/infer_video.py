"""``mbariml infer video``: run a YOLO model over video, and store detections
in the same curation database schema everything else uses.

Two modes, for two genuinely different jobs:

**stride** -- sample every Nth frame and treat each one as an independent
image. No tracking, so a slow-moving animal in view for 300 frames yields a
detection every ``--stride`` frames. Simple, streamable, one pass.

**track** -- follow each object across frames and keep exactly ONE ROI per
track. This is what you want for building training/curation data out of
video: 300 near-identical crops of the same animal are worthless for
clustering (they'd swamp it) and tedious to review, whereas one good crop per
individual is exactly one reviewable observation.

Why tracking is two passes
--------------------------
You cannot pick a track's representative frame until the track has ended --
its centre and its best-confidence frame are only known once you've seen all
of it. So:

  Pass 1  Track the whole video, recording per-track observation *metadata*
          only (frame number, box, confidence, class). No pixels are
          retained, so memory is O(open tracks), not O(video).
  Pass 2  One forward sweep that pulls out just the chosen frames.

Pass 2 sweeps rather than seeks deliberately -- see ``mbariml.video``: seeking
by frame number is unreliable on long-GOP encodings and would silently pair a
box with the wrong pixels. Decode is far cheaper than inference, so the extra
pass costs a fraction of pass 1.

Why the frames get written to disk
----------------------------------
Each chosen frame is extracted to a real JPEG under ``OUTPUT_DIR/frames/``,
and ``image_path`` points at *that file*. Video rows are then indistinguishable
from image rows to every downstream step -- review, embed, cluster, all four
exports, html, stats all work unchanged, with no video-aware code anywhere
outside this module. The alternative (store the video path plus a frame
number, resolve frames lazily) would have meant teaching five separate
consumers to decode video, and the exports would have had to materialize
frames anyway: a VOC XML pointing at "video.mp4, frame 1234" isn't something
any trainer understands. ``video_path``/``frame_number``/``frame_time_s``/
``track_id``/``track_length`` are still recorded as provenance (see
``mbariml.db``), which is what the review GUI's "Open Video" button uses to
jump back to the moment in the source footage.

Only frames that actually produced a detection are written -- an empty frame
has no rows pointing at it, so saving it would be pure disk cost.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import typer
from tqdm import tqdm

from mbariml import db, video
from mbariml.image_quality import compute_sharpness
from mbariml.logging_utils import get_logger
from mbariml.steps.infer_images import PRESETS, _resolve_preset
from mbariml.yolo_utils import load_model, resolve_device

app = typer.Typer(help="Run a YOLO model over video (strided frames, or tracking).")
logger = get_logger(__name__)

DEFAULT_TRACKER = "tracktrack.yaml"
DEFAULT_STRIDE = 30

# How to pick the one frame that represents a track. Defaults to
# best-conf-central because a track's first and last frames are when the
# animal is entering/leaving view -- clipped at the image edge, occluded, or
# motion-blurred -- and plain max-confidence happily picks exactly those.
# Restricting to the middle third avoids the entry/exit artifacts; taking the
# most confident frame within it still favors a clean, unambiguous view.
TRACK_ROI_POLICIES = ("best-conf-central", "best-conf", "center", "sharpest-central")
DEFAULT_TRACK_ROI_POLICY = "best-conf-central"


@dataclass
class _Observation:
    """One frame's worth of a track: metadata only, never pixels."""

    frame_number: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_id: int


@dataclass
class _Track:
    track_id: int
    observations: list[_Observation] = field(default_factory=list)

    @property
    def dominant_class_id(self) -> int:
        """Majority class over the whole track.

        A tracker will happily keep an object's identity across frames where
        the detector flip-flops between two visually similar classes; taking
        the majority over every observation is far more stable than trusting
        whichever class the one chosen frame happened to get.
        """
        return Counter(observation.class_id for observation in self.observations).most_common(1)[0][0]


def _central_slice(observations: list[_Observation]) -> list[_Observation]:
    """The middle third of a track, or the whole thing if it's too short for
    a middle third to mean anything."""
    count = len(observations)
    if count < 3:
        return observations
    central = observations[count // 3 : (2 * count) // 3]
    return central or observations


def _candidate_observations(track: _Track, policy: str) -> list[_Observation]:
    """The observation(s) worth decoding for this track under *policy*.

    Confidence/position-based policies resolve to exactly one frame up front
    (no pixels needed to decide). ``sharpest-central`` can't: sharpness is a
    property of the decoded crop, so every frame in the middle third is a
    candidate and the winner is settled during the sweep in
    :func:`_extract_track_rois`.
    """
    observations = track.observations
    if policy == "center":
        return [observations[len(observations) // 2]]
    if policy == "best-conf":
        return [max(observations, key=lambda o: o.confidence)]
    if policy == "best-conf-central":
        return [max(_central_slice(observations), key=lambda o: o.confidence)]
    if policy == "sharpest-central":
        return _central_slice(observations)
    raise typer.BadParameter(f"--track-roi must be one of {list(TRACK_ROI_POLICIES)}")


def _insert_video_rows(conn, rows: list[tuple]) -> None:
    """Insert curation-schema rows that carry video provenance.

    Same explicit-column-list reasoning as the image path (see
    ``infer_images.insert_rows``), just with the video columns filled in
    rather than left NULL.
    """
    if not rows:
        return
    db.fast_executemany(
        conn,
        """INSERT INTO predictions
           (id, image_name, image_path, roi_index, x_min, y_min, x_max, y_max,
            class_id, confidence, label, embedding, new_label, roi, sharpness, verified,
            video_path, frame_number, frame_time_s, track_id, track_length)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


def _build_row(
    *,
    roi_id: int,
    frame_path: Path,
    frame_bgr,
    box: tuple[float, float, float, float],
    confidence: float,
    class_id: int,
    label: str,
    video_path: Path,
    frame_number: int,
    frame_time_s: float,
    track_id: Optional[int],
    track_length: Optional[int],
) -> Optional[tuple]:
    """Crop, encode, and assemble one row. Returns None for a degenerate crop."""
    x_min, y_min, x_max, y_max = (int(round(v)) for v in box)
    height, width = frame_bgr.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(width, x_max), min(height, y_max)
    if x_max <= x_min or y_max <= y_min:
        return None

    roi = frame_bgr[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None
    ok, roi_encoded = cv2.imencode(".jpg", roi)
    if not ok:
        logger.warning("Failed to encode ROI at frame %d of %s; skipping", frame_number, video_path.name)
        return None

    return (
        roi_id, frame_path.name, str(frame_path), roi_id,
        float(x_min), float(y_min), float(x_max), float(y_max),
        class_id, float(confidence), label,
        None,  # embedding, filled in by `mbariml embed`
        None,  # new_label, filled in by clustering/review
        roi_encoded.tobytes(),
        compute_sharpness(roi),
        0,  # verified, set by the review GUI
        str(video_path), int(frame_number), float(frame_time_s),
        track_id, track_length,
    )


# -- stride mode --------------------------------------------------------------


def _run_stride(
    model, video_path: Path, info: video.VideoInfo, conn, frames_dir: Path,
    *, stride: int, batch_size: int, yolo_params: dict, next_id: int,
) -> int:
    """Sample every *stride*-th frame, detect on it, and store any hits.

    Returns the next free id. Rows are written per batch, so an interrupted
    run keeps everything up to the last completed batch -- same durability
    rule as the image path.
    """
    expected = (info.frame_count // stride) if info.frame_count else None
    pending_frames: list[tuple[int, "cv2.typing.MatLike"]] = []
    total_rows = 0

    def flush(batch: list[tuple[int, "cv2.typing.MatLike"]]) -> int:
        nonlocal next_id, total_rows
        if not batch:
            return 0
        frames = [frame for _, frame in batch]
        results = model.predict(source=frames, **yolo_params)

        rows = []
        for (frame_number, frame_bgr), result in zip(batch, results):
            if not len(result.boxes):
                continue
            frame_path = frames_dir / f"{video.frame_stem(video_path, frame_number)}.jpg"
            if not video.save_frame(frame_bgr, frame_path):
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                row = _build_row(
                    roi_id=next_id,
                    frame_path=frame_path,
                    frame_bgr=frame_bgr,
                    box=tuple(box.xyxy[0].tolist()),
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                    label=model.names[class_id],
                    video_path=video_path,
                    frame_number=frame_number,
                    frame_time_s=video.frame_time_seconds(frame_number, info.fps),
                    track_id=None,
                    track_length=None,
                )
                if row is not None:
                    rows.append(row)
                    next_id += 1
        _insert_video_rows(conn, rows)
        total_rows += len(rows)
        return len(rows)

    with tqdm(total=expected, desc=f"{video_path.name} (stride {stride})", unit="frame") as progress:
        for frame_number, frame_bgr in video.iter_strided_frames(video_path, stride):
            pending_frames.append((frame_number, frame_bgr))
            if len(pending_frames) >= batch_size:
                flush(pending_frames)
                pending_frames = []
            progress.update(1)
        flush(pending_frames)

    logger.info("%s: %d detection(s) from strided frames", video_path.name, total_rows)
    return next_id


# -- tracking mode ------------------------------------------------------------


def _track_pass(model, video_path: Path, info: video.VideoInfo, *, tracker: str, yolo_params: dict) -> dict[int, _Track]:
    """Pass 1: track the whole video, keeping metadata only (never pixels).

    ``stream=True`` is not optional here: without it Ultralytics accumulates
    every frame's Results object -- the whole video's worth -- in memory
    before returning, which is the same "buffer everything" failure the image
    path was already burned by, except worse at video scale.
    """
    tracks: dict[int, _Track] = {}
    results = model.track(source=str(video_path), stream=True, tracker=tracker, verbose=False, **yolo_params)

    with tqdm(total=info.frame_count or None, desc=f"{video_path.name} (tracking)", unit="frame") as progress:
        for frame_number, result in enumerate(results):
            progress.update(1)
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue  # no confirmed tracks on this frame
            for box in boxes:
                if box.id is None:
                    continue  # detection the tracker hasn't confirmed into a track yet
                track_id = int(box.id[0])
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                tracks.setdefault(track_id, _Track(track_id=track_id)).observations.append(
                    _Observation(
                        frame_number=frame_number,
                        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                        confidence=float(box.conf[0]),
                        class_id=int(box.cls[0]),
                    )
                )

    logger.info(
        "%s: %d track(s) over %d observation(s)",
        video_path.name, len(tracks), sum(len(t.observations) for t in tracks.values()),
    )
    return tracks


def _extract_track_rois(
    model, video_path: Path, info: video.VideoInfo, tracks: dict[int, _Track], conn, frames_dir: Path,
    *, policy: str, next_id: int,
) -> int:
    """Pass 2: sweep the video once and write one ROI per track.

    For every policy except ``sharpest-central`` each track has a single
    candidate frame, so the first (only) crop seen for it wins outright. For
    ``sharpest-central`` every middle-third frame is a candidate and the
    sweep keeps whichever crop scores highest, which is why the winner is
    held as an already-cropped row rather than re-decoded afterwards.
    """
    if not tracks:
        return next_id

    # frame number -> [(track, observation), ...] to evaluate when the sweep
    # reaches that frame.
    candidates_by_frame: dict[int, list[tuple[_Track, _Observation]]] = {}
    for track in tracks.values():
        for observation in _candidate_observations(track, policy):
            candidates_by_frame.setdefault(observation.frame_number, []).append((track, observation))

    # track id -> (score, row-building inputs) for the best crop seen so far.
    best: dict[int, tuple[float, tuple]] = {}

    for frame_number, frame_bgr in tqdm(
        video.extract_frames(video_path, candidates_by_frame.keys()),
        total=len(candidates_by_frame),
        desc=f"{video_path.name} (extracting ROIs)",
        unit="frame",
    ):
        for track, observation in candidates_by_frame.get(frame_number, ()):
            box = (observation.x_min, observation.y_min, observation.x_max, observation.y_max)
            if policy == "sharpest-central":
                crop = frame_bgr[
                    max(0, int(observation.y_min)) : int(observation.y_max),
                    max(0, int(observation.x_min)) : int(observation.x_max),
                ]
                score = compute_sharpness(crop) if crop.size else -1.0
            else:
                score = observation.confidence
            if track.track_id in best and best[track.track_id][0] >= score:
                continue
            best[track.track_id] = (score, (frame_number, frame_bgr.copy(), box, observation))

    rows = []
    for track_id, (_score, (frame_number, frame_bgr, box, observation)) in sorted(best.items()):
        track = tracks[track_id]
        frame_path = frames_dir / f"{video.frame_stem(video_path, frame_number)}.jpg"
        if not video.save_frame(frame_bgr, frame_path):
            continue
        class_id = track.dominant_class_id
        row = _build_row(
            roi_id=next_id,
            frame_path=frame_path,
            frame_bgr=frame_bgr,
            box=box,
            confidence=observation.confidence,
            class_id=class_id,
            label=model.names[class_id],
            video_path=video_path,
            frame_number=frame_number,
            frame_time_s=video.frame_time_seconds(frame_number, info.fps),
            track_id=track_id,
            track_length=len(track.observations),
        )
        if row is not None:
            rows.append(row)
            next_id += 1

    _insert_video_rows(conn, rows)
    logger.info("%s: wrote %d ROI(s), one per track (%s)", video_path.name, len(rows), policy)
    return next_id


# -- command ------------------------------------------------------------------


@app.command()
def infer_video(
    model_path: str = typer.Argument(..., help="Path to a YOLO .pt model, or a stock Ultralytics model name."),
    input_path: str = typer.Argument(..., help="A video file, or a directory of videos (searched recursively)."),
    output_dir: str = typer.Argument(..., help="Directory for the database and the extracted frames/ directory."),
    mode: str = typer.Option(
        "track",
        help="'track' keeps ONE ROI per tracked object (what you want for curation/training data); "
        "'stride' samples every --stride-th frame and treats each as an independent image.",
    ),
    stride: int = typer.Option(DEFAULT_STRIDE, help="stride mode only: sample every Nth frame."),
    tracker: str = typer.Option(
        DEFAULT_TRACKER,
        help="track mode only: Ultralytics tracker config -- a shipped name (tracktrack.yaml, "
        "botsort.yaml, bytetrack.yaml, ocsort.yaml, deepocsort.yaml, fasttrack.yaml) or a path "
        "to your own YAML of tracking hyperparameters.",
    ),
    track_roi: str = typer.Option(
        DEFAULT_TRACK_ROI_POLICY,
        help="track mode only: which frame of a track becomes its ROI. 'best-conf-central' "
        "(default) takes the most confident frame from the track's middle third, avoiding the "
        "entry/exit frames where the animal is clipped or blurred; 'sharpest-central' picks the "
        "least blurry one there instead; 'best-conf' and 'center' use the whole track.",
    ),
    min_track_length: int = typer.Option(
        1, help="track mode only: ignore tracks with fewer than this many observations."
    ),
    preset: str = typer.Option(
        "curate",
        help="'curate' (conf 0.005, imgsz 1952) or 'predict' (conf 0.08, imgsz 992). Any option "
        "below overrides its preset value. Annotated-image saving does not apply to video.",
    ),
    limit: Optional[int] = typer.Option(None, help="Only process the first N videos found."),
    batch_size: int = typer.Option(16, help="stride mode only: frames sent to YOLO per batch."),
    conf: Optional[float] = typer.Option(None, help="Confidence threshold. [default: from --preset]"),
    iou: Optional[float] = typer.Option(None, help="IoU threshold for NMS. [default: from --preset]"),
    max_det: int = typer.Option(500, help="Maximum detections per frame."),
    imgsz: Optional[int] = typer.Option(None, help="Inference image size. [default: from --preset]"),
    device: str = typer.Option("auto", help="Device to run on: 'auto', 'mps', 'cuda', or 'cpu'."),
) -> None:
    """Run YOLO over video and store detections (with ROI crops and extracted
    frames) in OUTPUT_DIR/yolo_predictions.duckdb."""
    if mode not in ("track", "stride"):
        raise typer.BadParameter("--mode must be 'track' or 'stride'")
    if track_roi not in TRACK_ROI_POLICIES:
        raise typer.BadParameter(f"--track-roi must be one of {list(TRACK_ROI_POLICIES)}")
    if preset not in PRESETS:
        raise typer.BadParameter(f"--preset must be one of {sorted(PRESETS)}")

    settings = _resolve_preset(preset, conf=conf, iou=iou, imgsz=imgsz)
    # Resolved to absolute deliberately: image_path/video_path are how every
    # later command (review, the exports, the GUI's "Open Video" button) finds
    # the pixels again, and those run from whatever directory the user happens
    # to be in -- a relative path recorded here silently stops resolving the
    # moment anything runs from somewhere else.
    output_dir_path = Path(output_dir).resolve()
    frames_dir = output_dir_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    videos = [v.resolve() for v in video.collect_videos(input_path)]
    if limit:
        videos = videos[:limit]
    logger.info(
        "Mode '%s' | preset '%s': conf=%.4g, iou=%.4g, imgsz=%d | device: %s | %d video(s)",
        mode, preset, settings["conf"], settings["iou"], settings["imgsz"], resolved_device, len(videos),
    )

    model = load_model(model_path)
    (output_dir_path / "names.txt").write_text("\n".join(model.names.values()))

    yolo_params = {
        "conf": settings["conf"],
        "iou": settings["iou"],
        "imgsz": settings["imgsz"],
        "max_det": max_det,
        "device": resolved_device,
        "agnostic_nms": True,
        "save": False,
    }

    db_path = output_dir_path / "yolo_predictions.duckdb"
    failures = 0
    with db.init_curation_db(db_path) as conn:
        next_id = db.next_free_id(conn)
        for video_path in videos:
            try:
                info = video.probe_video(video_path)
                logger.info(
                    "=== %s: %dx%d, %.2f fps, %d frames (%.1f s) ===",
                    video_path.name, info.width, info.height, info.fps, info.frame_count, info.duration_s,
                )
                if mode == "stride":
                    next_id = _run_stride(
                        model, video_path, info, conn, frames_dir,
                        stride=stride, batch_size=batch_size, yolo_params=yolo_params, next_id=next_id,
                    )
                else:
                    tracks = _track_pass(model, video_path, info, tracker=tracker, yolo_params=yolo_params)
                    if min_track_length > 1:
                        dropped = {k: v for k, v in tracks.items() if len(v.observations) < min_track_length}
                        if dropped:
                            logger.info(
                                "Dropping %d track(s) shorter than --min-track-length %d",
                                len(dropped), min_track_length,
                            )
                        tracks = {k: v for k, v in tracks.items() if len(v.observations) >= min_track_length}
                    next_id = _extract_track_rois(
                        model, video_path, info, tracks, conn, frames_dir,
                        policy=track_roi, next_id=next_id,
                    )
            except Exception:
                failures += 1
                logger.exception("Failed on %s; continuing with the remaining videos", video_path)

        final_count = db.row_count(conn)
        conn.execute("DELETE FROM run_info")
        conn.execute("INSERT INTO run_info VALUES (?, CURRENT_TIMESTAMP)", (model_path,))

    logger.info(
        "Done: %d video(s), %d total row(s) in %s, extracted frames in %s",
        len(videos), final_count, db_path, frames_dir,
    )
    if final_count:
        logger.info(
            "Ready to continue with: mbariml embed %s | mbariml review %s", db_path, db_path
        )
    if failures:
        logger.error("%d video(s) failed -- see the tracebacks above.", failures)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
