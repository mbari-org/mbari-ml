"""Optional SAM3 assistance for the review GUI: one-click boxes and box tightening.

Uses the *interactive* half of Meta's Segment Anything 3 through Ultralytics'
``SAM3Predictor`` -- point and box prompts on one image, SAM2-style, not
SAM3's text/concept search:

  a point     the object under the click; SAM3 offers ~3 nested masks
              (a part, the object, the object plus its surroundings)
  a box       a tight box around what that box contains

SAM3 is entirely optional. Nothing here loads at startup: :func:`unavailable`
only looks for files, and the model loads on first use, on a worker thread.
If anything is missing or fails to load, review works exactly as without it
and the SAM3 controls are disabled, with the reason as their tooltip.

Setup, once:

  pip install -e ".[sam3]"   the `sam3` extra: Ultralytics' CLIP, not on PyPI
  mbariml sam3 download      sam3.pt into the Hugging Face cache. Gated:
                             request access at huggingface.co/facebook/sam3,
                             then `hf auth login`
  mbariml review DB          finds it there by itself

:func:`resolve_model_path` picks the model: ``--sam3-model``, else
``$MBARIML_SAM3_MODEL``, else ``sam3.pt`` in the local Hugging Face cache (a
file lookup, no network). ``mbariml sam3 check`` reports which one it found,
or what is missing.

Ultralytics' CLIP has to be checked for up front: when it's missing,
Ultralytics tries to ``pip install`` it from GitHub in the middle of loading
the model, and PyPI's unrelated ``clip`` package (a clipboard tool) imports
fine and then fails confusingly.

Measured on an M3 Ultra (MPS): ~6 s to load, ~0.3-0.45 s of image features per
new image (cached for the image on screen), then ~15-50 ms per prompt.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mbariml.logging_utils import get_logger

logger = get_logger(__name__)

MODEL_ENV_VAR = "MBARIML_SAM3_MODEL"
CLIP_INSTALL = "pip install -e \".[sam3]\" (from the mbari-ml checkout), or pip install git+https://github.com/ultralytics/CLIP.git"
HF_REPO = "facebook/sam3"
HF_FILENAME = "sam3.pt"

Box = tuple[float, float, float, float]  # x_min, y_min, x_max, y_max in image pixels

TIGHTEN_ROUNDS = 4  # prompts per box at most -- see Sam3.tighten
TIGHTEN_MARGIN = 0.05  # how far past the original box (per side, as a fraction) a tightened box may reach


@dataclass
class Candidate:
    box: Box
    score: float
    kind: str  # 'point' | 'box'


def resolve_model_path(model_path: str | None) -> str | None:
    """The ``--sam3-model`` option, else ``$MBARIML_SAM3_MODEL``, else the copy
    ``mbariml sam3 download`` put in the Hugging Face cache, else None."""
    return model_path or os.environ.get(MODEL_ENV_VAR) or cached_model_path()


def cached_model_path() -> str | None:
    """``sam3.pt`` from the local Hugging Face cache, if it was downloaded.
    Only looks on disk -- never touches the network."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(HF_REPO, HF_FILENAME)
    return path if isinstance(path, str) else None  # else None or a "known missing" sentinel


def download_model() -> str:
    """Download ``sam3.pt`` into the Hugging Face cache (or find it already
    there) and return its path. Raises RuntimeError with what to do when the
    repo's gate hasn't been passed."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        return hf_hub_download(HF_REPO, HF_FILENAME)
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise RuntimeError(
            f"{HF_REPO} is gated. Request access at https://huggingface.co/{HF_REPO}, wait for approval, "
            f"then log in with `hf auth login` (or set HF_TOKEN) and run this again.\n({type(exc).__name__}: {exc})"
        ) from exc


def unavailable(model_path: str | None) -> str | None:
    """Why SAM3 can't be used, or None if it looks usable. Cheap: no imports
    of torch or Ultralytics, just file checks -- safe to call at startup."""
    if not model_path:
        return (f"No SAM3 model: run `mbariml sam3 download` once, or start review with "
                f"--sam3-model /path/to/sam3.pt (or set {MODEL_ENV_VAR}).")
    if not Path(model_path).is_file():
        return f"SAM3 model not found: {model_path}"
    spec = importlib.util.find_spec("clip")
    if spec is None:
        return f"SAM3 needs Ultralytics' CLIP: {CLIP_INSTALL}"
    locations = spec.submodule_search_locations or []
    if not any((Path(loc) / "simple_tokenizer.py").is_file() for loc in locations):
        return ("The installed 'clip' package isn't Ultralytics' CLIP (PyPI's 'clip' is an unrelated "
                f"clipboard tool): pip uninstall clip, then {CLIP_INSTALL}")
    return None


def iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: Box, b: Box) -> float:
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


def judge_tightening(old: Box, new: Box) -> str:
    """Classify SAM3's box for an existing one:

    'tighter'    mostly inside the old box, and noticeably different from it
    'unchanged'  already tight -- nothing worth writing
    'disagrees'  SAM3 found something else: it reaches well outside the old
                 box (the object was cut off, or SAM3 grabbed a neighbour),
                 or it shrank to a small part of it
    """
    if _area(new) <= 0 or _area(old) <= 0:
        return "disagrees"
    if iou(old, new) >= 0.95:
        return "unchanged"
    inside = _intersection(old, new) / _area(new)
    if inside < 0.85 or _area(new) < 0.15 * _area(old):
        return "disagrees"
    return "tighter"


class Sam3:
    """SAM3's interactive predictor, loaded once and shared; thread-safe.

    The review GUI runs every call on a one-thread pool of its own (shared
    with its DINOv3 embeddings), so the lock here only matters to other
    callers.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._lock = threading.Lock()
        self._pr = None
        self._feat_key: str | None = None
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._pr is not None

    def load(self) -> None:
        with self._lock:
            self._load()

    def _load(self) -> None:
        if self._pr is not None:
            return
        if self._load_error is not None:  # don't spend seconds failing the same way again
            raise RuntimeError(self._load_error)
        try:
            self._pr = self._build()
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise
        self._feat_key = None

    def _build(self):
        import torch
        from ultralytics.models.sam import SAM3Predictor

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading SAM3 from %s on %s", self.model_path, device)
        pr = SAM3Predictor(overrides=dict(model=self.model_path, device=device, half=device != "cpu",
                                          imgsz=1008, conf=0.0, verbose=False, save=False))
        pr.setup_model(verbose=False)
        # The first prompt compiles kernels (~1.5 s): pay it here, not on the first click.
        dummy = np.zeros((256, 256, 3), np.uint8)
        pr.set_image(dummy)
        pr.inference_features(pr.features, dummy.shape[:2], bboxes=[[10, 10, 100, 100]])
        return pr

    def _features(self, key: str, image_bgr: np.ndarray):
        if self._feat_key != key:
            self._pr.set_image(image_bgr)
            self._feat_key = key
        return self._pr.features

    def _run(self, feats, shape, **prompt) -> list[tuple[Box, float]]:
        _masks, boxes = self._pr.inference_features(feats, shape[:2], **prompt)
        if boxes is None or len(boxes) == 0:
            return []
        h, w = shape[:2]
        out = []
        for b in boxes.float().cpu().numpy():
            x0, y0, x1, y1 = (float(np.clip(b[0], 0, w)), float(np.clip(b[1], 0, h)),
                              float(np.clip(b[2] + 1, 0, w)), float(np.clip(b[3] + 1, 0, h)))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                out.append(((x0, y0, x1, y1), float(b[4])))
        return out

    def point_candidates(self, image_key: str, image_bgr: np.ndarray, x: float, y: float) -> list[Candidate]:
        """SAM3's boxes for the object at (x, y), smallest first, near-duplicates
        merged. ``image_key`` (the image path) lets features be reused."""
        with self._lock:
            self._load()
            feats = self._features(image_key, image_bgr)
            raw = [Candidate(b, s, "point") for b, s in
                   self._run(feats, image_bgr.shape, points=[[float(x), float(y)]], labels=[1],
                             multimask_output=True)]
        kept: list[Candidate] = []
        for c in sorted(raw, key=lambda c: -c.score):
            if all(iou(c.box, k.box) < 0.92 for k in kept):
                kept.append(c)
        return sorted(kept, key=lambda c: _area(c.box))

    def tighten(self, image_key: str, image_bgr: np.ndarray, boxes: list[Box]) -> list[Box | None]:
        """A tight box around what each of ``boxes`` contains, or None where
        SAM3 finds nothing. All boxes share one image's features.

        Each prompt is the box *plus* a positive click at its centre, and the
        new box is measured from SAM3's most confident mask, cut to the
        original box (plus TIGHTEN_MARGIN) and with stray specks dropped (see
        _tight_box). The result is fed back in as the next prompt until it
        stops changing (IoU > 0.97), at most TIGHTEN_ROUNDS prompts.

        A box prompt alone stays close to the box it's given, and its mask
        often spills onto the seabed around the object, so the box SAM3
        returned could even grow or drift sideways. Measured on 60 imported
        Cyprus litter boxes: median area after tightening 0.63 of the
        original with the box prompt alone, 0.45 now; boxes judged
        "disagrees" (reaching outside, or shrinking to a small part) went
        from 14 to 2. ~0.6 s per box on an M3 Ultra, image features included."""
        out: list[Box | None] = []
        with self._lock:
            self._load()
            feats = self._features(image_key, image_bgr)
            for box in boxes:
                original = tuple(map(float, box))
                current: Box | None = None
                prompt = original
                for _ in range(TIGHTEN_ROUNDS):
                    new = self._tight_box(feats, image_bgr.shape, prompt, original)
                    if new is None:
                        break
                    settled = iou(new, prompt) > 0.97  # SAM3 handed back the box it was given
                    current = prompt = new
                    if settled:
                        break
                out.append(current)
        return out

    def _tight_box(self, feats, shape, prompt: Box, original: Box) -> Box | None:
        """One tightening prompt: ``prompt`` plus a click at its centre. The
        box around SAM3's best mask, counting only mask pixels within
        ``original`` (+ TIGHTEN_MARGIN), and only connected pieces at least a
        tenth the size of the biggest -- a few stray pixels anywhere would
        otherwise stretch the box to reach them."""
        cx, cy = (prompt[0] + prompt[2]) / 2, (prompt[1] + prompt[3]) / 2
        masks, boxes = self._pr.inference_features(feats, shape[:2], bboxes=[list(prompt)], points=[[cx, cy]],
                                                   labels=[1], multimask_output=True)
        if masks is None or len(masks) == 0:
            return None
        mask = masks[int(boxes[:, 4].argmax())].cpu().numpy()
        h, w = shape[:2]
        x0, y0, x1, y1 = original
        mx, my = TIGHTEN_MARGIN * (x1 - x0), TIGHTEN_MARGIN * (y1 - y0)
        left, top = int(max(0, x0 - mx)), int(max(0, y0 - my))
        right, bottom = int(min(w, x1 + mx + 1)), int(min(h, y1 + my + 1))
        window = mask[top:bottom, left:right].astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(window, connectivity=8)
        if n <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i + 1 for i, a in enumerate(areas) if a >= 0.1 * areas.max()]
        ys, xs = np.nonzero(np.isin(labels, keep))
        bx0, by0, bx1, by1 = left + xs.min(), top + ys.min(), left + xs.max() + 1, top + ys.max() + 1
        if bx1 - bx0 < 4 or by1 - by0 < 4:
            return None
        return float(bx0), float(by0), float(bx1), float(by1)

__all__ = ["Sam3", "Candidate", "Box", "MODEL_ENV_VAR", "HF_REPO", "HF_FILENAME", "resolve_model_path",
           "cached_model_path", "download_model", "unavailable", "judge_tightening", "iou"]
