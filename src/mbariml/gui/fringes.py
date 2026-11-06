"""Display-only colour-fringe removal for ROI crops ("Fix colour fringes").

Bright animals often show rainbow edges: red displaced one way, blue the
other. On the Coral Observatory camera the lens itself accounts for little --
measured over 1,485 textured 256 px patches of 6 frames, red and blue sit only
~1-1.5 px off green at the corners of a 6720x4480 frame -- but individual
crops are off by 2-7 px, which is a lot once a 150 px crop is enlarged to fill
a tile. Two steps, both on the crop itself:

  1. Shift red and blue back onto green. The offset is found by ECC
     (OpenCV's findTransformECC, translation only) on each channel's
     normalised gradient magnitude, so it locks onto edges rather than onto
     the seafloor's brightness, and a weak, noisy red channel still aligns.
     The shift is kept only if it lowers the fringe measure (fringe_energy)
     by at least ACCEPT_GAIN; otherwise the crop is left as it was.
  2. Median-filter the colour (Lab a/b), not the brightness, near edges. What
     remains of a rainbow after alignment is chroma changing over a few
     pixels along an edge; an animal's own colour is broad and survives a
     CHROMA_MEDIAN_PX median.

Measured with fringe_energy on 291 crops whose channels were >= 2 px apart
(median, relative to the original; lower is better): phase correlation on the
raw channels -- this module's first version -- 0.975, i.e. it did almost
nothing; ECC on gradients 0.80 (a brute-force search for the best shift, the
limit for any translation, 0.785); this module (ECC, the guard, then the
edge chroma median) 0.66, with 3 of 291 worse by > 2%, ~24 ms a crop. On 132
random crops: 0.83, none worse, ~10 ms.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_SIDE_PX = 24  # too small to align
MAX_SHIFT_PX = 12.0
MAX_SHIFT_FRACTION = 1 / 8  # of the crop's short side
ACCEPT_GAIN = 0.02  # keep a shift only if it cuts fringe_energy by >= 2%
CHROMA_MEDIAN_PX = 7
ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)


def _gradients(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channel = cv2.GaussianBlur(channel.astype(np.float32), (0, 0), 0.7)
    return cv2.Sobel(channel, cv2.CV_32F, 1, 0), cv2.Sobel(channel, cv2.CV_32F, 0, 1)


def _gradient_magnitude(channel: np.ndarray) -> np.ndarray:
    m = cv2.magnitude(*_gradients(channel))
    return (m - m.mean()) / (m.std() + 1e-6)


def fringe_energy(bgr: np.ndarray, border: int = 8) -> float:
    """How much red's and blue's edges disagree with green's: mean distance
    between each channel's gradient and green's, every channel scaled to unit
    gradient spread first (so a dim red counts as much as a bright green)."""
    inner = (slice(border, -border), slice(border, -border))

    def normalised(channel):
        gx, gy = _gradients(channel)
        spread = cv2.magnitude(gx, gy)[inner].std() + 1e-6
        return gx[inner] / spread, gy[inner] / spread

    b, g, r = (normalised(bgr[:, :, k]) for k in range(3))
    return float(np.mean(np.hypot(r[0] - g[0], r[1] - g[1])) + np.mean(np.hypot(b[0] - g[0], b[1] - g[1])))


def _shift(channel: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return cv2.warpAffine(channel, np.float32([[1, 0, -dx], [0, 1, -dy]]), channel.shape[::-1],
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _aligned(bgr: np.ndarray) -> np.ndarray:
    """Red and blue shifted onto green (step 1), or *bgr* unchanged."""
    limit = min(MAX_SHIFT_PX, MAX_SHIFT_FRACTION * min(bgr.shape[:2]))
    green = _gradient_magnitude(bgr[:, :, 1])
    channels = list(cv2.split(bgr))
    for k in (0, 2):
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(green, _gradient_magnitude(bgr[:, :, k]), warp,
                                           cv2.MOTION_TRANSLATION, ECC_CRITERIA, None, 3)
        except cv2.error:  # didn't converge: flat or featureless crop
            continue
        dx, dy = float(warp[0, 2]), float(warp[1, 2])
        if np.hypot(dx, dy) <= limit:
            channels[k] = _shift(channels[k], dx, dy)
    out = cv2.merge(channels)
    return out if fringe_energy(out) <= fringe_energy(bgr) * (1 - ACCEPT_GAIN) else bgr


def _edge_chroma_median(bgr: np.ndarray) -> np.ndarray:
    """Step 2: a/b median-filtered, blended in by edge strength (luminance untouched)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    weight = cv2.magnitude(*_gradients(light))
    weight = np.clip(weight / (np.percentile(weight, 90) + 1e-6), 0, 1)
    weight = cv2.GaussianBlur(weight, (0, 0), 2)[:, :, None]
    ab = np.dstack([a, b]).astype(np.float32)
    ab_median = np.dstack([cv2.medianBlur(a, CHROMA_MEDIAN_PX), cv2.medianBlur(b, CHROMA_MEDIAN_PX)]).astype(np.float32)
    ab = np.clip(ab * (1 - weight) + ab_median * weight + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([light, ab[:, :, 0], ab[:, :, 1]]), cv2.COLOR_LAB2BGR)


def fix_fringes(bgr: np.ndarray) -> np.ndarray:
    """*bgr* with colour fringes reduced (see the module docstring). ~20 ms
    for a typical crop; safe to call from a worker thread."""
    if bgr is None or bgr.ndim != 3 or min(bgr.shape[:2]) < MIN_SIDE_PX:
        return bgr
    return _edge_chroma_median(_aligned(bgr))
