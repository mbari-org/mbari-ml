"""Display-only colour correction for underwater images, measured per IMAGE.

Originally copied from mbariml-autolabel (src/autolabel/colour.py, commit
82676c0), which blended a per-channel 0.5-99.5 percentile stretch with
gray-world. That was tuned on shallow SeaClear tiles and broke elsewhere,
because stretching each channel to its own range is a second, content-driven
white balance: a channel with a narrow range gets a huge gain. On Prosilica
deep-sea survey frames (strobe-lit, mild blue cast, red spanning ~40 levels)
it turned the seafloor pink -- mean output a* +20, where 0 is neutral; on
shallow Greek footage the bright blue water column went magenta (a* +14) and
sediment plumes glowed red; on empty i2MAP midwater frames (~25 grey levels in
all) it multiplied sensor noise ~10x into blue speckle. This version measures
each concern separately, from the whole image (a 1/8-size decode, ~3 ms):

  1. Red compensation (Ancuti et al. 2018, "Color balance and fusion for
     underwater image enhancement"): r += (mean_g - mean_r) * (1 - r) * g.
     Red is rebuilt from green, which survives water far better, instead of
     multiplying a weak, noisy red channel. It vanishes when red isn't weak,
     so strobe-lit deep frames get little and shallow frames a lot.
  2. Gray-world white balance at 0.85 strength (in log-gain), gains clamped to
     0.5-3x, aimed at a slightly WARM grey (red +4%, blue -8%) rather than a
     neutral one. Full strength forces a frame that is mostly open water to
     grey, tinting the far water pink; 0.85 leaves it blue. Neutral grey left
     sand and sediment cool and lavender (mean b* -7 to -12): natural scenes
     average slightly warm, so forcing their mean to exact grey overshoots.
     The warm target is a blue-yellow shift with a little red, since cutting
     blue alone tips everything green. Clipped and near-black pixels
     (burned-in overlays, black borders) don't vote.
  3. One contrast stretch on luminance, the same for every channel so it can't
     shift hue, with the gain capped at 3x so flat frames aren't turned into
     amplified noise, then a gamma (never below 0.6) that lifts a dark frame's
     median toward mid-grey. The top 0.5% is mapped to 92% white, not 100%:
     stretched to full white, bright sand blew out, and CLAHE then made it
     crunchy.
  4. On what's shown: a 1.15x saturation boost, and CLAHE on lightness only
     if "Local contrast" is turned up (off by default). Measured over the same
     40 images per set, CLAHE at clip 1.0 raised object-scale contrast by
     ~40-45% but fine grain (sensor noise, JPEG blocks) by ~50-60%, and it
     runs on whatever is shown, so a small tile is equalized differently
     from the same spot in the full frame. Colour doesn't need it.

A picked white point replaces steps 1-2 for every frame from the same
source (the video it was extracted from, else the image's folder): click
something you know is white or grey -- sand, a sponge, a calibration target
-- and the red compensation and channel gains that make it neutral are used
for the whole dive instead of a per-frame guess. Survey footage shot with one
camera and one set of lights has one right answer for this, and no automatic
estimate will find it as well as a person who knows what's in the frame.
Step 3 still runs per frame (it can't change hue). Picks are stored in
``<database>_white.json`` next to the database, keyed by the source as
recorded in the database, so they survive the project folder moving.

Measured over 40 images from each of three sets (mean output Lab a*/b*,
0/0 is neutral, +b* is yellow; original -> old -> this):
  Greek shallow    a* -10.9 -> -2.2 -> -6.3   b* -34.2 -> -5.1 -> +5.7
    (top 15%, water) a*  -3.2 -> +14.1 -> +6.0  b* -48.1 -> -22.3 -> -13.7
  Prosilica deep   a*  -4.7 -> +20.0 -> +0.2  b* -11.8 -> -14.2 -> +5.9
  MBARI ROV/i2MAP  a*  -5.0 -> -0.1 -> -1.9   b*  +0.1 -> -8.1 -> +8.5
The spread of a*/b* between images fell from ~5-7 to ~1-4, so frames of one
dive look alike. Pixels clipped to white on the Greek set: 2.7% stretching to
full white, 1.5% now.

Also compared: a common recipe that gray-worlds red and blue to green with
fixed x1.1 red / x0.9 blue fudge factors, then CLAHE clip 2.0 and unsharp
masking. Its warm bias is where the warm target here came from, and it looks
natural on lit deep-sea sediment. But multiplying red by the full green/red
ratio (~4x in shallow water) turns hazy open water salmon-pink (top 15% a*
+13.4, b* +5.5) and amplifies red-channel noise. Without a contrast stretch it
leaves dark frames dark (mean L 30 on the ROV/i2MAP set, against 48 here), and
the sharpening adds crunch.

Steps 1-3 reduce to lookup tables -- red from a 256x256 table over (red,
green), green and blue from 1-D tables -- so ``apply`` costs about what the
old per-channel LUT did. Because the correction comes from the whole image, a
crop keeps its own colour relative to its surroundings. Per-image corrections
are kept in memory for the session only: measuring is ~3 ms an image, done on
the tile workers as tiles load (about 1 s for a 400-frame database, 3 s for
1,000), which isn't worth a cache file and the versioning it needs to not go
stale. (Earlier versions wrote ``<database>_colour.npz``; it's no longer read.)
Vectors, labels and boxes always use the original pixels. If the source
image can't be found (see mbariml.paths), the correction is measured on the
crop being shown instead -- less faithful, since a crop's own average is
pulled toward grey, but far better than no correction.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from mbariml.logging_utils import get_logger

logger = get_logger(__name__)

WB_STRENGTH = 0.85
WARM = np.array([0.92, 1.0, 1.04], np.float32)  # BGR gains on top of gray-world
TOP = 0.92  # where the brightest 0.5% lands after the contrast stretch
MAX_CONTRAST_GAIN = 3.0
SATURATION = 1.15
WHITE_SUFFIX = "_white.json"
WHITE_PATCH = 3  # a picked white is the median of the (2n+1)^2 full-resolution pixels around the click
_LUMA = np.array([0.114, 0.587, 0.299], np.float32)  # BGR


def _compensate_red(red: np.ndarray, green: np.ndarray, k: float) -> np.ndarray:
    return red + k * (1.0 - red) * green


def measure(image_path: str) -> np.ndarray | None:
    """(7,) per-image parameters [k, gain_b, gain_g, gain_r, lo, scale, gamma],
    or None if unreadable."""
    if not Path(image_path).is_file():  # imread would print a warning per call
        return None
    small = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_8)
    return None if small is None else measure_pixels(small)


def measure_pixels(bgr: np.ndarray, white: dict | None = None) -> np.ndarray:
    """``measure`` on pixels already decoded, downscaled to <= 256 px a side.
    With *white* (see ``Corrector.set_white``), its red compensation and gains
    are used instead of this image's own estimate."""
    scale = 256 / max(bgr.shape[:2])
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    f = small.reshape(-1, 3).astype(np.float32) / 255.0
    keep = (f.max(1) < 0.98) & (f.mean(1) > 0.02)
    if keep.sum() > 100:
        f = f[keep]
    if white is None:
        mean = f.mean(0)
        k = max(0.0, float(mean[1] - mean[2]))
        f[:, 2] = _compensate_red(f[:, 2], f[:, 1], k)
        m = f.mean(0)
        gains = np.clip((m.mean() / np.maximum(m, 1e-4)) ** WB_STRENGTH, 0.5, 3.0) * WARM
    else:
        k, gains = float(white["k"]), np.array(white["gains"], np.float32)
        f[:, 2] = _compensate_red(f[:, 2], f[:, 1], k)
    lum = (f * gains) @ _LUMA
    lo, hi = np.percentile(lum, [0.5, 99.5])
    scale = min(TOP / max(hi - lo, 1e-3), MAX_CONTRAST_GAIN)
    lo = max(lo - (TOP / scale - (hi - lo)) / 2, 0.0)  # capped: centre the window
    mid = np.clip((np.median(lum) - lo) * scale, 0.02, 0.98)
    gamma = float(np.clip(np.log(0.45) / np.log(mid), 0.6, 1.0))
    return np.array([k, *gains, lo, scale, gamma], np.float32)


def _tone(x: np.ndarray, lo: float, scale: float, gamma: float) -> np.ndarray:
    return (np.clip((x - lo) * scale, 0.0, 1.0) ** gamma * 255.0 + 0.5).astype(np.uint8)


def _luts(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(blue (256,), green (256,), red (256*256,) indexed red*256 + green)."""
    k, gb, gg, gr, lo, scale, gamma = (float(v) for v in p)
    x = np.arange(256, dtype=np.float32) / 255.0
    red = _compensate_red(x[:, None], x[None, :], k) * gr
    return (_tone(x * gb, lo, scale, gamma), _tone(x * gg, lo, scale, gamma),
            _tone(red, lo, scale, gamma).reshape(-1))


class Corrector:
    def __init__(self, white_path: str | Path | None = None):
        self.white_path = Path(white_path) if white_path else None
        self.params: dict[str, np.ndarray] = {}  # automatic, per image
        self._clahe = None  # see set_local_contrast; None is off
        self._unreadable: set[str] = set()  # not retried: imread would warn on every tile
        # Picked whites, by source (see source_of), and the per-image
        # parameters they give -- in memory only, keyed by the pick too so a
        # worker finishing after a re-pick can't store a stale answer.
        self.whites: dict[str, dict] = {}
        self._white_params: dict[tuple, np.ndarray] = {}
        self._sources: dict[str, str] = {}
        if self.white_path and self.white_path.exists():
            try:
                self.whites = json.loads(self.white_path.read_text())
            except Exception:  # noqa: BLE001 -- unreadable picks are dropped, not fatal
                logger.warning("Could not read white points from %s; ignoring them", self.white_path)

    def set_sources(self, sources: dict[str, str]) -> None:
        """Which source (video, or folder) each image path belongs to, for
        the white points. Paths not listed fall back to their folder."""
        self._sources = dict(sources)

    def source_of(self, image_path: str) -> str:
        return self._sources.get(image_path) or str(Path(image_path).parent)

    def set_local_contrast(self, clip: float) -> None:
        """CLAHE clip limit on what's shown; 0 turns it off."""
        self._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)) if clip > 0 else None

    def white_for(self, image_path: str) -> dict | None:
        return self.whites.get(self.source_of(image_path))

    def set_white(self, image_path: str, bgr: np.ndarray, x: float, y: float) -> str:
        """Make the pixels around (*x*, *y*) of *bgr* -- the ORIGINAL full
        image at *image_path* -- neutral, for every frame from its source.
        Returns the source. Raises ValueError for a point that can't be a
        white reference: clipped (its true colour is unknown) or too dark
        (the gains would mostly amplify noise)."""
        h, w = bgr.shape[:2]
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            raise ValueError("Click on the image itself.")
        patch = bgr[max(yi - WHITE_PATCH, 0):yi + WHITE_PATCH + 1, max(xi - WHITE_PATCH, 0):xi + WHITE_PATCH + 1]
        c = np.median(patch.reshape(-1, 3), axis=0).astype(np.float32) / 255.0
        if c.max() >= 0.98:
            raise ValueError("That spot is clipped to white, so its true colour is unknown -- "
                             "pick something light but not blown out.")
        if c.min() < 0.03 or c.mean() < 0.08:
            raise ValueError("That spot is too dark to judge colour from -- pick something lighter.")
        k = float(measure_pixels(bgr)[0])  # this frame's red compensation, kept for the whole source
        c[2] = _compensate_red(c[2], c[1], k)
        gains = np.clip(c.mean() / np.maximum(c, 1e-4), 0.25, 4.0)
        source = self.source_of(image_path)
        self.whites[source] = {"k": k, "gains": [float(v) for v in gains],
                               "image": image_path, "x": xi, "y": yi}
        self._save_whites()
        return source

    def clear_white(self, image_path: str) -> str | None:
        """Back to the automatic estimate for *image_path*'s source; the
        source, or None if it had no white point."""
        source = self.source_of(image_path)
        if self.whites.pop(source, None) is None:
            return None
        self._save_whites()
        return source

    def _save_whites(self) -> None:
        if not self.white_path:
            return
        try:
            if not self.whites:  # nothing picked: no file, rather than an empty one
                self.white_path.unlink(missing_ok=True)
                return
            tmp = self.white_path.with_name(self.white_path.name + ".tmp")
            tmp.write_text(json.dumps(self.whites, indent=1))
            tmp.replace(self.white_path)
        except OSError:
            logger.exception("Could not save white points to %s", self.white_path)

    def params_for(self, image_path: str) -> np.ndarray | None:
        white = self.white_for(image_path)
        if white is not None:
            key = (image_path, white["k"], *white["gains"])
            p = self._white_params.get(key)
            if p is None and image_path not in self._unreadable and Path(image_path).is_file():
                small = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_8)
                if small is not None:
                    p = self._white_params[key] = measure_pixels(small, white)
            return p
        p = self.params.get(image_path)
        if p is None and image_path not in self._unreadable:
            p = measure(image_path)
            if p is not None:
                self.params[image_path] = p
            else:
                if not self._unreadable:
                    logger.warning(
                        "Can't find source image %s (not at its recorded path or under the "
                        "database's folder); colour-correcting from the crop itself for this "
                        "and any other missing images", image_path)
                self._unreadable.add(image_path)
        return p

    def apply(self, bgr: np.ndarray, image_path: str) -> np.ndarray:
        if bgr is None or bgr.size == 0:
            return bgr
        p = self.params_for(image_path)
        if p is None:
            p = measure_pixels(bgr, self.white_for(image_path))
        lut_b, lut_g, lut_r = _luts(p)
        b, g, r = cv2.split(bgr)
        red = np.take(lut_r, (r.astype(np.intp) << 8) | g)
        out = cv2.merge([cv2.LUT(b, lut_b), cv2.LUT(g, lut_g), red])
        clahe = self._clahe
        if clahe is not None:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[..., 0] = clahe.apply(lab[..., 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=SATURATION)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
