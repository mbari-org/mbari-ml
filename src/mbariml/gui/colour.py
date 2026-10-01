# Copied verbatim from mbariml-autolabel (src/autolabel/colour.py, as of its
# commit 82676c0) so `mbariml review` colour-corrects exactly as autolabel
# does. Keep the two in sync: change it there, then re-copy it here. The
# experiments/ script the docstring cites lives in that repo.
"""Display-only colour correction for green/blue water, measured per IMAGE.

The first version white-balanced each tile's crop on its own (gray-world): it
forced every crop's average to grey, so an orange tire or a green bottle came
out grey -- and gray-world alone, even per image, drains what little colour
these images have. Compared side by side on SeaClear tiles
(experiments/compare_display_colour.py), the most natural result was:

  per channel, from the whole image (a 1/8-size decode, ~3 ms):
    half  a 0.5-99.5 percentile stretch   (restores contrast and the weak red)
    half  gray-world                      (removes the green/blue cast)
  then CLAHE on lightness and a 1.35x saturation boost, on what's shown.

Because the channel correction comes from the whole image, a crop keeps its
own colour relative to its surroundings. Per-image corrections are cached in
memory and in ``<database>_colour.npz`` next to the database. Vectors, labels
and boxes always use the original pixels.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

SATURATION = 1.35


def measure(image_path: str) -> np.ndarray | None:
    """(2, 3) per-channel [multiplier; offset] for one image, or None if unreadable."""
    small = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_8)
    if small is None:
        return None
    f = small.reshape(-1, 3).astype(np.float32)
    lo, hi = np.percentile(f, [0.5, 99.5], axis=0)
    stretch = 255.0 / np.maximum(hi - lo, 1.0)
    m = f.mean(0)
    gray = np.clip(m.mean() / np.maximum(m, 1e-3), 0.3, 4.0)
    return np.stack([0.5 * stretch + 0.5 * gray, -0.5 * lo * stretch]).astype(np.float32)


class Corrector:
    def __init__(self, cache_path: str | Path | None = None):
        self.cache_path = Path(cache_path) if cache_path else None
        self.params: dict[str, np.ndarray] = {}
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._saved = 0
        if self.cache_path and self.cache_path.exists():
            try:
                z = np.load(self.cache_path, allow_pickle=False)
                self.params = dict(zip(z["paths"].tolist(), z["params"]))
                self._saved = len(self.params)
            except Exception:  # noqa: BLE001 -- a bad cache is just rebuilt
                self.params = {}

    def params_for(self, image_path: str) -> np.ndarray | None:
        p = self.params.get(image_path)
        if p is None:
            p = measure(image_path)
            if p is not None:
                self.params[image_path] = p
        return p

    def apply(self, bgr: np.ndarray, image_path: str) -> np.ndarray:
        p = self.params_for(image_path)
        if bgr is None or p is None:
            return bgr
        x = np.arange(256, dtype=np.float32)[:, None]
        lut = np.clip(x * p[0] + p[1], 0, 255).astype(np.uint8)  # (256, 3): one curve per channel
        out = cv2.LUT(bgr, lut.reshape(256, 1, 3))
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lab[..., 0] = self._clahe.apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=SATURATION)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def save(self) -> None:
        if not self.cache_path or len(self.params) == self._saved:
            return
        items = list(self.params.items())
        tmp = self.cache_path.with_name(self.cache_path.stem + ".tmp.npz")
        np.savez(tmp, paths=np.array([k for k, _ in items]), params=np.stack([v for _, v in items]))
        tmp.replace(self.cache_path)
        self._saved = len(items)

    def warm(self, image_paths, done=None) -> threading.Thread:
        """Measure every image in the background (~3 ms each), then save the cache."""
        def run():
            for path in image_paths:
                if path not in self.params:
                    p = measure(path)
                    if p is not None:
                        self.params[path] = p
            try:
                self.save()
            except Exception:  # noqa: BLE001 -- the cache is a convenience
                pass
            if done:
                done()
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t
