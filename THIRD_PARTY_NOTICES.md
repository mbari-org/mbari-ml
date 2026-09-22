# Third-Party Notices

## Ultralytics YOLO

`mbariml` calls [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
for all model loading, inference, and multi-object tracking (see
`src/mbariml/yolo_utils.py`). It is imported as a dependency rather than
vendored, so no Ultralytics source is redistributed here.

**Ultralytics is licensed AGPL-3.0**, which is more restrictive than this
project's own MIT licence, and installing `mbariml` installs it. Anyone
redistributing work built on this pipeline, or offering it as a network
service, should read Ultralytics' terms and decide what they require; the
Ultralytics Enterprise licence exists for users who need permissive terms.

Model weights carry terms separate from the code that loads them. A fine-tuned
checkpoint inherits whatever its base weights were licensed under, and the
DINOv3 weights used for embeddings (pulled by `timm` as
`vit_large_patch16_dinov3.lvd1689m`) are covered by Meta's DINOv3 licence
rather than by `timm`'s Apache-2.0.

## Other dependencies

Beyond Ultralytics, the runtime dependencies are permissively licensed --
DuckDB (MIT), timm (Apache-2.0), torchvision (BSD), pandas (BSD), OpenCV
(Apache-2.0), pyqtgraph (MIT), tqdm (MPL-2.0/MIT) -- with one to be aware of:
**PySide6 is LGPL-3.0/GPL**, used here as an ordinary pip-installed dynamic
import, which is what the LGPL contemplates.

## vars-gridview

The mosaic rendering/threading engine in `src/mbariml/gui/` (`mosaic_view.py`,
`selection_coordinator.py`, `runnables.py`, and the core mechanics of
`rect_widget.py`, `roi_loading_coordinator.py`, and `colors.py`) is adapted
from [MBARI's vars-gridview](https://github.com/mbari-org/vars-gridview),
translated from PyQt6 to PySide6 and trimmed to mbariml's DuckDB schema. The
VARS/Annosaurus REST data layer (query controller, knowledgebase/concept
tree, session/login, video-sequence resolution, live per-tile embedding
model) is not part of this adaptation and has been replaced with mbariml's
own DuckDB-native services (`query_service.py`, `roi_service.py`,
`annotation_service.py`); see the module docstrings in `src/mbariml/gui/`
for exactly what was ported, trimmed, rewritten, or dropped.

vars-gridview is distributed under the MIT License:

```
MIT License

Copyright (c) 2020 Monterey Bay Aquarium Research Institute (MBARI)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
