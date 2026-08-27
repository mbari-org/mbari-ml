"""GUI engine for the interactive ROI review tool (``mbariml review``).

The rendering/threading core of this package (``mosaic_view.py``,
``selection_coordinator.py``, ``runnables.py``, and most of ``rect_widget.py``,
``roi_loading_coordinator.py``, ``bounding_box.py``, and ``detail_view.py``)
is adapted from MBARI's ``vars-gridview`` (MIT License, Copyright (c) 2020
Monterey Bay Aquarium Research Institute). See ``THIRD_PARTY_NOTICES.md`` at
the repository root. The VARS/Annosaurus REST data layer that gridview
normally sits on top of has been replaced end-to-end with DuckDB-native
services (``query_service.py``, ``roi_service.py``, ``annotation_service.py``)
matching mbariml's flat ``predictions`` schema (see ``mbariml.db``) -- there
is no concept tree, knowledgebase, or network fetch involved here.

The full-image detail view (``detail_view.py``, ``bounding_box.py``) uses
pyqtgraph for wheel-zoom/drag-pan and draggable/resizable box overlays, same
as vars-gridview. pyqtgraph auto-detects a Qt binding at import time by
checking which one is already loaded; this MUST be pinned to PySide6 before
anything imports pyqtgraph, since the rest of this app uses PySide6 and
loading PyQt6 as well (pyqtgraph's default when nothing else is loaded yet)
would mean two separate Qt runtimes in one process.
"""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
