"""Main window for the interactive ROI review/labeling GUI (``mbariml review``).

Orchestrates the pieces adapted from MBARI vars-gridview (mosaic rendering,
threaded tile loading, selection/keyboard-nav, the pyqtgraph-based detail
view -- see ``THIRD_PARTY_NOTICES.md``) together with mbariml's own
DuckDB-native services (``query_service``, ``roi_service``,
``annotation_service``). vars-gridview splits this responsibility across
``ui/MainWindow.py`` (window chrome) and ``ui/mosaic/image_mosaic.py``
(mosaic manager) because it also has to juggle VARS session/login state,
video-sequence prefetching, and knowledgebase dialogs; none of that exists
here, so it's one class.

Preserves every feature of the previous (synchronous, no-cache) revision of
this GUI: label filter, sort by label/name/sharpness, a relabel box (free
text, or pick from a dropdown of labels already in use), delete-with-
confirmation, right-click similarity sort, full-image preview with
selection boxes, and page navigation. What's new: threaded/bounded-
concurrency ROI decoding, a bounded LRU cache for full-image previews,
diff-based grid re-layout instead of full rebuilds on every page/sort
change, a tile zoom slider, a "verified" flag (with a Verify button and a
"hide verified" display filter), arrow-key tile navigation with Shift/Ctrl
range selection, wheel-zoom/drag-pan on the full-image panel, and
draggable/resizable bounding boxes for every detection on the shown image
(not just the current mosaic page's rows -- other detections on the same
source frame may be sorted onto a different page).
"""

from __future__ import annotations

from threading import Event

from PySide6.QtCore import QEvent, QThreadPool, Qt, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mbariml import db
from mbariml.gui import annotation_service, query_service
from mbariml.gui.detail_view import DetailView
from mbariml.gui.mosaic_view import MosaicView, MosaicVisibilityFilters
from mbariml.gui.rect_widget import RectWidget
from mbariml.gui.roi_loading_coordinator import MosaicRoiLoadingCoordinator
from mbariml.gui.roi_service import RoiService, crop_and_encode
from mbariml.gui.runnables import Worker
from mbariml.gui.selection_coordinator import MosaicSelectionCoordinator, SelectionModel
from mbariml.logging_utils import get_logger

logger = get_logger(__name__)

ZOOM_MIN, ZOOM_MAX, ZOOM_DEFAULT = 50, 200, 100  # percent
ROI_LOAD_CONCURRENCY = 8

# Each pyqtgraph RectROI overlay (handles, region-changed wiring, a TextItem
# label) costs ~1-1.5ms to construct on the GUI thread -- fine for the tens
# of detections a typical frame has, but a frame with hundreds+ (a dense
# fish/coral count) would freeze the UI for a second or more if every one
# became a fully interactive overlay. Cap it, always keeping whichever
# detection(s) are currently active so the thing you clicked never
# disappears, and say so in the status bar rather than truncating silently.
MAX_DETAIL_BOXES = 200

# Dark palette for the whole window -- the default light Qt theme's bright
# white panels are genuinely hard on the eyes reviewing imagery in a dark
# room for long stretches. Applied once via self.setStyleSheet() in
# __init__, so it cascades to every child widget in the controls panel;
# per-widget stylesheets set elsewhere (the red Quit button, tile fills in
# RectWidget.paint()) still take precedence for their own widget, same as
# any CSS cascade. The mosaic QGraphicsView/QGraphicsScene background and
# the pyqtgraph detail view aren't styleable via Qt stylesheets -- see
# their own setBackground()/setBackgroundBrush() calls below.
_BG = "#1e1f22"  # window background
_PANEL = "#2b2d31"  # inputs/buttons
_PANEL_HOVER = "#35373c"
_PANEL_PRESSED = "#222327"
_BORDER = "#46484d"
_TEXT = "#e3e3e3"
_TEXT_DISABLED = "#6f7278"
_ACCENT = "#34a1eb"  # matches RectWidget.SELECTION_HIGHLIGHT_COLOR

DARK_STYLESHEET = f"""
    QWidget {{
        background-color: {_BG};
        color: {_TEXT};
        selection-background-color: {_ACCENT};
        selection-color: #ffffff;
    }}
    QMainWindow, QGraphicsView {{
        background-color: {_BG};
        border: none;
    }}
    QLabel {{
        background: transparent;
    }}
    QPushButton {{
        background-color: {_PANEL};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        padding: 4px 10px;
    }}
    QPushButton:hover {{
        background-color: {_PANEL_HOVER};
    }}
    QPushButton:pressed {{
        background-color: {_PANEL_PRESSED};
    }}
    QPushButton:disabled {{
        color: {_TEXT_DISABLED};
        background-color: {_PANEL_PRESSED};
    }}
    QComboBox, QLineEdit, QSpinBox {{
        background-color: {_PANEL};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        padding: 3px 6px;
    }}
    QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled {{
        color: {_TEXT_DISABLED};
    }}
    QComboBox QAbstractItemView {{
        background-color: {_PANEL};
        color: {_TEXT};
        selection-background-color: {_ACCENT};
        outline: none;
    }}
    QCheckBox {{
        background: transparent;
        spacing: 6px;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: {_BORDER};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {_ACCENT};
        width: 14px;
        margin: -6px 0;
        border-radius: 7px;
    }}
    QMenu {{
        background-color: {_PANEL};
        color: {_TEXT};
        border: 1px solid {_BORDER};
    }}
    QMenu::item:selected {{
        background-color: {_ACCENT};
    }}
    QScrollBar:vertical, QScrollBar:horizontal {{
        background: {_BG};
        border: none;
    }}
    QScrollBar::handle {{
        background: {_PANEL_HOVER};
        border-radius: 4px;
    }}
    QMessageBox {{
        background-color: {_BG};
    }}
"""


class MainWindow(QMainWindow):
    def __init__(self, database_path: str, label: str | None = None, page_size: int = 500) -> None:
        super().__init__()
        self.setWindowTitle("ROI Labeling App")
        self.setGeometry(100, 100, 2400, 1400)
        self.setStyleSheet(DARK_STYLESHEET)

        # mbariml.db.init_curation_db() is a context manager (guarantees the
        # DuckDB connection is flushed/closed) but this window needs a
        # connection that lives as long as it does; enter it manually and
        # exit in closeEvent() rather than bypassing it with a raw
        # duckdb.connect(). Deliberately init_curation_db(), not the plain
        # db.connect() an earlier revision used: every *pipeline* step reads
        # a database that some earlier step already created (and therefore
        # already schema-migrated), but `review` is often the very first
        # thing pointed at a database in this session -- e.g. right after
        # `detect`/`infer-images` on an older mbariml version, or reopening
        # a database from before a column (like `verified`) existed. Plain
        # connect() only opens the file; it never runs CURATION_SCHEMA_SQL's
        # ALTER TABLE ADD COLUMN IF NOT EXISTS migrations, so a column added
        # after that database was created would 500 on the first query
        # referencing it -- confirmed directly: exactly this "Referenced
        # column 'verified' not found" error against a real pre-existing
        # database.
        self._db_ctx = db.init_curation_db(database_path)
        self.conn = self._db_ctx.__enter__()
        self._roi_service = RoiService()

        self.label_filter = label
        self.page_size = page_size
        self.current_page = 0
        self._total_rows = 0  # updated by every load_page() -- see _fetch_page_worker
        self._verified_count = 0  # updated by every load_page() and verify/label/delete action
        self._unverified_count = 0
        self.sort_option = query_service.DEFAULT_SORT_OPTION
        self.similarity_order: list[int] | None = None
        self.similarity_reference: int | None = None
        self.hide_verified = False
        # 0.0 = no floor (every confidence value passes). Applied at the
        # query level everywhere label_filter/exclude_verified are -- see
        # query_service._build_where.
        self.min_confidence: float = 0.0
        self._zoom_percent = ZOOM_DEFAULT

        self._rect_widgets: list[RectWidget] = []
        self._n_columns = 0
        # The widgets actually laid out in the grid as of the last
        # render_mosaic() call -- may be a subset of _rect_widgets when
        # "Hide verified" is on. Keyboard navigation must use this, not
        # _rect_widgets, or its index arithmetic (paired with _n_columns,
        # itself computed from this same set) disagrees with what's
        # actually on screen -- see MosaicSelectionCoordinator's
        # visible_widgets_getter.
        self._visible_widgets: list[RectWidget] = []
        self._page_load_generation = 0
        self._roi_loading_cancel_event = Event()

        # QRunnable (unlike QObject) isn't kept alive by parent-child
        # ownership once handed to QThreadPool -- with nothing else
        # referencing a dispatched Worker, CPython can garbage-collect it
        # (and its `signals` QObject) while the pool is still running it on
        # another thread, which crashes (this was an actual segfault hit
        # during testing). Every Worker dispatched from this window is kept
        # here for the window's lifetime; the memory cost is negligible.
        self._workers: list[Worker] = []

        # Full-image detail view state: every detection on the currently
        # shown image (not just this page's rows -- see fetch_rois_for_image),
        # plus which one is highlighted when it isn't also a loaded grid tile.
        self._current_detail_image_path: str | None = None
        self._frame_rois: list[query_service.FrameRoi] = []
        self._frame_roi_total: int = 0
        self._detail_only_active_roi_index: int | None = None

        self.graphics_view = QGraphicsView()
        self.graphics_view.installEventFilter(self)
        self._mosaic_view = MosaicView(self.graphics_view)

        self.selection_model = SelectionModel(parent=self)
        self.selection_model.selection_changed.connect(self._on_selection_changed)
        self._selection = MosaicSelectionCoordinator(
            parent=self,
            selection_model=self.selection_model,
            mosaic_view=self._mosaic_view,
            all_widgets_getter=lambda: self._rect_widgets,
            visible_widgets_getter=lambda: self._visible_widgets,
        )

        self._roi_loading = MosaicRoiLoadingCoordinator(parent=self, max_concurrency=ROI_LOAD_CONCURRENCY)
        self._roi_loading.progress.connect(self._on_roi_loading_progress)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.graphics_view, stretch=3)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self._create_full_image_view(), stretch=2)
        right_layout.addWidget(self._create_controls(), stretch=1)
        right_container = QWidget()
        right_container.setLayout(right_layout)
        main_layout.addWidget(right_container, stretch=2)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self._install_shortcuts()
        self._refresh_known_labels()
        self.load_page()

    # -- Lifecycle ----------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Stop pending background loads and flush/close the DuckDB connection."""
        self._roi_loading_cancel_event.set()
        self._roi_loading.cancel_pending()
        try:
            self._db_ctx.__exit__(None, None, None)
        except Exception:
            logger.exception("Error closing database connection")
        super().closeEvent(event)

    def _install_shortcuts(self) -> None:
        """Delete removes the selection (with confirmation); Escape clears
        it; V verifies the selection, Shift+V unverifies it; arrow keys
        navigate/extend the selection (handled in eventFilter)."""
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self.delete_selected_rois)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.unselect_all)
        QShortcut(QKeySequence(Qt.Key.Key_V), self, activated=self.verify_selected)
        QShortcut(QKeySequence("Shift+V"), self, activated=self.unverify_selected)

    def eventFilter(self, source, event) -> bool:  # noqa: N802
        if source is self.graphics_view and event.type() == QEvent.Type.Resize:
            self.render_mosaic()
        if source is self.graphics_view and event.type() == QEvent.Type.KeyPress:
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if self._selection.select_relative(
                key=event.key(),
                columns=self._n_columns,
                activate_callback=lambda w: self._on_rect_clicked(w, None),
                shift=shift,
            ):
                return True  # consume so QGraphicsView doesn't also auto-scroll
        return super().eventFilter(source, event)

    # -- Controls panel -------------------------------------------------------

    def _create_full_image_view(self) -> QWidget:
        self._detail_view = DetailView(self)
        return self._detail_view

    def _create_controls(self) -> QWidget:
        controls_layout = QVBoxLayout()

        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Sort by:"))
        self.sort_dropdown = QComboBox()
        self.sort_dropdown.addItems(list(query_service.SORT_COLUMNS.keys()))
        self.sort_dropdown.setCurrentText(self.sort_option)
        self.sort_dropdown.currentTextChanged.connect(self._on_sort_changed)
        sort_layout.addWidget(self.sort_dropdown)
        self.clear_similarity_button = QPushButton("Clear Similarity Sort")
        self.clear_similarity_button.setEnabled(False)
        self.clear_similarity_button.clicked.connect(self._on_clear_similarity_sort)
        sort_layout.addWidget(self.clear_similarity_button)
        controls_layout.addLayout(sort_layout)
        controls_layout.addWidget(QLabel("Right-click an ROI to sort all pages by similarity to it."))

        controls_layout.addWidget(
            QLabel("Tile/box captions show both labels: bold = curated, plain = original YOLO class.")
        )

        controls_layout.addWidget(QLabel("Search (filters the grid to matching ROIs):"))
        search_layout = QHBoxLayout()
        self.search_combo = QComboBox()
        self.search_combo.setEditable(True)
        self.search_combo.setPlaceholderText("Type or select a concept... (Enter to search)")
        # Same reasoning as label_combo below: only _refresh_known_labels()
        # should change this list.
        self.search_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.search_combo.setCurrentText(self.label_filter or "")
        self.search_combo.lineEdit().returnPressed.connect(self._on_search)
        # activated (not currentTextChanged) fires only when an item is
        # actually picked from the popup list, not on every keystroke while
        # typing -- typing shouldn't reload the whole grid character by
        # character.
        self.search_combo.activated.connect(lambda _index: self._on_search())
        search_layout.addWidget(self.search_combo)
        search_button = QPushButton("Search")
        search_button.clicked.connect(self._on_search)
        search_layout.addWidget(search_button)
        self.clear_search_button = QPushButton("Clear Search")
        self.clear_search_button.setEnabled(bool(self.label_filter))
        self.clear_search_button.clicked.connect(self._on_clear_search)
        search_layout.addWidget(self.clear_search_button)
        controls_layout.addLayout(search_layout)

        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Tile size:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(ZOOM_MIN)
        self.zoom_slider.setMaximum(ZOOM_MAX)
        self.zoom_slider.setValue(ZOOM_DEFAULT)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)
        controls_layout.addLayout(zoom_layout)

        confidence_layout = QHBoxLayout()
        self.confidence_label = QLabel("Min confidence: 0.00")
        confidence_layout.addWidget(self.confidence_label)
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)  # 0-100 -> 0.00-1.00
        self.confidence_slider.setValue(0)
        self.confidence_slider.valueChanged.connect(self._on_confidence_slider_moved)
        confidence_layout.addWidget(self.confidence_slider)
        show_confidence_button = QPushButton("Show")
        show_confidence_button.clicked.connect(self._on_confidence_filter_clicked)
        confidence_layout.addWidget(show_confidence_button)
        controls_layout.addLayout(confidence_layout)
        controls_layout.addWidget(
            QLabel("View-only: hides ROIs below this YOLO detection confidence. Never changes stored data.")
        )

        visibility_layout = QHBoxLayout()
        verify_button = QPushButton("Verify (V)")
        verify_button.clicked.connect(self.verify_selected)
        visibility_layout.addWidget(verify_button)
        unverify_button = QPushButton("Unverify (Shift+V)")
        unverify_button.clicked.connect(self.unverify_selected)
        visibility_layout.addWidget(unverify_button)
        hide_verified_box = QCheckBox("Hide verified")
        hide_verified_box.toggled.connect(self._on_hide_verified_toggled)
        visibility_layout.addWidget(hide_verified_box)
        controls_layout.addLayout(visibility_layout)

        controls_layout.addWidget(QLabel("Relabel selected (type a new label, or pick an existing one):"))
        label_editor_layout = QHBoxLayout()
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.setPlaceholderText("Enter or select a label... (Enter to apply)")
        # Editable QComboBox defaults to inserting whatever you type into
        # its own item list on Enter -- that would pile up ad hoc duplicate
        # entries every time this is used. _refresh_known_labels() is the
        # only thing that should ever change the item list (it already adds
        # a newly-used label next refresh, sourced from the database).
        self.label_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.label_combo.lineEdit().returnPressed.connect(self.update_labels)
        label_editor_layout.addWidget(self.label_combo)
        label_button = QPushButton("Label")
        label_button.clicked.connect(self.update_labels)
        label_editor_layout.addWidget(label_button)
        controls_layout.addLayout(label_editor_layout)

        action_layout = QHBoxLayout()
        delete_button = QPushButton("Delete (Del)")
        delete_button.clicked.connect(self.delete_selected_rois)
        action_layout.addWidget(delete_button)
        unselect_button = QPushButton("Unselect All (Esc)")
        unselect_button.clicked.connect(self.unselect_all)
        action_layout.addWidget(unselect_button)
        controls_layout.addLayout(action_layout)

        nav_layout = QHBoxLayout()
        prev_button = QPushButton("Previous")
        prev_button.clicked.connect(self.previous_page)
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next_page)
        nav_layout.addWidget(prev_button)
        nav_layout.addWidget(next_button)
        nav_layout.addWidget(QLabel("Go to page:"))
        self.page_jump_spinbox = QSpinBox()
        self.page_jump_spinbox.setMinimum(1)
        self.page_jump_spinbox.setMaximum(1)  # widened once a page has actually loaded
        self.page_jump_spinbox.setValue(1)
        nav_layout.addWidget(self.page_jump_spinbox)
        jump_button = QPushButton("Go")
        jump_button.clicked.connect(self._on_jump_to_page)
        nav_layout.addWidget(jump_button)
        # Enter in the spinbox jumps too, without needing the button.
        self.page_jump_spinbox.editingFinished.connect(self._on_jump_to_page)
        self.verified_count_label = QLabel("")
        nav_layout.addWidget(self.verified_count_label)
        quit_button = QPushButton("Quit")
        quit_button.clicked.connect(self.close)
        quit_button.setStyleSheet(
            "QPushButton { background-color: #ff3b30; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #e0342b; }"
        )
        nav_layout.addWidget(quit_button)
        controls_layout.addLayout(nav_layout)

        self.status_label = QLabel("")
        controls_layout.addWidget(self.status_label)

        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        return controls_widget

    # -- Status / quick labels ------------------------------------------------

    @property
    def total_pages(self) -> int:
        """Total pages for the current filter/similarity-sort state. At
        least 1 even when _total_rows is 0 (nothing loaded yet), so "Page
        1/1" rather than a division-by-zero-flavored "Page 1/0"."""
        return max(1, -(-self._total_rows // self.page_size))  # ceil division

    def _update_page_jump_range(self) -> None:
        """Keep the jump-to-page spinbox's range in sync with the current
        total_pages -- called whenever _total_rows changes (every page load,
        since a label filter/similarity sort/delete can all change it)."""
        self.page_jump_spinbox.blockSignals(True)
        self.page_jump_spinbox.setMaximum(self.total_pages)
        self.page_jump_spinbox.blockSignals(False)

    def update_status_bar(self) -> None:
        total = len(self._rect_widgets)
        selected = len(self.selection_model.selected)
        filter_text = f" | filter: {self.label_filter}" if self.label_filter else ""
        confidence_text = f" | min confidence: {self.min_confidence:.2f}" if self.min_confidence > 0 else ""
        similarity_active = self.similarity_order is not None
        mode_text = (
            f" | sorted by similarity to ROI #{self.similarity_reference}" if similarity_active else ""
        )
        # Only actionable while a similarity sort is actually narrowing the
        # grid -- e.g. a same-label search that matches just one concept,
        # with nothing else in the controls panel telling you how to get
        # back to the normal view short of picking a different sort option
        # (which does clear it, but non-obviously -- see _on_sort_changed).
        self.clear_similarity_button.setEnabled(similarity_active)
        capped_text = (
            f" | showing {len(self._frame_rois)}/{self._frame_roi_total} boxes on this frame"
            if self._frame_roi_total > len(self._frame_rois)
            else ""
        )
        self.status_label.setText(
            f"Page {self.current_page + 1}/{self.total_pages} | {total} shown | {selected} selected"
            f"{filter_text}{confidence_text}{mode_text}{capped_text}"
        )
        self._refresh_verified_count_label()
        # Keep the spinbox showing the page we're actually on, without
        # re-triggering a jump (it only acts on an explicit Enter/click).
        self.page_jump_spinbox.blockSignals(True)
        self.page_jump_spinbox.setValue(self.current_page + 1)
        self.page_jump_spinbox.blockSignals(False)

    def _refresh_verified_count_label(self) -> None:
        """Show the review-progress counter next to the page controls.

        Kept as its own small setter (rather than only inline in
        update_status_bar()) because a couple of paths -- e.g. the detail-
        view delete, which sets a custom one-off status_label message
        instead of calling update_status_bar() -- still need to keep this
        counter in sync without also overwriting that custom message.
        """
        self.verified_count_label.setText(
            f"Verified: {self._verified_count} | Unverified: {self._unverified_count}"
        )

    def _refresh_known_labels(self) -> None:
        """Repopulate the relabel dropdown AND the search dropdown with
        every label currently in use (alphabetically) -- called at startup
        and after every label change, so a freshly-typed label shows up as
        a pick-list option immediately in both. Doesn't restrict what you
        can type into either; see ``query_service.fetch_known_labels``."""
        labels = query_service.fetch_known_labels(self.conn)

        current_label_text = self.label_combo.currentText()
        self.label_combo.blockSignals(True)
        self.label_combo.clear()
        self.label_combo.addItems(labels)
        self.label_combo.setCurrentText(current_label_text)
        self.label_combo.blockSignals(False)

        current_search_text = self.search_combo.currentText()
        self.search_combo.blockSignals(True)
        self.search_combo.clear()
        self.search_combo.addItem("")  # "" = no filter/show everything
        self.search_combo.addItems(labels)
        self.search_combo.setCurrentText(current_search_text)
        self.search_combo.blockSignals(False)

    # -- Paging / loading -----------------------------------------------------

    def load_page(self) -> None:
        """Fetch the current page off the GUI thread, then (re)build tiles."""
        self._page_load_generation += 1
        generation = self._page_load_generation
        offset = self.current_page * self.page_size

        # Immediate feedback: the query dispatched below runs on a worker
        # thread and can take a moment on a large database (a full similarity
        # ordering, a big label filter, ...) -- without this, the status bar
        # just shows stale info from before the click until the page lands.
        self.status_label.setText("Loading page...")

        # self.conn itself (not a cursor made from it) is passed to the
        # worker thread, which calls .cursor() on it from within its own
        # run(): per DuckDB's own docs, a cursor must be created *on* the
        # thread that will use it, not created on the GUI thread and handed
        # to a worker -- doing it backwards (as an earlier revision of this
        # code did) is undefined behavior and was the actual cause of a
        # SIGSEGV under load (a large similarity-sort result feeding
        # straight into the next page load). See
        # https://duckdb.org/docs/current/guides/python/multiple_threads.
        #
        # The signal is connected to a genuine bound method (`self` is a
        # QObject), not a lambda: PySide can only auto-detect that a
        # cross-thread emission needs to be queued onto the GUI event loop
        # when the receiver is a QObject method with known thread affinity.
        # A lambda has none, so Qt invokes it immediately on the worker
        # thread instead -- which "works" in a bare script with no event
        # loop contention but silently never touches the GUI once a real
        # window/event loop is running. The per-call context (`generation`)
        # is threaded through as part of the worker's own return value
        # instead, following the same pattern as RectWidget's ROI refresh.
        worker = Worker(
            self._fetch_page_worker,
            generation,
            self.conn,
            offset=offset,
            page_size=self.page_size,
            label_filter=self.label_filter,
            sort_option=self.sort_option,
            similarity_order=self.similarity_order,
            exclude_verified=self.hide_verified,
            min_confidence=self.min_confidence,
        )
        worker.signals.result.connect(self._on_page_loaded)
        worker.signals.error.connect(self._on_page_load_error)
        self._workers.append(worker)  # see __init__ comment: must outlive the thread pool run
        QThreadPool.globalInstance().start(worker)

    @staticmethod
    def _fetch_page_worker(generation: int, conn, **kwargs) -> tuple[int, list, int, int, int]:
        # .cursor() called here, i.e. on the worker thread that will use it.
        cursor = conn.cursor()
        rows = query_service.fetch_page(cursor, **kwargs)
        # Total row count for "Page X/Y" and the jump-to-page range. In
        # similarity mode the ranking already covers every matching row
        # (already filtered by exclude_verified/min_confidence -- see
        # compute_similarity_order), so its length is the total for free --
        # no need for a second query.
        similarity_order = kwargs.get("similarity_order")
        label_filter = kwargs.get("label_filter")
        min_confidence = kwargs.get("min_confidence")
        if similarity_order is not None:
            total_rows = len(similarity_order)
        else:
            total_rows = query_service.count_rows(
                cursor,
                label_filter=label_filter,
                exclude_verified=kwargs.get("exclude_verified", False),
                min_confidence=min_confidence,
            )
        # Review-progress counter (see update_status_bar): deliberately
        # scoped to label_filter/min_confidence only, not hide_verified or
        # any active similarity sort -- see count_verified's docstring for
        # why.
        verified_count, unverified_count = query_service.count_verified(
            cursor, label_filter=label_filter, min_confidence=min_confidence
        )
        return generation, rows, total_rows, verified_count, unverified_count

    @Slot(object)
    def _on_page_loaded(self, payload) -> None:
        generation, rows, total_rows, verified_count, unverified_count = payload
        if generation != self._page_load_generation:
            return  # superseded by a newer page/sort/filter change
        self._total_rows = total_rows
        self._verified_count = verified_count
        self._unverified_count = unverified_count
        self._update_page_jump_range()
        self._build_rect_widgets(rows)

    @Slot(tuple)
    def _on_page_load_error(self, err: tuple) -> None:
        logger.error("Failed to load page: %s", err[1])
        QMessageBox.critical(self, "Load Error", f"Failed to load page:\n{err[1]}")

    def _build_rect_widgets(self, rows: list[query_service.RoiRow]) -> None:
        self._roi_loading.cancel_pending()
        self._selection.reset()  # also forgets anchor/nav-cursor -- see its docstring
        self._mosaic_view.clear(self._rect_widgets)

        zoom = self._zoom_percent / 100.0
        self._rect_widgets = [
            RectWidget(
                row,
                self._roi_service,
                clicked_slot=self._on_rect_clicked,
                similarity_sort_slot=self._on_similarity_sort_requested,
                zoom=zoom,
            )
            for row in rows
        ]
        self.render_mosaic()
        self.update_status_bar()

        self._roi_loading_cancel_event = Event()
        self._roi_loading.start_loading(
            rect_widgets=self._rect_widgets,
            on_complete=lambda: None,
            cancel_event=self._roi_loading_cancel_event,
        )

    def render_mosaic(self) -> None:
        visible_widgets = self._mosaic_view.select_visible_widgets(
            all_widgets=self._rect_widgets,
            filters=MosaicVisibilityFilters(hide_verified=self.hide_verified),
        )
        result = self._mosaic_view.render(all_widgets=self._rect_widgets, visible_widgets=visible_widgets)
        self._n_columns = result.columns
        # Kept for keyboard navigation -- see MosaicSelectionCoordinator's
        # visible_widgets_getter docstring for why this must stay in sync
        # with what was actually just laid out, not _rect_widgets.
        self._visible_widgets = visible_widgets

    @Slot(int, int)
    def _on_roi_loading_progress(self, done: int, total: int) -> None:
        if done < total:
            self.status_label.setText(f"Loading thumbnails: {done}/{total}...")
        else:
            self.update_status_bar()

    # -- Sort / filter / zoom / visibility controls --------------------------

    def _on_sort_changed(self, text: str) -> None:
        self.sort_option = text
        self.similarity_order = None
        self.similarity_reference = None
        self.current_page = 0
        self.load_page()

    def _on_clear_similarity_sort(self) -> None:
        """Drop the current similarity ranking and go back to normal
        sort/paging. Changing the "Sort by" dropdown already clears this as
        a side effect, but that's a non-obvious way to escape a similarity
        search that happens to match just one (or a handful of) ROI(s) --
        this button is the explicit, always-available way out."""
        if self.similarity_order is None:
            return
        self.similarity_order = None
        self.similarity_reference = None
        self.current_page = 0
        self.load_page()

    def _on_search(self) -> None:
        """Filter the grid to ROIs whose curated OR original label matches
        the selected/typed concept (see
        ``query_service._label_filter_clause``), jump back to page 1, and
        clear any active similarity sort -- fetch_page's similarity-mode
        branch ignores label_filter entirely, so leaving a stale similarity
        ranking in place would make a brand-new search silently do nothing
        until the ranking was cleared some other way."""
        text = self.search_combo.currentText().strip()
        self.label_filter = text or None
        self.similarity_order = None
        self.similarity_reference = None
        self.current_page = 0
        self.clear_search_button.setEnabled(bool(self.label_filter))
        self.load_page()

    def _on_clear_search(self) -> None:
        """Drop the search filter and go back to showing everything."""
        if not self.label_filter:
            return
        self.search_combo.setCurrentText("")
        self.label_filter = None
        self.current_page = 0
        self.clear_search_button.setEnabled(False)
        self.load_page()

    def _on_zoom_changed(self, value: int) -> None:
        self._zoom_percent = value
        zoom = value / 100.0
        for rect_widget in self._rect_widgets:
            rect_widget.update_zoom(zoom)
        self.render_mosaic()

    def _on_confidence_slider_moved(self, value: int) -> None:
        """Live-update the label only -- actually narrowing the view is an
        explicit "Show" click, not tied to every intermediate drag/keyboard
        step. Unlike the zoom slider (purely client-side, cheap), this
        drives a requery; QSlider's sliderReleased signal doesn't fire for
        keyboard-driven changes, so tying the filter to that would silently
        never apply when adjusted via arrow keys -- an explicit button
        sidesteps that regardless of input method."""
        self.confidence_label.setText(f"Min confidence: {value / 100:.2f}")

    def _on_confidence_filter_clicked(self) -> None:
        """Apply the slider's current position as a minimum-confidence
        floor on what's SHOWN, and requery -- clearing any active
        similarity sort, same as search/sort/hide-verified changes, since
        it also narrows the ranking pool.

        Purely a view constraint, exactly like "Hide verified": it's a
        WHERE clause on SELECT queries only (see
        query_service._build_where) -- nothing here ever UPDATEs the
        stored `confidence` column or any other row data.
        """
        new_value = self.confidence_slider.value() / 100.0
        if new_value == self.min_confidence:
            return
        self.min_confidence = new_value
        self.similarity_order = None
        self.similarity_reference = None
        self.current_page = 0
        self.load_page()

    def _on_hide_verified_toggled(self, checked: bool) -> None:
        """Toggling this now requeries (not just a client-side re-render):
        exclude_verified is applied at the query level (see
        query_service._build_where's docstring for why -- otherwise "Page
        X/Y" and per-page counts include rows that can never actually be
        shown). A stale similarity ranking computed under the old setting
        would still contain/omit the wrong rows, so it's cleared, same as
        a sort or search change."""
        self.hide_verified = checked
        self.similarity_order = None
        self.similarity_reference = None
        self.current_page = 0
        self.load_page()

    # -- Selection / preview --------------------------------------------------

    def _on_rect_clicked(self, rect_widget: RectWidget, event) -> None:
        """Left click: select (Ctrl to toggle, Shift to range-select), as before."""
        modifiers = event.modifiers() if event is not None else Qt.KeyboardModifier.NoModifier

        if modifiers & Qt.KeyboardModifier.ShiftModifier and self._selection.anchor is not None:
            add = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            self._selection.select_range(self._selection.anchor, rect_widget, add=add)
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            if self.selection_model.is_selected(rect_widget):
                self._selection.deselect(rect_widget)
            else:
                self._selection.select(rect_widget, clear=False)
        else:
            self._selection.select(rect_widget, clear=True)

        self._detail_only_active_roi_index = None
        if rect_widget.row.image_path != self._current_detail_image_path:
            self._show_image_and_boxes(rect_widget.row.image_path)
        else:
            # Same image already shown (e.g. selecting a different detection
            # on it, or Ctrl/Cmd-multi-selecting several tiles on it in a
            # row) -- the set of boxes hasn't changed, only which one(s) are
            # active, so just recolor the existing overlays in place. This
            # must NOT be a full rebuild: every additional box in
            # a Cmd-click multi-select would otherwise tear down and
            # recreate every pyqtgraph box overlay on the image, which is
            # both slow and -- confirmed by an actual SIGSEGV under rapid
            # multi-select on a real database -- can crash under repeated
            # rapid teardown/recreate of the underlying Qt graphics items.
            self._detail_view.update_active(self._active_roi_indices())

    @Slot(list)
    def _on_selection_changed(self, selected: list[RectWidget]) -> None:
        self._selection.update_widget_selection_flags(selected)
        self.update_status_bar()
        # Recolor only -- see the comment in _on_rect_clicked's "same image"
        # branch for why this must not rebuild the box overlays.
        self._detail_view.update_active(self._active_roi_indices())

    def _show_image_and_boxes(self, image_path: str) -> None:
        """Show the full image, then overlay every detection on it as an
        editable box -- not just the current mosaic page's rows, since other
        detections on the same frame may be sorted onto a different page."""
        image = self._roi_service.fetch_full_image(image_path)
        if image is None:
            logger.warning("Image not found: %s", image_path)
            return

        self._detail_view.show_image(image_path, image)
        self._current_detail_image_path = image_path

        all_frame_rois = query_service.fetch_rois_for_image(self.conn, image_path)
        self._frame_roi_total = len(all_frame_rois)
        if len(all_frame_rois) > MAX_DETAIL_BOXES:
            active = self._active_roi_indices()
            kept_active = [f for f in all_frame_rois if f.roi_index in active]
            rest = [f for f in all_frame_rois if f.roi_index not in active]
            self._frame_rois = kept_active + rest[: max(0, MAX_DETAIL_BOXES - len(kept_active))]
            logger.warning(
                "Image %s has %d detections; showing %d (capped at %d) to keep the "
                "detail view responsive.",
                image_path,
                len(all_frame_rois),
                len(self._frame_rois),
                MAX_DETAIL_BOXES,
            )
        else:
            self._frame_rois = all_frame_rois

        self._rebuild_detail_boxes()
        # update_status_bar() already ran once via the selection-changed
        # signal that led here, but that was before _frame_roi_total above
        # was known -- run it again so a capped-boxes note actually shows.
        self.update_status_bar()

    def _active_roi_indices(self) -> set[int]:
        active = {rw.roi_index for rw in self.selection_model.selected}
        if self._detail_only_active_roi_index is not None:
            active.add(self._detail_only_active_roi_index)
        return active

    def _rebuild_detail_boxes(self) -> None:
        """Tear down and recreate every box overlay for the current image.

        Only call this when the set of detections being shown actually
        changed (i.e. after fetching a new image's rows in
        _show_image_and_boxes). For a selection-only change, call
        ``self._detail_view.update_active(...)`` instead -- see the comment
        in _on_rect_clicked for why: rebuilding on every selection change
        (as an earlier revision did) is both slow and was the actual cause
        of a SIGSEGV under rapid Ctrl/Cmd-multi-select on a real database.
        """
        self._detail_view.set_boxes(
            self._frame_rois,
            self._active_roi_indices(),
            on_clicked=self._on_detail_box_clicked,
            on_changed=self._on_detail_box_changed,
            on_delete=self._on_detail_box_delete_requested,
        )

    def _find_rect_widget(self, roi_index: int) -> RectWidget | None:
        """Return the loaded tile for *roi_index*, if its page is currently loaded."""
        for rect_widget in self._rect_widgets:
            if rect_widget.roi_index == roi_index:
                return rect_widget
        return None

    def _on_detail_box_clicked(self, roi_index: int, event) -> None:
        """A box was clicked (not dragged) in the detail view."""
        rect_widget = self._find_rect_widget(roi_index)
        if rect_widget is not None:
            # Loaded on the current page -- route through the normal tile
            # click so grid selection (and Ctrl/Shift semantics) stay in sync.
            self._detail_only_active_roi_index = None
            self._on_rect_clicked(rect_widget, event)
        else:
            # Row lives on another page: just highlight it here. Quick-label/
            # delete still act on the grid selection, so labeling this
            # particular detection means paging to it -- an acceptable edge
            # case, not the primary workflow.
            self._detail_only_active_roi_index = roi_index
            self._detail_view.update_active(self._active_roi_indices())

    def _on_detail_box_changed(
        self, roi_index: int, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> None:
        """A box was dragged/resized and released: persist immediately."""
        image = (
            self._roi_service.fetch_full_image(self._current_detail_image_path)
            if self._current_detail_image_path
            else None
        )
        roi_blob = crop_and_encode(image, x_min, y_min, x_max, y_max) if image is not None else None

        try:
            annotation_service.update_bbox(self.conn, roi_index, x_min, y_min, x_max, y_max, roi_blob)
        except Exception:
            logger.exception("Error updating bounding box for ROI #%s", roi_index)
            self.status_label.setText(f"Failed to save bounding box for ROI #{roi_index} -- see log.")
            return

        for frame_roi in self._frame_rois:
            if frame_roi.roi_index == roi_index:
                frame_roi.x_min, frame_roi.y_min = x_min, y_min
                frame_roi.x_max, frame_roi.y_max = x_max, y_max
                break

        rect_widget = self._find_rect_widget(roi_index)
        if rect_widget is not None:
            rect_widget.row.x_min, rect_widget.row.y_min = x_min, y_min
            rect_widget.row.x_max, rect_widget.row.y_max = x_max, y_max
            rect_widget.row.roi_blob = roi_blob
            rect_widget.request_roi_refresh()  # redecode the tile thumbnail, threaded

        logger.info("Updated bounding box for ROI #%s", roi_index)
        self.status_label.setText(f"Saved bounding box for ROI #{roi_index}.")

    def _on_detail_box_delete_requested(self, roi_index: int) -> None:
        """A box's right-click "Delete" was chosen."""
        confirmed = QMessageBox.question(
            self,
            "Delete ROI",
            f"Permanently delete ROI #{roi_index}? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return

        # This delete path doesn't go through load_page() (see the in-place
        # update comment on the RectWidget branch below), so the review-
        # progress counter needs its own explicit adjustment -- look up the
        # row's verified state before it's gone.
        rect_widget = self._find_rect_widget(roi_index)
        if rect_widget is not None:
            was_verified = rect_widget.row.verified
        else:
            row = self.conn.execute(
                "SELECT verified FROM predictions WHERE roi_index = ?", (roi_index,)
            ).fetchone()
            was_verified = bool(row[0]) if row else False

        annotation_service.delete_rois(self.conn, [roi_index])
        if was_verified:
            self._verified_count = max(0, self._verified_count - 1)
        else:
            self._unverified_count = max(0, self._unverified_count - 1)
        self._frame_rois = [f for f in self._frame_rois if f.roi_index != roi_index]
        self._frame_roi_total = max(0, self._frame_roi_total - 1)
        self._detail_view.remove_box(roi_index)
        if self._detail_only_active_roi_index == roi_index:
            self._detail_only_active_roi_index = None

        # rect_widget (looked up above, before the delete) is still valid --
        # nothing in between touched self._rect_widgets.
        if rect_widget is not None:
            if rect_widget in self.selection_model.selected:
                self.selection_model.remove(rect_widget)
            rect_widget.hide()  # see MosaicView.clear()'s docstring: layout
            rect_widget.setParentItem(None)  # detach from the scene graph -- see MosaicView.clear()
            rect_widget.cleanup()  # break the reference cycle -- see RectWidget.cleanup()
            self._rect_widgets.remove(rect_widget)  # removal alone won't hide a stale tile
            self.render_mosaic()

        logger.info("Deleted ROI #%s via the detail view.", roi_index)
        self.status_label.setText(f"Deleted ROI #{roi_index}.")
        self._refresh_verified_count_label()

    def _on_similarity_sort_requested(self, rect_widget: RectWidget, same_label_only: bool) -> None:
        """Right-click menu action: re-rank every ROI (optionally restricted to
        the clicked tile's own label) by embedding similarity to it."""
        roi_index = rect_widget.roi_index
        if same_label_only:
            # Both labels are always shown now (no display-mode toggle), so
            # there's no single "currently displayed" label to key off of
            # any more -- prefer the curated new_label when the clicked tile
            # has one (the more specific, human-reviewed grouping), falling
            # back to the raw original_label pre-clustering, when new_label
            # is still blank for every row.
            if rect_widget.row.label:
                label_filter = rect_widget.row.label
                label_mode = "new"
            else:
                label_filter = rect_widget.row.original_label
                label_mode = "original"
        else:
            label_filter = self.label_filter
            label_mode = "new"

        # See load_page()'s comment: self.conn (not a cursor made from it on
        # this thread) is passed through, and a real bound method -- not a
        # lambda -- is connected so PySide correctly queues delivery onto
        # the GUI thread. exclude_verified/min_confidence narrow the
        # ranking pool itself (see compute_similarity_order's docstring for
        # why that has to happen here, not just as a display-time filter).
        worker = Worker(
            self._compute_similarity_worker,
            roi_index,
            self.conn,
            label_filter,
            label_mode,
            self.hide_verified,
            self.min_confidence,
        )
        worker.signals.result.connect(self._on_similarity_computed)
        worker.signals.error.connect(self._on_similarity_error)
        self._workers.append(worker)  # see __init__ comment: must outlive the thread pool run
        QThreadPool.globalInstance().start(worker)
        self.status_label.setText(f"Sorting by similarity to ROI #{roi_index}...")

    @staticmethod
    def _compute_similarity_worker(
        roi_index: int,
        conn,
        label_filter: str | None,
        label_mode: str,
        exclude_verified: bool,
        min_confidence: float | None,
    ):
        # .cursor() called here, i.e. on the worker thread that will use it.
        order = query_service.compute_similarity_order(
            conn.cursor(),
            roi_index,
            label_filter,
            label_mode,
            exclude_verified=exclude_verified,
            min_confidence=min_confidence,
        )
        return roi_index, order

    @Slot(object)
    def _on_similarity_computed(self, payload) -> None:
        roi_index, order = payload
        if order is None:
            QMessageBox.warning(
                self, "No Embedding", f"ROI #{roi_index} has no embedding yet -- run `mbariml embed` first."
            )
            self.update_status_bar()
            return
        self.similarity_order = order
        self.similarity_reference = roi_index
        self.current_page = 0
        logger.info("Sorted %d ROI(s) by similarity to ROI #%s", len(order), roi_index)
        self.load_page()

    @Slot(tuple)
    def _on_similarity_error(self, err: tuple) -> None:
        logger.error("Similarity sort failed: %s", err[1])
        QMessageBox.critical(self, "Error", f"Similarity sort failed:\n{err[1]}")

    # -- Labeling (in-place, no grid rebuild) ----------------------------------

    def update_labels(self) -> None:
        """Apply the typed-or-picked label to the selected ROIs."""
        new_label = self.label_combo.currentText().strip()
        if not new_label:
            logger.info("No label entered.")
            return
        self._apply_label_to_selected(new_label)
        self.label_combo.clearEditText()

    def _apply_label_to_selected(self, new_label: str) -> None:
        """Write new_label to the database for every selected ROI -- which
        also marks them verified (see ``annotation_service.apply_label``:
        labeling something is itself an act of review) -- then update just
        those tiles' captions/badges in place -- no requery, no rebuild."""
        selected = self.selection_model.selected
        if not selected:
            logger.info("No ROIs selected.")
            return

        roi_indices = [rw.roi_index for rw in selected]
        try:
            annotation_service.apply_label(self.conn, roi_indices, new_label)
        except Exception:
            logger.exception("Error applying label '%s'", new_label)
            return

        # Count *before* mutating row.verified below, so this reflects how
        # many actually flip from unverified -- some selected ROIs may
        # already have been verified, which shouldn't double-count.
        newly_verified = sum(1 for rw in selected if not rw.row.verified)
        self._verified_count += newly_verified
        self._unverified_count = max(0, self._unverified_count - newly_verified)

        for rect_widget in selected:
            rect_widget.set_label(new_label)
            rect_widget.set_verified(True)
            self._detail_view.update_label(rect_widget.roi_index, new_label)

        logger.info("Applied label '%s' to %d ROI(s) (also verified).", new_label, len(selected))
        self.selection_model.clear()
        self._refresh_known_labels()
        if self.hide_verified:
            self.render_mosaic()  # newly-verified tiles may need to disappear now
        self.update_status_bar()

    def verify_selected(self) -> None:
        """Mark every selected ROI as verified."""
        self._set_verified_for_selected(True)

    def unverify_selected(self) -> None:
        """Clear the verified flag on every selected ROI -- the explicit way
        back after ``verify_selected()`` or an auto-verifying label apply."""
        self._set_verified_for_selected(False)

    def _set_verified_for_selected(self, verified: bool) -> None:
        selected = self.selection_model.selected
        if not selected:
            logger.info("No ROIs selected.")
            return

        roi_indices = [rw.roi_index for rw in selected]
        try:
            annotation_service.set_verified(self.conn, roi_indices, verified)
        except Exception:
            logger.exception("Error setting verified=%s", verified)
            return

        # Count *before* mutating row.verified below -- set_verified() sets
        # every selected ROI to the same target state regardless of its
        # prior value, so some may already have been there; only count the
        # ones that actually flip.
        flipped = sum(1 for rw in selected if bool(rw.row.verified) != verified)
        if verified:
            self._verified_count += flipped
            self._unverified_count = max(0, self._unverified_count - flipped)
        else:
            self._verified_count = max(0, self._verified_count - flipped)
            self._unverified_count += flipped

        for rect_widget in selected:
            rect_widget.set_verified(verified)

        logger.info("%s %d ROI(s).", "Verified" if verified else "Unverified", len(selected))
        if self.hide_verified:
            self.render_mosaic()  # verified/unverified tiles may need to appear/disappear now
        self.update_status_bar()

    def delete_selected_rois(self) -> None:
        """Delete the selected ROIs from the database, after confirmation."""
        selected = self.selection_model.selected
        if not selected:
            logger.info("No ROIs selected.")
            return

        count = len(selected)
        confirmed = QMessageBox.question(
            self,
            "Delete ROIs",
            f"Permanently delete {count} selected ROI(s)? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return

        roi_indices = [rw.roi_index for rw in selected]
        annotation_service.delete_rois(self.conn, roi_indices)
        logger.info("Deleted %d ROI(s).", count)

        deleted = set(roi_indices)
        self._frame_rois = [f for f in self._frame_rois if f.roi_index not in deleted]
        self._frame_roi_total = max(0, self._frame_roi_total - len(deleted))
        for roi_index in deleted:
            self._detail_view.remove_box(roi_index)

        self.load_page()

    def unselect_all(self) -> None:
        self.selection_model.clear()

    # -- Page navigation --------------------------------------------------

    def next_page(self) -> None:
        self.current_page += 1
        self.load_page()

    def previous_page(self) -> None:
        if self.current_page > 0:
            self.current_page -= 1
            self.load_page()

    def _on_jump_to_page(self) -> None:
        """Jump straight to the page typed into the spinbox (button click or
        Enter/focus-loss in the spinbox -- both can fire for one "type a
        number then click Go" interaction, so this is a no-op if already on
        the target page rather than double-loading it)."""
        target_page = self.page_jump_spinbox.value() - 1
        if target_page == self.current_page:
            return
        self.current_page = target_page
        self.load_page()


__all__ = ["MainWindow"]
