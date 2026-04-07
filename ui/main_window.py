from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QToolBar,
)

import config
import core.consolidation as consolidation
from core.database import DatabaseManager
from core.faiss_index import FAISSIndex
from ui.canvas import WorkbenchCanvas
from ui.cluster_panel import ClusterPanel
from ui.consolidation_dialog import ConsolidationDialog
from ui.scan_dialog import ScanDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceGraph Manager")
        self.resize(1280, 800)
        self.setStyleSheet("background: #1a1a2a; color: #ddd;")

        self.db = DatabaseManager(config.DB_PATH)
        self.faiss = FAISSIndex()
        self.faiss.load(config.FAISS_INDEX_PATH)

        self._build_toolbar()
        self._build_canvas()
        self._build_cluster_panel()
        self._build_statusbar()
        self._connect_signals()

        self._canvas.load_from_db()
        self._update_status()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setStyleSheet("QToolBar { background: #1e1e2e; border-bottom: 1px solid #333; spacing: 4px; }")
        self.addToolBar(tb)

        self._act_scan = QAction("Scan Folder", self)
        self._act_scan.setShortcut(QKeySequence("Ctrl+O"))
        self._act_scan.triggered.connect(self._on_scan)
        tb.addAction(self._act_scan)

        tb.addSeparator()

        self._act_consolidate = QAction("Consolidate…", self)
        self._act_consolidate.setShortcut(QKeySequence("Ctrl+M"))
        self._act_consolidate.triggered.connect(self._on_consolidate)
        tb.addAction(self._act_consolidate)

        self._act_undo = QAction("Undo Last Consolidation", self)
        self._act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self._act_undo.triggered.connect(self._on_undo)
        self._act_undo.setEnabled(config.UNDO_LOG_PATH.exists())
        tb.addAction(self._act_undo)

        tb.addSeparator()

        help_label = QLabel(
            "  Drag between faces to connect  |  Double-click edge to remove  |  Scroll to zoom  |  Ctrl+O scan  Ctrl+M merge"
        )
        help_label.setStyleSheet("color: #666; font-size: 10px;")
        tb.addWidget(help_label)

    def _build_canvas(self):
        self._canvas = WorkbenchCanvas(self.db)
        self.setCentralWidget(self._canvas)

    def _build_cluster_panel(self):
        self._panel = ClusterPanel(self.db)
        dock = QDockWidget("Cluster Inspector", self)
        dock.setWidget(self._panel)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setStyleSheet("QDockWidget { color: #aaa; }")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_statusbar(self):
        self._statusbar = QStatusBar()
        self._statusbar.setStyleSheet("background: #1e1e2e; color: #888; font-size: 10px;")
        self.setStatusBar(self._statusbar)

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        # Canvas → panel
        self._canvas.identity_clicked.connect(self._panel.load_cluster)

        # Canvas connection events → FAISS + status
        self._canvas.connection_added.connect(lambda a, b: self._update_status())
        self._canvas.connection_removed.connect(lambda a, b: self._update_status())

        # Panel operations → canvas refresh
        self._panel.cluster_split.connect(self._on_cluster_split)
        self._panel.face_ignored.connect(self._on_face_ignored)
        self._panel.key_rep_toggled.connect(self._on_key_rep_toggled)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_scan(self):
        dlg = ScanDialog(self.db, self)
        dlg.exec()

        # Rebuild FAISS and canvas after scan
        self.faiss.build_from_db(self.db)
        self.faiss.save(config.FAISS_INDEX_PATH)
        self._canvas.load_from_db()
        self._update_status()

    def _on_consolidate(self):
        dlg = ConsolidationDialog(self.db, self)
        dlg.exec()
        self._act_undo.setEnabled(config.UNDO_LOG_PATH.exists())

    def _on_undo(self):
        log = config.UNDO_LOG_PATH
        if not log.exists():
            QMessageBox.information(self, "Nothing to Undo", "No undo log found.")
            return
        confirm = QMessageBox.question(
            self,
            "Undo Consolidation",
            f"Reverse all moves recorded in:\n{log}\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            consolidation.undo(log)
            QMessageBox.information(self, "Done", "Consolidation reversed successfully.")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
        self._act_undo.setEnabled(config.UNDO_LOG_PATH.exists())

    # ── Panel callbacks → canvas ──────────────────────────────────────────────

    def _on_cluster_split(self, old_cid: int, new_cid: int):
        old_identity = self.db.get_identity(old_cid)
        new_identity = self.db.get_identity(new_cid)

        if old_identity:
            self._canvas.refresh_identity_thumbnail(old_cid, old_identity)

        if new_identity:
            self._canvas.add_identity_node(new_cid, new_identity["folder_id"], new_identity)

        self._canvas._rebuild_group_edges()
        self._update_status()

    def _on_face_ignored(self, face_id: int, cluster_id: int):
        identity = self.db.get_identity(cluster_id)
        if identity and identity.get("sample_count", 0) == 0:
            self._canvas.remove_identity_node(cluster_id)
        elif identity:
            self._canvas.refresh_identity_thumbnail(cluster_id, identity)
        self._update_status()

    def _on_key_rep_toggled(self, cluster_id: int, is_key: bool):
        identity = self.db.get_identity(cluster_id)
        if identity:
            self._canvas.refresh_identity_thumbnail(cluster_id, identity)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _update_status(self):
        folders = self.db.get_all_folders()
        identities = self.db.get_all_identities()
        connections = self.db.get_all_face_connections()
        self._statusbar.showMessage(
            f"Folders: {len(folders)}   Identities: {len(identities)}   "
            f"Connections: {len(connections)}   FAISS vectors: {len(self.faiss)}"
        )
