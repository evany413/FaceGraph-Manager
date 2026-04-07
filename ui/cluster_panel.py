from __future__ import annotations

import json
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import config
from core.database import DatabaseManager
import core.clustering as clustering


THUMB = 80


def _face_pixmap(file_path: str, bbox_json: str) -> QPixmap:
    try:
        x1, y1, x2, y2 = json.loads(bbox_json)
        img = Image.open(file_path).convert("RGB")
        face = img.crop((x1, y1, x2, y2)).resize((THUMB, THUMB), Image.LANCZOS)
        data = np.array(face, dtype=np.uint8)
        qimg = QImage(data.tobytes(), THUMB, THUMB, THUMB * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)
    except Exception:
        px = QPixmap(THUMB, THUMB)
        px.fill(QColor(80, 80, 80))
        return px


class FaceTile(QFrame):
    ignore_requested = Signal(int)  # face_id

    def __init__(self, face: dict):
        super().__init__()
        self.face_id = face["face_id"]
        self._selected = False

        self.setFixedSize(THUMB + 4, THUMB + 28)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet("background: #2a2a3a; border: 1px solid #555;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._img_label = QLabel()
        self._img_label.setFixedSize(THUMB, THUMB)
        self._img_label.setPixmap(_face_pixmap(face["file_path"], face["bounding_box"]))
        layout.addWidget(self._img_label)

        self._check = QCheckBox("Select")
        self._check.setStyleSheet("color: #ccc; font-size: 9px;")
        layout.addWidget(self._check)

    def is_selected(self) -> bool:
        return self._check.isChecked()

    def face_id_val(self) -> int:
        return self.face_id


class ClusterPanel(QWidget):
    """
    Right-side panel showing all faces in the currently selected identity cluster.
    Provides Split and Ignore operations.
    """

    cluster_split = Signal(int, int)   # old_cluster_id, new_cluster_id
    face_ignored = Signal(int, int)    # face_id, cluster_id
    name_changed = Signal(int, str)    # cluster_id, new_name
    key_rep_toggled = Signal(int, bool)  # cluster_id, is_key

    def __init__(self, db: DatabaseManager):
        super().__init__()
        self.db = db
        self._cluster_id: Optional[int] = None
        self._tiles: list[FaceTile] = []

        self.setMinimumWidth(240)
        self.setMaximumWidth(320)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #1e1e2e; color: #ddd;")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Title
        self._title = QLabel("No cluster selected")
        self._title.setStyleSheet("font-weight: bold; font-size: 11px; color: #aaa;")
        root.addWidget(self._title)

        # Name field
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("optional label")
        self._name_edit.setStyleSheet("background: #2a2a3a; color: #eee; border: 1px solid #555;")
        self._name_edit.editingFinished.connect(self._on_name_changed)
        name_row.addWidget(self._name_edit)
        root.addLayout(name_row)

        # Key representative toggle
        self._key_rep_check = QCheckBox("Key Representative")
        self._key_rep_check.setStyleSheet("color: #ffd700;")
        self._key_rep_check.toggled.connect(self._on_key_rep_toggled)
        root.addWidget(self._key_rep_check)

        # Scroll area for face tiles
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        self._tile_container = QWidget()
        self._tile_layout = _WrapLayout(self._tile_container)
        scroll.setWidget(self._tile_container)
        root.addWidget(scroll)

        # Action buttons
        btn_row = QHBoxLayout()
        self._split_btn = QPushButton("Split Selected")
        self._split_btn.setEnabled(False)
        self._split_btn.clicked.connect(self._on_split)
        self._split_btn.setStyleSheet("background: #3a5a8a; color: #eee; padding: 4px;")

        self._ignore_btn = QPushButton("Ignore Selected")
        self._ignore_btn.setEnabled(False)
        self._ignore_btn.clicked.connect(self._on_ignore)
        self._ignore_btn.setStyleSheet("background: #6a3a3a; color: #eee; padding: 4px;")

        btn_row.addWidget(self._split_btn)
        btn_row.addWidget(self._ignore_btn)
        root.addLayout(btn_row)

    # ── Public ────────────────────────────────────────────────────────────────

    def load_cluster(self, cluster_id: int):
        self._cluster_id = cluster_id
        identity = self.db.get_identity(cluster_id)
        if not identity:
            self.clear()
            return

        self._title.setText(f"Cluster #{cluster_id}  ({identity['sample_count']} faces)")
        self._name_edit.setText(identity.get("name") or "")
        self._key_rep_check.blockSignals(True)
        self._key_rep_check.setChecked(bool(identity.get("is_key_representative")))
        self._key_rep_check.blockSignals(False)

        self._clear_tiles()
        faces = self.db.get_faces_by_cluster(cluster_id)
        for face in faces:
            tile = FaceTile(face)
            tile._check.toggled.connect(self._update_button_states)
            self._tiles.append(tile)
            self._tile_layout.addWidget(tile)

        self._update_button_states()

    def clear(self):
        self._cluster_id = None
        self._title.setText("No cluster selected")
        self._name_edit.clear()
        self._clear_tiles()
        self._split_btn.setEnabled(False)
        self._ignore_btn.setEnabled(False)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_name_changed(self):
        if self._cluster_id is None:
            return
        name = self._name_edit.text().strip()
        self.db.update_identity(self._cluster_id, name=name or None)
        self.name_changed.emit(self._cluster_id, name)

    def _on_key_rep_toggled(self, checked: bool):
        if self._cluster_id is None:
            return
        self.db.update_identity(self._cluster_id, is_key_representative=int(checked))
        self.key_rep_toggled.emit(self._cluster_id, checked)

    def _on_split(self):
        if self._cluster_id is None:
            return
        selected_ids = [t.face_id_val() for t in self._tiles if t.is_selected()]
        if not selected_ids:
            return
        new_cid = clustering.split_cluster(self._cluster_id, selected_ids, self.db)
        old_cid = self._cluster_id
        self.cluster_split.emit(old_cid, new_cid)
        self.load_cluster(old_cid)

    def _on_ignore(self):
        if self._cluster_id is None:
            return
        for tile in [t for t in self._tiles if t.is_selected()]:
            clustering.ignore_face(tile.face_id_val(), self._cluster_id, self.db)
            self.face_ignored.emit(tile.face_id_val(), self._cluster_id)
        self.load_cluster(self._cluster_id)

    def _update_button_states(self):
        any_selected = any(t.is_selected() for t in self._tiles)
        total = len(self._tiles)
        selected = sum(1 for t in self._tiles if t.is_selected())
        # Can only split if selected is a proper subset
        self._split_btn.setEnabled(any_selected and selected < total)
        self._ignore_btn.setEnabled(any_selected)

    def _clear_tiles(self):
        for tile in self._tiles:
            self._tile_layout.removeWidget(tile)
            tile.deleteLater()
        self._tiles.clear()


class _WrapLayout(QVBoxLayout):
    """Simple vertical layout that wraps tiles in rows."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setContentsMargins(4, 4, 4, 4)
        self.setSpacing(4)
        self._rows: list[QHBoxLayout] = []
        self._current_row: Optional[QHBoxLayout] = None
        self._count_in_row = 0
        self._per_row = 2

    def addWidget(self, widget: QWidget):
        if self._current_row is None or self._count_in_row >= self._per_row:
            self._current_row = QHBoxLayout()
            self._current_row.setSpacing(4)
            self._rows.append(self._current_row)
            super().addLayout(self._current_row)
            self._count_in_row = 0
        self._current_row.addWidget(widget)
        self._count_in_row += 1

    def removeWidget(self, widget: QWidget):
        widget.setParent(None)

    def clear(self):
        for row in self._rows:
            while row.count():
                item = row.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
        self._rows.clear()
        self._current_row = None
        self._count_in_row = 0
