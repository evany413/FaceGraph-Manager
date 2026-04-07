from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QImage,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

import config

if TYPE_CHECKING:
    from ui.canvas import WorkbenchCanvas


# ── Thumbnail helper ──────────────────────────────────────────────────────────

def _load_thumbnail(file_path: str, bbox_json: str, size: int = config.THUMBNAIL_SIZE) -> QPixmap:
    try:
        x1, y1, x2, y2 = json.loads(bbox_json)
        img = Image.open(file_path).convert("RGB")
        face = img.crop((x1, y1, x2, y2))
        face = face.resize((size, size), Image.LANCZOS)
        data = np.array(face, dtype=np.uint8)
        h, w, _ = data.shape
        qimg = QImage(data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)
    except Exception:
        px = QPixmap(size, size)
        px.fill(QColor(80, 80, 80))
        return px


# ── IdentityNodeItem ──────────────────────────────────────────────────────────

class IdentityNodeItem(QGraphicsObject):
    """
    Represents one identity cluster (one assumed unique person) within a folder.
    Lives as a child of GroupNodeItem.
    """

    SIZE = config.IDENTITY_NODE_SIZE

    def __init__(self, cluster_id: int, folder_id: int, parent: "GroupNodeItem"):
        super().__init__(parent)
        self.cluster_id = cluster_id
        self.folder_id = folder_id
        self._thumbnail: Optional[QPixmap] = None
        self._is_key_rep = False

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def load_thumbnail(self, file_path: Optional[str], bbox_json: Optional[str]):
        if file_path and bbox_json:
            self._thumbnail = _load_thumbnail(file_path, bbox_json)
        else:
            px = QPixmap(self.SIZE, self.SIZE)
            px.fill(QColor(80, 80, 80))
            self._thumbnail = px
        self.update()

    def set_key_representative(self, is_key: bool):
        self._is_key_rep = is_key
        self.update()

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(self.SIZE / 2, self.SIZE / 2))

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.SIZE, self.SIZE)

    def paint(self, painter: QPainter, option, widget=None):
        r = QRectF(0, 0, self.SIZE, self.SIZE)

        if self._thumbnail:
            painter.drawPixmap(r.toRect(), self._thumbnail)
        else:
            painter.fillRect(r, QColor(80, 80, 80))

        # Border
        if self._is_key_rep:
            pen = QPen(QColor(255, 215, 0), 3)  # gold
        else:
            pen = QPen(QColor(200, 200, 200), 1)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(r.adjusted(1, 1, -1, -1))

    def mousePressEvent(self, event):
        # Consumed here to prevent GroupNodeItem from moving on face-node click.
        event.accept()

    def mouseReleaseEvent(self, event):
        event.accept()


# ── GroupNodeItem ─────────────────────────────────────────────────────────────

HEADER_H = 28
PAD = config.GROUP_NODE_PADDING
NODE_W = config.GROUP_NODE_WIDTH
NODE_SIZE = config.IDENTITY_NODE_SIZE
PER_ROW = config.IDENTITY_NODES_PER_ROW


class GroupNodeItem(QGraphicsObject):
    """
    Represents one physical folder. Contains IdentityNodeItems as children.
    Draggable around the canvas.
    """

    def __init__(self, folder_id: int, folder_path: str, image_count: int = 0):
        super().__init__()
        self.folder_id = folder_id
        self.folder_path = folder_path
        self.folder_name = Path(folder_path).name
        self.image_count = image_count
        self._identity_items: list[IdentityNodeItem] = []

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def add_identity(self, item: IdentityNodeItem):
        self._identity_items.append(item)
        self._layout_identities()

    def remove_identity(self, cluster_id: int):
        self._identity_items = [i for i in self._identity_items if i.cluster_id != cluster_id]
        self._layout_identities()

    def get_identity(self, cluster_id: int) -> Optional[IdentityNodeItem]:
        for item in self._identity_items:
            if item.cluster_id == cluster_id:
                return item
        return None

    def _layout_identities(self):
        for idx, item in enumerate(self._identity_items):
            row = idx // PER_ROW
            col = idx % PER_ROW
            x = PAD + col * (NODE_SIZE + PAD)
            y = HEADER_H + PAD + row * (NODE_SIZE + PAD)
            item.setPos(x, y)
        self.prepareGeometryChange()
        self.update()

    def _body_height(self) -> float:
        n = len(self._identity_items)
        rows = max(1, (n + PER_ROW - 1) // PER_ROW)
        return HEADER_H + PAD + rows * (NODE_SIZE + PAD)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, NODE_W, self._body_height())

    def scene_center(self) -> QPointF:
        r = self.boundingRect()
        return self.mapToScene(r.center())

    def paint(self, painter: QPainter, option, widget=None):
        h = self._body_height()
        body = QRectF(0, 0, NODE_W, h)

        # Background
        painter.setBrush(QBrush(QColor(40, 40, 50)))
        painter.setPen(QPen(QColor(100, 100, 130), 1))
        painter.drawRoundedRect(body, 6, 6)

        # Header
        header = QRectF(0, 0, NODE_W, HEADER_H)
        painter.setBrush(QBrush(QColor(60, 60, 80)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(header, 6, 6)
        painter.drawRect(QRectF(0, HEADER_H / 2, NODE_W, HEADER_H / 2))

        # Folder name + image count
        painter.setPen(QPen(QColor(220, 220, 220)))
        font = QFont()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        label = f"{self.folder_name}  ({self.image_count} imgs)"
        painter.drawText(
            QRectF(PAD, 0, NODE_W - PAD * 2, HEADER_H),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            label,
        )

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            scene = self.scene()
            if scene:
                for view in scene.views():
                    if hasattr(view, "_update_all_edges"):
                        view._update_all_edges()
        return super().itemChange(change, value)


# ── Edge items ────────────────────────────────────────────────────────────────

class FaceEdgeItem(QGraphicsItem):
    """Line between two IdentityNodeItems (user-drawn 'same person' connection)."""

    def __init__(self, source: IdentityNodeItem, dest: IdentityNodeItem):
        super().__init__()
        self.source = source
        self.dest = dest
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

    def endpoints(self) -> tuple[QPointF, QPointF]:
        return self.source.scene_center(), self.dest.scene_center()

    def boundingRect(self) -> QRectF:
        p1, p2 = self.endpoints()
        return QRectF(p1, p2).normalized().adjusted(-4, -4, 4, 4)

    def paint(self, painter: QPainter, option, widget=None):
        p1, p2 = self.endpoints()
        pen = QPen(QColor(100, 180, 255), 2)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

    def contains(self, point: QPointF) -> bool:
        p1, p2 = self.endpoints()
        # Simple hit test: distance from point to segment < 6px
        dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return False
        t = max(0, min(1, ((point.x() - p1.x()) * dx + (point.y() - p1.y()) * dy) / length_sq))
        proj_x = p1.x() + t * dx
        proj_y = p1.y() + t * dy
        dist_sq = (point.x() - proj_x) ** 2 + (point.y() - proj_y) ** 2
        return dist_sq < 36


class GroupEdgeItem(QGraphicsItem):
    """Derived dashed line between two GroupNodeItems."""

    def __init__(self, source: GroupNodeItem, dest: GroupNodeItem):
        super().__init__()
        self.source = source
        self.dest = dest
        self.setZValue(-2)

    def endpoints(self) -> tuple[QPointF, QPointF]:
        return self.source.scene_center(), self.dest.scene_center()

    def boundingRect(self) -> QRectF:
        p1, p2 = self.endpoints()
        return QRectF(p1, p2).normalized().adjusted(-4, -4, 4, 4)

    def paint(self, painter: QPainter, option, widget=None):
        p1, p2 = self.endpoints()
        pen = QPen(QColor(255, 160, 50), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
