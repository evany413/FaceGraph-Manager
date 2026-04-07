from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsView,
)

from core.database import DatabaseManager
from core.graph import GraphManager
from ui.nodes import (
    FaceEdgeItem,
    GroupEdgeItem,
    GroupNodeItem,
    IdentityNodeItem,
)


class WorkbenchCanvas(QGraphicsView):
    """
    The main interactive canvas.

    - GroupNodeItems are draggable.
    - Drag from one IdentityNodeItem to another to create a face connection.
    - Click an IdentityNodeItem to inspect its cluster.
    - Group edges are derived automatically from face connections.
    """

    identity_clicked = Signal(int)           # cluster_id
    connection_added = Signal(int, int)      # cluster_id_a, cluster_id_b
    connection_removed = Signal(int, int)    # cluster_id_a, cluster_id_b

    def __init__(self, db: DatabaseManager):
        super().__init__()
        self.db = db
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QBrush(QColor(25, 25, 35)))
        self.setScene(self._scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)  # Antialiasing
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Node / edge registries
        self._group_nodes: dict[int, GroupNodeItem] = {}          # folder_id -> node
        self._identity_nodes: dict[int, IdentityNodeItem] = {}    # cluster_id -> node
        self._face_edges: dict[tuple[int, int], FaceEdgeItem] = {}  # (cid_a, cid_b) -> edge
        self._group_edges: list[GroupEdgeItem] = []

        # Drag-to-connect state
        self._drag_source: Optional[IdentityNodeItem] = None
        self._drag_start_pos: Optional[QPointF] = None
        self._is_dragging_connection = False
        self._temp_line: Optional[QGraphicsLineItem] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load_from_db(self):
        """Rebuild the entire canvas from the current database state."""
        self._scene.clear()
        self._group_nodes.clear()
        self._identity_nodes.clear()
        self._face_edges.clear()
        self._group_edges.clear()

        folders = self.db.get_all_folders()
        cols = max(1, int(len(folders) ** 0.5) + 1)
        spacing_x, spacing_y = 280, 260

        for idx, folder in enumerate(folders):
            row, col = divmod(idx, cols)
            x = folder["node_x"] or col * spacing_x + 20
            y = folder["node_y"] or row * spacing_y + 20
            self._add_group_node(folder, x, y)

        for connection in self.db.get_all_face_connections():
            self._add_face_edge_item(connection[0], connection[1])

        self._rebuild_group_edges()

    def add_group_node(self, folder: dict):
        """Add a newly scanned folder to the canvas."""
        existing = len(self._group_nodes)
        cols = max(1, int((existing + 1) ** 0.5) + 1)
        col = existing % cols
        row = existing // cols
        x = col * 280 + 20
        y = row * 260 + 20
        self._add_group_node(folder, x, y)

    def add_identity_node(self, cluster_id: int, folder_id: int, identity: dict):
        group = self._group_nodes.get(folder_id)
        if group is None:
            return
        item = IdentityNodeItem(cluster_id, folder_id, group)
        item.load_thumbnail(identity.get("representative_path"), identity.get("representative_bbox"))
        item.set_key_representative(bool(identity.get("is_key_representative")))
        self._identity_nodes[cluster_id] = item
        group.add_identity(item)
        self._scene.addItem(item)  # ensure scene knows about child

    def remove_identity_node(self, cluster_id: int):
        item = self._identity_nodes.pop(cluster_id, None)
        if item is None:
            return
        # Remove all face edges connected to this node
        to_remove = [k for k in self._face_edges if cluster_id in k]
        for key in to_remove:
            edge = self._face_edges.pop(key)
            self._scene.removeItem(edge)
        group = self._group_nodes.get(item.folder_id)
        if group:
            group.remove_identity(cluster_id)
        self._scene.removeItem(item)
        self._rebuild_group_edges()

    def refresh_identity_thumbnail(self, cluster_id: int, identity: dict):
        item = self._identity_nodes.get(cluster_id)
        if item:
            item.load_thumbnail(identity.get("representative_path"), identity.get("representative_bbox"))
            item.set_key_representative(bool(identity.get("is_key_representative")))

    def add_face_connection(self, cluster_id_a: int, cluster_id_b: int):
        key = (min(cluster_id_a, cluster_id_b), max(cluster_id_a, cluster_id_b))
        if key in self._face_edges:
            return
        self._add_face_edge_item(cluster_id_a, cluster_id_b)
        self._rebuild_group_edges()

    def remove_face_connection(self, cluster_id_a: int, cluster_id_b: int):
        key = (min(cluster_id_a, cluster_id_b), max(cluster_id_a, cluster_id_b))
        edge = self._face_edges.pop(key, None)
        if edge:
            self._scene.removeItem(edge)
        self._rebuild_group_edges()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    # ── Mouse interaction ─────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            node = self._identity_node_at(scene_pos)
            if node:
                self._drag_source = node
                self._drag_start_pos = scene_pos
                self._is_dragging_connection = False
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_source and event.buttons() & Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if not self._is_dragging_connection:
                dist = (scene_pos - self._drag_start_pos)
                if abs(dist.x()) + abs(dist.y()) > 8:
                    self._is_dragging_connection = True
                    self._temp_line = QGraphicsLineItem()
                    self._temp_line.setPen(QPen(QColor(100, 200, 255), 2, Qt.PenStyle.DashLine))
                    self._temp_line.setZValue(100)
                    self._scene.addItem(self._temp_line)

            if self._is_dragging_connection and self._temp_line:
                src = self._drag_source.scene_center()
                self._temp_line.setLine(src.x(), src.y(), scene_pos.x(), scene_pos.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_source and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())

            if self._is_dragging_connection:
                target = self._identity_node_at(scene_pos)
                if (
                    target
                    and target != self._drag_source
                    and target.folder_id != self._drag_source.folder_id
                ):
                    self._create_connection(self._drag_source, target)
            else:
                # It was a click — open cluster panel
                self.identity_clicked.emit(self._drag_source.cluster_id)

            self._cancel_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click a face edge to remove it."""
        scene_pos = self.mapToScene(event.pos())
        for key, edge in list(self._face_edges.items()):
            if edge.contains(scene_pos):
                cid_a, cid_b = key
                self.db.remove_face_connection(cid_a, cid_b)
                self.remove_face_connection(cid_a, cid_b)
                self.connection_removed.emit(cid_a, cid_b)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _add_group_node(self, folder: dict, x: float, y: float):
        node = GroupNodeItem(
            folder["folder_id"],
            folder["original_path"],
            folder.get("image_count", 0),
        )
        node.setPos(x, y)
        self._scene.addItem(node)
        self._group_nodes[folder["folder_id"]] = node

        for identity in self.db.get_identities_by_folder(folder["folder_id"]):
            child = IdentityNodeItem(identity["cluster_id"], folder["folder_id"], node)
            child.load_thumbnail(
                identity.get("representative_path"),
                identity.get("representative_bbox"),
            )
            child.set_key_representative(bool(identity.get("is_key_representative")))
            self._identity_nodes[identity["cluster_id"]] = child
            node.add_identity(child)

    def _add_face_edge_item(self, cid_a: int, cid_b: int):
        key = (min(cid_a, cid_b), max(cid_a, cid_b))
        if key in self._face_edges:
            return
        src = self._identity_nodes.get(key[0])
        dst = self._identity_nodes.get(key[1])
        if src and dst:
            edge = FaceEdgeItem(src, dst)
            self._scene.addItem(edge)
            self._face_edges[key] = edge

    def _rebuild_group_edges(self):
        for edge in self._group_edges:
            self._scene.removeItem(edge)
        self._group_edges.clear()

        gm = GraphManager(self.db)
        for fid_a, fid_b in gm.get_folder_edges():
            node_a = self._group_nodes.get(fid_a)
            node_b = self._group_nodes.get(fid_b)
            if node_a and node_b:
                edge = GroupEdgeItem(node_a, node_b)
                self._scene.addItem(edge)
                self._group_edges.append(edge)

    def _update_all_edges(self):
        for edge in self._face_edges.values():
            edge.prepareGeometryChange()
            edge.update()
        for edge in self._group_edges:
            edge.prepareGeometryChange()
            edge.update()

    def _identity_node_at(self, scene_pos: QPointF) -> Optional[IdentityNodeItem]:
        for item in self._scene.items(scene_pos):
            if isinstance(item, IdentityNodeItem):
                return item
        return None

    def _create_connection(self, src: IdentityNodeItem, dst: IdentityNodeItem):
        cid_a, cid_b = src.cluster_id, dst.cluster_id
        if self.db.connection_exists(cid_a, cid_b):
            return
        self.db.add_face_connection(cid_a, cid_b)
        self.add_face_connection(cid_a, cid_b)
        self.connection_added.emit(cid_a, cid_b)

    def _cancel_drag(self):
        if self._temp_line:
            self._scene.removeItem(self._temp_line)
            self._temp_line = None
        self._drag_source = None
        self._drag_start_pos = None
        self._is_dragging_connection = False
