from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

import config
import core.consolidation as consolidation
from core.database import DatabaseManager


class ConsolidationDialog(QDialog):
    def __init__(self, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db
        self._moves: list[dict] = []

        self.setWindowTitle("Consolidate Folders")
        self.resize(760, 480)
        self.setStyleSheet("background: #1e1e2e; color: #ddd;")

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Output directory picker
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output directory:"))
        self._dir_edit = QLineEdit()
        self._dir_edit.setStyleSheet("background: #2a2a3a; color: #eee; border: 1px solid #555; padding: 3px;")
        self._dir_edit.setPlaceholderText("Select destination folder…")
        dir_row.addWidget(self._dir_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        browse_btn.setStyleSheet("background: #3a3a5a; color: #eee; padding: 4px 8px;")
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._generate_preview)
        preview_btn.setStyleSheet("background: #2a4a6a; color: #eee; padding: 4px 12px;")
        layout.addWidget(preview_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # Preview table
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Parent", "Folder", "Source", "Destination"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setStyleSheet(
            "QTableWidget { background: #1e1e2e; color: #ddd; gridline-color: #444; }"
            "QHeaderView::section { background: #2a2a3a; color: #aaa; border: 1px solid #444; }"
        )
        layout.addWidget(self._table)

        # Summary label
        self._summary = QLabel("")
        self._summary.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self._summary)

        # Buttons
        btn_box = QDialogButtonBox()
        self._commit_btn = btn_box.addButton("Commit", QDialogButtonBox.ButtonRole.AcceptRole)
        self._commit_btn.setEnabled(False)
        self._commit_btn.setStyleSheet("background: #2a6a2a; color: #eee; padding: 4px 12px;")
        cancel_btn = btn_box.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        cancel_btn.setStyleSheet("background: #4a3a3a; color: #eee; padding: 4px 12px;")
        btn_box.accepted.connect(self._on_commit)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._dir_edit.setText(d)

    def _generate_preview(self):
        out_dir = self._dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Missing Directory", "Please select an output directory first.")
            return

        self._moves = consolidation.preview(self.db, out_dir)
        self._table.setRowCount(0)

        if not self._moves:
            self._summary.setText("No connected folder groups found. Draw face connections on the canvas first.")
            self._commit_btn.setEnabled(False)
            return

        for move in self._moves:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(move["parent"]))
            self._table.setItem(row, 1, QTableWidgetItem(move["folder_name"]))
            self._table.setItem(row, 2, QTableWidgetItem(move["source"]))
            self._table.setItem(row, 3, QTableWidgetItem(move["destination"]))

        self._table.resizeColumnsToContents()
        self._summary.setText(f"{len(self._moves)} folder(s) will be moved.")
        self._commit_btn.setEnabled(True)

    def _on_commit(self):
        if not self._moves:
            return

        errors = consolidation.check_preconditions(self._moves)
        if errors:
            QMessageBox.critical(self, "Cannot Proceed", "\n".join(errors))
            return

        confirm = QMessageBox.question(
            self,
            "Confirm",
            f"This will move {len(self._moves)} folder(s). An undo log will be saved to:\n"
            f"{config.UNDO_LOG_PATH}\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            log_path = consolidation.commit(self._moves)
            QMessageBox.information(
                self,
                "Done",
                f"Folders consolidated successfully.\nUndo log: {log_path}",
            )
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Consolidation failed:\n{exc}")
