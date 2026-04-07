from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from core.database import DatabaseManager
from core.scanner import ScanWorker
import core.clustering as clustering


class ScanDialog(QDialog):
    def __init__(self, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db
        self._worker: ScanWorker | None = None

        self.setWindowTitle("Scan Folder")
        self.resize(560, 380)
        self.setStyleSheet("background: #1e1e2e; color: #ddd;")

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Directory row
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Root folder:"))
        self._dir_edit = QLineEdit()
        self._dir_edit.setStyleSheet("background: #2a2a3a; color: #eee; border: 1px solid #555; padding: 3px;")
        dir_row.addWidget(self._dir_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        browse_btn.setStyleSheet("background: #3a3a5a; color: #eee; padding: 4px 8px;")
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(
            "QProgressBar { background: #2a2a3a; border: 1px solid #555; text-align: center; }"
            "QProgressBar::chunk { background: #2a6aaa; }"
        )
        layout.addWidget(self._progress)

        self._status_label = QLabel("Ready.")
        self._status_label.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self._status_label)

        # Log output
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(200)
        self._log.setStyleSheet("background: #111; color: #9f9; font-family: monospace; font-size: 10px;")
        layout.addWidget(self._log)

        # Buttons
        btn_row = QHBoxLayout()
        self._scan_btn = QPushButton("Start Scan")
        self._scan_btn.clicked.connect(self._start_scan)
        self._scan_btn.setStyleSheet("background: #2a5a2a; color: #eee; padding: 4px 12px;")

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_scan)
        self._cancel_btn.setStyleSheet("background: #5a2a2a; color: #eee; padding: 4px 12px;")

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        self._close_btn.setStyleSheet("background: #3a3a3a; color: #eee; padding: 4px 12px;")

        btn_row.addWidget(self._scan_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select Root Folder")
        if d:
            self._dir_edit.setText(d)

    def _start_scan(self):
        root = self._dir_edit.text().strip()
        if not root:
            QMessageBox.warning(self, "No Folder", "Please select a root folder first.")
            return

        self.db.clear_all()
        self._log.clear()
        self._progress.setValue(0)
        self._scan_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)

        self._worker = ScanWorker(root, self.db)
        self._worker.progress.connect(self._on_progress)
        self._worker.folder_done.connect(self._on_folder_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _cancel_scan(self):
        if self._worker:
            self._worker.abort()

    def _on_progress(self, current: int, total: int, filename: str):
        pct = int(current / total * 100) if total else 0
        self._progress.setValue(pct)
        self._status_label.setText(f"[{current}/{total}] {filename}")
        self._log.appendPlainText(f"  {filename}")

    def _on_folder_done(self, folder_id: int):
        self._log.appendPlainText(f"Clustering folder {folder_id}…")
        clustering.cluster_folder(folder_id, self.db)

    def _on_finished(self):
        self._progress.setValue(100)
        self._status_label.setText("Scan complete.")
        self._scan_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._log.appendPlainText("Done.")

    def _on_error(self, msg: str):
        self._log.appendPlainText(f"ERROR: {msg}")
        self._scan_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
