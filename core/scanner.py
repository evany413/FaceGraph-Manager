from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

import config
from core.database import DatabaseManager
from core.engine import FaceEngine


class ScanWorker(QThread):
    progress = Signal(int, int, str)   # current, total, current_file
    folder_done = Signal(int)          # folder_id
    finished = Signal()
    error = Signal(str)

    def __init__(self, root_dir: str, db: DatabaseManager):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.db = db
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            engine = FaceEngine()
            self._scan(engine)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _scan(self, engine: FaceEngine):
        root = self.root_dir

        # Each immediate subdirectory = one group/folder node.
        # If the root itself contains images, treat root as a group too.
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        if not subdirs:
            subdirs = [root]

        all_images: list[tuple[Path, Path]] = []  # (folder_dir, image_path)
        for folder_dir in subdirs:
            for ext in config.IMAGE_EXTENSIONS:
                for img_path in folder_dir.rglob(f"*{ext}"):
                    all_images.append((folder_dir, img_path))
                for img_path in folder_dir.rglob(f"*{ext.upper()}"):
                    all_images.append((folder_dir, img_path))

        total = len(all_images)
        if total == 0:
            return

        folder_image_counts: dict[int, int] = {}

        for idx, (folder_dir, img_path) in enumerate(all_images):
            if self._abort:
                break

            self.progress.emit(idx + 1, total, img_path.name)

            folder_id = self.db.add_folder(str(folder_dir))
            folder_image_counts[folder_id] = folder_image_counts.get(folder_id, 0) + 1

            faces = engine.analyze(str(img_path))
            for face in faces:
                emb_bytes = face["embedding"].astype(np.float32).tobytes()
                self.db.add_face(
                    embedding=emb_bytes,
                    file_path=str(img_path),
                    bbox=face["bbox"],
                    folder_id=folder_id,
                    confidence=face["confidence"],
                )

        for folder_id, count in folder_image_counts.items():
            self.db.update_folder_image_count(folder_id, count)
            self.folder_done.emit(folder_id)
