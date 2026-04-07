from pathlib import Path

import faiss
import numpy as np

from core.database import DatabaseManager


class FAISSIndex:
    DIM = 512

    def __init__(self):
        self._index = faiss.IndexFlatIP(self.DIM)
        self._face_ids: list[int] = []

    # ── Build / Persist ───────────────────────────────────────────────────────

    def build_from_db(self, db: DatabaseManager):
        self._index = faiss.IndexFlatIP(self.DIM)
        self._face_ids = []
        for row in db.get_all_embeddings():
            emb = np.frombuffer(row["embedding_vector"], dtype=np.float32).copy()
            self._add_raw(row["face_id"], emb)

    def save(self, path: Path):
        faiss.write_index(self._index, str(path))
        np.save(str(path) + ".ids.npy", np.array(self._face_ids, dtype=np.int64))

    def load(self, path: Path):
        if path.exists():
            self._index = faiss.read_index(str(path))
            ids_path = Path(str(path) + ".ids.npy")
            if ids_path.exists():
                self._face_ids = np.load(str(ids_path)).tolist()

    # ── Operations ────────────────────────────────────────────────────────────

    def add(self, face_id: int, embedding: np.ndarray):
        emb = embedding.astype(np.float32).copy()
        self._add_raw(face_id, emb)

    def search(self, embedding: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Returns list of (face_id, cosine_similarity) sorted descending."""
        if not self._face_ids:
            return []
        vec = embedding.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(vec)
        k = min(k, len(self._face_ids))
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self._face_ids[idx], float(score)))
        return results

    def __len__(self):
        return len(self._face_ids)

    def _add_raw(self, face_id: int, emb: np.ndarray):
        vec = emb.reshape(1, -1)
        faiss.normalize_L2(vec)
        self._index.add(vec)
        self._face_ids.append(face_id)
