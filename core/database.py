import sqlite3
import json
from pathlib import Path
from typing import Optional


class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._create_tables()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _create_tables(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS folders (
                    folder_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_path TEXT UNIQUE NOT NULL,
                    node_x      REAL DEFAULT 0,
                    node_y      REAL DEFAULT 0,
                    image_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS faces (
                    face_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding_vector BLOB NOT NULL,
                    file_path       TEXT NOT NULL,
                    bounding_box    TEXT NOT NULL,
                    folder_id       INTEGER NOT NULL,
                    cluster_id      INTEGER,
                    confidence_score REAL NOT NULL,
                    status          TEXT DEFAULT 'New',
                    FOREIGN KEY (folder_id) REFERENCES folders(folder_id)
                );

                CREATE TABLE IF NOT EXISTS identities (
                    cluster_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    folder_id           INTEGER NOT NULL,
                    name                TEXT,
                    representative_path TEXT,
                    representative_bbox TEXT,
                    centroid_vector     BLOB,
                    sample_count        INTEGER DEFAULT 0,
                    similarity_threshold REAL DEFAULT 0.6,
                    is_key_representative INTEGER DEFAULT 0,
                    FOREIGN KEY (folder_id) REFERENCES folders(folder_id)
                );

                CREATE TABLE IF NOT EXISTS face_connections (
                    connection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id_a  INTEGER NOT NULL,
                    cluster_id_b  INTEGER NOT NULL,
                    UNIQUE(cluster_id_a, cluster_id_b)
                );

                CREATE TABLE IF NOT EXISTS negative_samples (
                    face_id    INTEGER NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    PRIMARY KEY (face_id, cluster_id)
                );
            """)

    # ── Folders ──────────────────────────────────────────────────────────────

    def add_folder(self, path: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO folders (original_path) VALUES (?)", (path,)
            )
            if cur.lastrowid:
                return cur.lastrowid
            row = conn.execute(
                "SELECT folder_id FROM folders WHERE original_path=?", (path,)
            ).fetchone()
            return row["folder_id"]

    def update_folder_image_count(self, folder_id: int, count: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE folders SET image_count=? WHERE folder_id=?", (count, folder_id)
            )

    def update_folder_position(self, folder_id: int, x: float, y: float):
        with self._conn() as conn:
            conn.execute(
                "UPDATE folders SET node_x=?, node_y=? WHERE folder_id=?", (x, y, folder_id)
            )

    def get_all_folders(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM folders").fetchall()
            return [dict(r) for r in rows]

    def clear_all(self):
        with self._conn() as conn:
            conn.executescript("""
                DELETE FROM negative_samples;
                DELETE FROM face_connections;
                DELETE FROM faces;
                DELETE FROM identities;
                DELETE FROM folders;
            """)

    # ── Faces ─────────────────────────────────────────────────────────────────

    def add_face(
        self,
        embedding: bytes,
        file_path: str,
        bbox: list,
        folder_id: int,
        confidence: float,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO faces
                   (embedding_vector, file_path, bounding_box, folder_id, confidence_score)
                   VALUES (?, ?, ?, ?, ?)""",
                (embedding, file_path, json.dumps(bbox), folder_id, confidence),
            )
            return cur.lastrowid

    def get_faces_by_folder(self, folder_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM faces WHERE folder_id=? AND status != 'Ignored'", (folder_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_faces_by_cluster(self, cluster_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM faces WHERE cluster_id=? AND status != 'Ignored'",
                (cluster_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_embeddings(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT face_id, embedding_vector FROM faces WHERE status != 'Ignored'"
            ).fetchall()
            return [dict(r) for r in rows]

    def update_face_cluster(self, face_id: int, cluster_id: Optional[int]):
        with self._conn() as conn:
            conn.execute(
                "UPDATE faces SET cluster_id=? WHERE face_id=?", (cluster_id, face_id)
            )

    def update_face_status(self, face_id: int, status: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE faces SET status=? WHERE face_id=?", (status, face_id)
            )

    # ── Identities ────────────────────────────────────────────────────────────

    def create_identity(self, folder_id: int) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO identities (folder_id) VALUES (?)", (folder_id,)
            )
            return cur.lastrowid

    def get_identity(self, cluster_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM identities WHERE cluster_id=?", (cluster_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_identities_by_folder(self, folder_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM identities WHERE folder_id=?", (folder_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_identities(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM identities").fetchall()
            return [dict(r) for r in rows]

    def update_identity(self, cluster_id: int, **kwargs):
        if not kwargs:
            return
        cols = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [cluster_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE identities SET {cols} WHERE cluster_id=?", vals)

    def delete_identity(self, cluster_id: int):
        with self._conn() as conn:
            conn.execute("UPDATE faces SET cluster_id=NULL WHERE cluster_id=?", (cluster_id,))
            conn.execute("DELETE FROM face_connections WHERE cluster_id_a=? OR cluster_id_b=?",
                         (cluster_id, cluster_id))
            conn.execute("DELETE FROM negative_samples WHERE cluster_id=?", (cluster_id,))
            conn.execute("DELETE FROM identities WHERE cluster_id=?", (cluster_id,))

    # ── Face Connections ──────────────────────────────────────────────────────

    def add_face_connection(self, cluster_id_a: int, cluster_id_b: int):
        lo, hi = min(cluster_id_a, cluster_id_b), max(cluster_id_a, cluster_id_b)
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO face_connections (cluster_id_a, cluster_id_b) VALUES (?,?)",
                (lo, hi),
            )

    def remove_face_connection(self, cluster_id_a: int, cluster_id_b: int):
        lo, hi = min(cluster_id_a, cluster_id_b), max(cluster_id_a, cluster_id_b)
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM face_connections WHERE cluster_id_a=? AND cluster_id_b=?",
                (lo, hi),
            )

    def get_all_face_connections(self) -> list[tuple[int, int]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT cluster_id_a, cluster_id_b FROM face_connections"
            ).fetchall()
            return [(r["cluster_id_a"], r["cluster_id_b"]) for r in rows]

    def connection_exists(self, cluster_id_a: int, cluster_id_b: int) -> bool:
        lo, hi = min(cluster_id_a, cluster_id_b), max(cluster_id_a, cluster_id_b)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM face_connections WHERE cluster_id_a=? AND cluster_id_b=?",
                (lo, hi),
            ).fetchone()
            return row is not None

    # ── Negative Samples ──────────────────────────────────────────────────────

    def add_negative_sample(self, face_id: int, cluster_id: int):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO negative_samples (face_id, cluster_id) VALUES (?,?)",
                (face_id, cluster_id),
            )

    def get_negative_samples(self, cluster_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT f.* FROM faces f
                   JOIN negative_samples ns ON f.face_id = ns.face_id
                   WHERE ns.cluster_id=?""",
                (cluster_id,),
            ).fetchall()
            return [dict(r) for r in rows]
