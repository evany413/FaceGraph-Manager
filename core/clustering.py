import json

import numpy as np
from sklearn.cluster import HDBSCAN

import config
from core.database import DatabaseManager


def cluster_folder(folder_id: int, db: DatabaseManager):
    """Run HDBSCAN on all unassigned faces in a folder and create identity records."""
    faces = db.get_faces_by_folder(folder_id)
    if not faces:
        return

    embeddings = np.array(
        [np.frombuffer(f["embedding_vector"], dtype=np.float32) for f in faces]
    )

    if len(faces) < config.HDBSCAN_MIN_CLUSTER_SIZE:
        # Too few faces — each becomes its own identity
        for face in faces:
            cid = db.create_identity(folder_id)
            _finalize_single(cid, face, embeddings[faces.index(face)], db)
        return

    clusterer = HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)

    unique_labels = set(labels)
    cluster_map: dict[int, int] = {}  # hdbscan label -> db cluster_id

    for label in unique_labels:
        if label == -1:
            # Noise: each face becomes its own ignored-status entry
            for idx in np.where(labels == -1)[0]:
                db.update_face_status(faces[idx]["face_id"], "Ignored")
            continue

        cid = db.create_identity(folder_id)
        cluster_map[label] = cid

        indices = np.where(labels == label)[0]
        member_embeddings = embeddings[indices]
        centroid = _normalized_mean(member_embeddings)

        # Representative = face closest to centroid
        sims = member_embeddings @ centroid
        rep_idx = int(np.argmax(sims))
        rep_face = faces[indices[rep_idx]]

        db.update_identity(
            cid,
            centroid_vector=centroid.tobytes(),
            sample_count=len(indices),
            representative_path=rep_face["file_path"],
            representative_bbox=rep_face["bounding_box"],
        )
        for idx in indices:
            db.update_face_cluster(faces[idx]["face_id"], cid)
            db.update_face_status(faces[idx]["face_id"], "New")


def _finalize_single(cluster_id: int, face: dict, embedding: np.ndarray, db: DatabaseManager):
    emb = embedding / (np.linalg.norm(embedding) or 1.0)
    db.update_identity(
        cluster_id,
        centroid_vector=emb.tobytes(),
        sample_count=1,
        representative_path=face["file_path"],
        representative_bbox=face["bounding_box"],
    )
    db.update_face_cluster(face["face_id"], cluster_id)
    db.update_face_status(face["face_id"], "New")


# ── Cluster operations called from the UI ─────────────────────────────────────

def ignore_face(face_id: int, cluster_id: int, db: DatabaseManager):
    """Remove a face from its cluster and mark it ignored."""
    db.update_face_status(face_id, "Ignored")
    db.update_face_cluster(face_id, None)
    _recalculate_centroid(cluster_id, db)
    _tighten_threshold(cluster_id, db)


def split_cluster(cluster_id: int, face_ids_to_split: list[int], db: DatabaseManager) -> int:
    """
    Move face_ids_to_split into a new identity cluster.
    Returns the new cluster_id.
    """
    identity = db.get_identity(cluster_id)
    if identity is None:
        raise ValueError(f"Cluster {cluster_id} not found")

    new_cid = db.create_identity(identity["folder_id"])

    for fid in face_ids_to_split:
        db.add_negative_sample(fid, cluster_id)
        db.update_face_cluster(fid, new_cid)
        db.update_face_status(fid, "New")

    _recalculate_centroid(cluster_id, db)
    _recalculate_centroid(new_cid, db)
    _tighten_threshold(cluster_id, db)

    return new_cid


def _recalculate_centroid(cluster_id: int, db: DatabaseManager):
    faces = db.get_faces_by_cluster(cluster_id)
    if not faces:
        db.update_identity(cluster_id, centroid_vector=None, sample_count=0,
                           representative_path=None, representative_bbox=None)
        return

    embeddings = np.array(
        [np.frombuffer(f["embedding_vector"], dtype=np.float32) for f in faces]
    )
    centroid = _normalized_mean(embeddings)
    sims = embeddings @ centroid
    rep_idx = int(np.argmax(sims))
    rep = faces[rep_idx]

    db.update_identity(
        cluster_id,
        centroid_vector=centroid.tobytes(),
        sample_count=len(faces),
        representative_path=rep["file_path"],
        representative_bbox=rep["bounding_box"],
    )


def _tighten_threshold(cluster_id: int, db: DatabaseManager):
    identity = db.get_identity(cluster_id)
    if identity is None:
        return
    new_thresh = max(0.3, identity["similarity_threshold"] - config.THRESHOLD_DELTA)
    db.update_identity(cluster_id, similarity_threshold=new_thresh)


def _loosen_threshold(cluster_id: int, db: DatabaseManager):
    identity = db.get_identity(cluster_id)
    if identity is None:
        return
    new_thresh = min(0.95, identity["similarity_threshold"] + config.THRESHOLD_DELTA)
    db.update_identity(cluster_id, similarity_threshold=new_thresh)


def _normalized_mean(embeddings: np.ndarray) -> np.ndarray:
    mean = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean
