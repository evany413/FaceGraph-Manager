# Technical Requirement Specification: FaceGraph Manager

## 1. Project Vision

FaceGraph Manager is a local-first desktop application that uses facial recognition and graph theory to identify shared identities across fragmented photo folders. Users verify and refine AI predictions through an interactive canvas, then consolidate related folders under auto-generated parent directories — preserving all internal folder structure.

---

## 2. Core Functional Requirements

### 2.1 Image Processing & AI Engine

**Recursive Scanning**
- Scan a user-selected root directory recursively for image files (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`).
- Each immediate subfolder under the root is treated as one **Group** (folder node on the canvas).
- Record each folder's path in the database on scan.

**Face Analysis Pipeline (per image)**
1. **Detection**: Locate all faces in the image using RetinaFace (via InsightFace). Discard detections with confidence score below a configurable threshold (default: 0.6).
2. **Alignment**: Normalize facial orientation using detected landmarks.
3. **Embedding**: Extract a 512-dimensional feature vector per face using ArcFace (`buffalo_l` model pack).
4. **Storage**: Persist each embedding, its source file path, bounding box, folder ID, and confidence score to the database.

**Parallelism**
- Face extraction runs via `ProcessPoolExecutor` (not threads) to bypass the Python GIL.
- GPU acceleration is optional and auto-detected at startup (see Section 5.2).

**Initial Clustering**
- After scanning, group all face embeddings within each folder using **HDBSCAN**.
- Each resulting cluster = one **Identity Node** (face node) for that folder.
- Noise points flagged by HDBSCAN (label `-1`) are treated as `Ignored` by default.
- One Identity Node represents one assumed unique person within that folder.

---

### 2.2 Interactive Canvas (The Workbench)

The canvas is the primary UI surface. It has two levels of nodes rendered simultaneously.

**Group Nodes (Folder Level)**
- Each Group Node represents one physical folder.
- Size of the node is proportional to the image count in that folder.
- Rendered as a container that holds its Identity Nodes visually inside.

**Identity Nodes (Face Level)**
- Each Identity Node lives inside a Group Node.
- Represents one identity cluster: one assumed unique person detected within that folder.
- Displays a representative face crop (the detection closest to the cluster centroid).
- Clicking an Identity Node opens a **Cluster Inspection Panel** showing all face crops in that cluster.

**Group Edges (Derived, Not Manual)**
- A Group Edge (line between two Group Nodes) is drawn **automatically** when at least one Identity Node in Group A is connected to at least one Identity Node in Group B.
- Group Edges are never drawn manually by the user.
- Group Edges are removed automatically when all face-level connections between those two groups are removed.

**Face Connections (User-Drawn)**
- The user draws edges between Identity Nodes across different groups to assert "same person."
- Drawing a face connection triggers an immediate canvas refresh (see Section 2.3 — Reactivity).

**Canvas Reactivity**
- Every face connection change (add or remove) triggers a full recalculation of:
  1. Which Group Nodes share a derived Group Edge.
  2. The consolidation preview (which folders will be grouped under which parent).
- The preview updates in real-time without any explicit "refresh" button.

**Representative Picker**
- The user can designate one or more Identity Nodes per Group as "Key Representatives."
- These are highlighted in the canvas and used as the visual identity of that group.

---

### 2.3 Identity Refinement (Cluster Operations)

The AI model (InsightFace/ArcFace) is **frozen** — its weights never change. All "learning" happens in the database via centroid and threshold updates.

**Three cluster-level operations:**

#### Merge
- **Trigger**: User draws a face connection between Identity Node A (in Group X) and Identity Node B (in Group Y).
- **Effect**:
  - The two clusters are merged into one logical identity.
  - The mean centroid is recalculated from all member embeddings.
  - `SampleCount` is updated accordingly.
  - Canvas refreshes: Group X and Group Y now share a Group Edge.

#### Split
- **Trigger**: User opens the Cluster Inspection Panel, selects a subset of face crops, and clicks "Split."
- **Effect**:
  - Selected faces are removed from the original cluster.
  - A new Identity Node is created in the same Group from those faces.
  - Both clusters recalculate their centroids independently.
  - Canvas refreshes: any Group Edges that depended on the split cluster are re-evaluated.

#### Ignore
- **Trigger**: User selects a face crop within the Cluster Inspection Panel and clicks "Ignore."
- **Effect**:
  - The face is marked `Ignored` in the database.
  - It is excluded from centroid calculations and future similarity queries.
  - It no longer appears in the UI.

**Negative Samples**
- When a Merge is undone or a Split is performed, the separated embeddings are recorded as **negative samples** against the original cluster.
- During future similarity queries, a candidate is rejected if it is closer (cosine distance) to any negative sample of an identity than to its centroid.

**Per-Identity Similarity Threshold**
- Each identity cluster stores a `SimilarityThreshold` float (default: 0.6, cosine similarity).
- This threshold tightens when the user frequently splits or rejects faces from that cluster.
- It loosens when the user frequently merges faces into that cluster.
- Adjustment is a small fixed delta (e.g., ±0.02) per user action — no ML required.

---

### 2.4 File Consolidation

**Logical Phase (Preview)**
- The system inspects all connected components in the Group Edge graph using NetworkX.
- Each connected component = one folder group that will share a parent directory.
- Auto-generate sequential parent names: `001`, `002`, `003`, ...
- Display a preview table:

```
Parent  | Folder             | Action
--------|--------------------|--------------------------
001     | /photos/trip_rome  | Move → /output/001/trip_rome
001     | /photos/vacation   | Move → /output/001/vacation
002     | /photos/family     | Move → /output/002/family
```

**Physical Phase (Commit)**
- Executed only after explicit user confirmation.
- Pre-execution checks:
  - Verify write permissions on destination.
  - Verify sufficient disk space.
  - Abort entirely if either check fails — no partial moves.
- Move entire folders (not individual files) to preserve all internal subfolder structure.
- Example:
  ```
  Before:  /photos/trip_rome/day1/img001.jpg
  After:   /output/001/trip_rome/day1/img001.jpg
  ```
- Filename collision within a folder: append incrementing suffix (`img001_1.jpg`, `img001_2.jpg`, ...) until unique.

**Undo**
- Before any move, write a JSON undo log:
  ```json
  [
    {"from": "/output/001/trip_rome", "to": "/photos/trip_rome"},
    {"from": "/output/001/vacation",  "to": "/photos/vacation"}
  ]
  ```
- A dedicated "Undo Last Consolidation" button reads this log and reverses the moves.

---

## 3. Technical Stack

| Category | Component | Selection | Notes |
|---|---|---|---|
| Language | Runtime | Python 3.10+ | |
| AI — Detection | Face Detection | InsightFace (RetinaFace) | ONNX Runtime backend |
| AI — Recognition | Face Embedding | InsightFace (ArcFace, `buffalo_l`) | ONNX Runtime backend, no PyTorch |
| Clustering | Unsupervised Grouping | HDBSCAN | Via `hdbscan` package or scikit-learn 1.3+ |
| Vector Search | Similarity Queries | FAISS (`faiss-cpu`) | Alongside SQLite; `.index` file on disk |
| Metadata DB | Persistence | SQLite | Single-file, no server |
| Graph Logic | Relationship Mapping | NetworkX | Connected components for consolidation |
| GUI Framework | Desktop UI | PySide6 | LGPL license |
| Graph Canvas | Interactive Graph | QGraphicsView | Native Qt, no browser dependency |
| Parallelism | Extraction | `ProcessPoolExecutor` | Bypasses GIL for CPU-bound inference |
| Math / ML | Numerics | NumPy, scikit-learn | |

---

## 4. Data Architecture

### 4.1 SQLite Schema

**Faces**
```
FaceID          INTEGER PRIMARY KEY
EmbeddingVector BLOB            -- 512-d float32, numpy tobytes()
FilePath        TEXT
BoundingBox     TEXT            -- JSON: [x, y, w, h]
FolderID        INTEGER
ClusterID       INTEGER
ConfidenceScore REAL
Status          TEXT            -- New | Verified | Negative | Ignored
```

**Identities**
```
ClusterID           INTEGER PRIMARY KEY
Name                TEXT            -- Optional, user-assigned
RepresentativePath  TEXT
CentroidVector      BLOB            -- 512-d float32, numpy tobytes()
SampleCount         INTEGER         -- Required for incremental mean
SimilarityThreshold REAL            -- Per-identity, default 0.6
```

**Folders**
```
FolderID      INTEGER PRIMARY KEY
OriginalPath  TEXT
NodeX         REAL    -- Canvas position
NodeY         REAL    -- Canvas position
```

**FaceConnections**
```
ConnectionID  INTEGER PRIMARY KEY
ClusterID_A   INTEGER
ClusterID_B   INTEGER
```

**NegativeSamples**
```
FaceID     INTEGER
ClusterID  INTEGER   -- Which identity this face was rejected from
PRIMARY KEY (FaceID, ClusterID)
```

**ConsolidationGroups**
```
GroupID     INTEGER PRIMARY KEY
ParentName  TEXT    -- "001", "002", ...
```

**ConsolidationGroupMembers**
```
GroupID   INTEGER
FolderID  INTEGER
PRIMARY KEY (GroupID, FolderID)
```

### 4.2 FAISS Index

- One `IndexFlatIP` (inner product = cosine similarity on normalized vectors) per session.
- Persisted to `facegraph.index` alongside the SQLite file.
- Rebuilt from the Faces table on startup if the index file is missing.
- Updated incrementally as new embeddings are added.

### 4.3 Algorithm Logic

- **Similarity Metric**: Cosine similarity (vectors L2-normalized before storage and query).
- **Clustering**: HDBSCAN with `min_cluster_size=2`; noise label `-1` → `Ignored`.
- **Centroid Update**: Incremental mean using `SampleCount`:
  ```
  new_centroid = (old_centroid * SampleCount + new_vector) / (SampleCount + 1)
  ```
- **Negative Sample Rejection**: Candidate rejected if `cosine_sim(candidate, negative) > cosine_sim(candidate, centroid)`.
- **Group Edge Derivation**: NetworkX graph of FaceConnections; `connected_components()` determines consolidation groups.

---

## 5. Non-Functional Requirements

### 5.1 Performance
- Extraction pipeline must handle 10,000+ images without blocking the UI (run in background process).
- FAISS query latency must remain under 100ms for up to 100,000 face vectors.
- Canvas must re-render reactively within 500ms of any user action.

### 5.2 GPU / Platform
- ONNX Runtime execution provider is auto-detected at startup:
  ```
  Windows + NVIDIA → CUDAExecutionProvider
  macOS (Apple Silicon) → CoreMLExecutionProvider
  Fallback → CPUExecutionProvider
  ```
- Manual override via `FORCE_DEVICE` in `config.py` (`"cuda"` | `"coreml"` | `"cpu"`).
- FAISS uses `faiss-cpu` on all platforms. GPU FAISS is out of scope.

### 5.3 Privacy
- All processing is local. No images, embeddings, or metadata are transmitted to any external service.
- First run requires internet access to download InsightFace model files (~300MB). All subsequent runs are fully offline.

### 5.4 Reliability
- The Commit phase aborts entirely (zero moves) if any pre-execution check fails.
- An undo log is written before the first file move. If the process is interrupted mid-commit, the undo log can be used for manual recovery.
- The app must handle corrupt or unreadable image files gracefully (log and skip, do not crash).
