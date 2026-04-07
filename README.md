# FaceGraph Manager

A local-first desktop application that uses facial recognition and graph theory to identify shared identities across fragmented photo folders. Draw connections between faces on an interactive canvas, then consolidate related folders with one click.

---

## How It Works

1. **Scan** a root directory. Each immediate subfolder becomes a group node on the canvas.
2. Faces are detected, embedded (ArcFace 512-d vectors), and clustered (HDBSCAN) per folder. Each cluster = one identity node.
3. **Draw connections** between identity nodes across groups to assert "same person."
4. Group edges are derived automatically — folders sharing a connected identity are linked.
5. **Inspect** any identity node to split mis-clustered faces or ignore noise.
6. **Consolidate** — related folders move under auto-named parent directories (`001/`, `002/`, …), preserving all internal structure.

---

## Tech Stack

| Component | Library |
|---|---|
| Face detection | InsightFace (RetinaFace, ONNX Runtime) |
| Face embedding | InsightFace (ArcFace `buffalo_l`, ONNX Runtime) |
| Clustering | HDBSCAN (scikit-learn 1.3+) |
| Vector search | FAISS (`faiss-cpu`) |
| Graph logic | NetworkX |
| UI | PySide6 + QGraphicsView |
| Database | SQLite (stdlib) |

No PyTorch. No TensorFlow. No cloud.

---

## Installation

```bash
pip install -r requirements.txt
```

**Windows (NVIDIA GPU):**
```bash
pip install onnxruntime-gpu faiss-cpu
```

**macOS:**
```bash
pip install onnxruntime faiss-cpu
```

On first run, InsightFace downloads the `buffalo_l` model pack (~300 MB). All subsequent runs are fully offline.

---

## Running

```bash
python main.py
```

---

## Usage

### Scan
`Ctrl+O` or click **Scan Folder**. Select a root directory. Each immediate subdirectory is treated as one group. Progress is shown in the scan dialog; clustering runs automatically when each folder finishes.

### Canvas Interactions

| Action | How |
|---|---|
| Connect two faces (same person) | Drag from one identity node to another in a different group |
| Remove a face connection | Double-click the blue edge between two identity nodes |
| Inspect a cluster | Click any identity node |
| Move a group node | Drag the group node header |
| Zoom | Scroll wheel |

### Cluster Inspector (right panel)
- **Select** individual face thumbnails with checkboxes.
- **Split Selected** — moves selected faces into a new identity node (use when two people are incorrectly merged into one cluster).
- **Ignore Selected** — removes noisy or background faces from the cluster.
- **Name** — optional label for the identity.
- **Key Representative** — marks this cluster as a highlighted identity for the group.

### Consolidate
`Ctrl+M` or click **Consolidate…**. Select an output directory, click **Preview** to see the proposed moves, then **Commit** to execute. An undo log is written before any files move.

### Undo
`Ctrl+Z` or click **Undo Last Consolidation** to reverse the last commit using the saved log.

---

## Configuration

Edit `config.py` before running:

```python
FORCE_DEVICE = None        # None=auto | "cuda" | "coreml" | "cpu"
DETECTION_THRESHOLD = 0.6  # Minimum face detection confidence
HDBSCAN_MIN_CLUSTER_SIZE = 2
DEFAULT_SIMILARITY_THRESHOLD = 0.6
```

---

## Data Storage

All data is stored in `~/.facegraph/`:

| File | Contents |
|---|---|
| `facegraph.db` | SQLite database (faces, identities, connections) |
| `facegraph.index` | FAISS vector index |
| `undo_log.json` | Last consolidation undo log |

---

## Project Structure

```
FaceGraph-Manager/
├── main.py
├── config.py
├── requirements.txt
├── core/
│   ├── database.py       # SQLite schema and all DB operations
│   ├── engine.py         # InsightFace wrapper (ONNX Runtime)
│   ├── scanner.py        # QThread background scanner
│   ├── clustering.py     # HDBSCAN + split/ignore operations
│   ├── faiss_index.py    # FAISS index management
│   ├── graph.py          # NetworkX group edge derivation
│   └── consolidation.py  # Preview, commit, undo file operations
└── ui/
    ├── nodes.py           # QGraphicsItems (group, identity, edge nodes)
    ├── canvas.py          # QGraphicsView workbench
    ├── cluster_panel.py   # Right-side cluster inspector
    ├── consolidation_dialog.py
    ├── scan_dialog.py
    └── main_window.py
```
