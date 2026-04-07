from pathlib import Path

APP_DIR = Path.home() / ".facegraph"
APP_DIR.mkdir(exist_ok=True)

DB_PATH = APP_DIR / "facegraph.db"
FAISS_INDEX_PATH = APP_DIR / "facegraph.index"
UNDO_LOG_PATH = APP_DIR / "undo_log.json"

# Device: None = auto-detect, "cuda" | "coreml" | "cpu" = override
FORCE_DEVICE = None

DETECTION_THRESHOLD = 0.6
HDBSCAN_MIN_CLUSTER_SIZE = 2
DEFAULT_SIMILARITY_THRESHOLD = 0.6
THRESHOLD_DELTA = 0.02

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

THUMBNAIL_SIZE = 70
GROUP_NODE_WIDTH = 240
GROUP_NODE_PADDING = 12
IDENTITY_NODE_SIZE = 70
IDENTITY_NODES_PER_ROW = 3
