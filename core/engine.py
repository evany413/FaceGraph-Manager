import numpy as np
import onnxruntime as ort
import config


def _resolve_providers() -> tuple[list[str], int]:
    device = config.FORCE_DEVICE
    if device is None:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            device = "cuda"
        elif "CoreMLExecutionProvider" in available:
            device = "coreml"
        else:
            device = "cpu"

    provider_map = {
        "cuda":   (["CUDAExecutionProvider", "CPUExecutionProvider"], 0),
        "coreml": (["CoreMLExecutionProvider", "CPUExecutionProvider"], -1),
        "cpu":    (["CPUExecutionProvider"], -1),
    }
    return provider_map[device]


class FaceEngine:
    """Thin wrapper around InsightFace. One instance per process."""

    def __init__(self):
        from insightface.app import FaceAnalysis

        providers, ctx_id = _resolve_providers()
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._threshold = config.DETECTION_THRESHOLD

    def analyze(self, image_path: str) -> list[dict]:
        """
        Returns a list of detected faces, each as:
            {
                "embedding": np.ndarray (512,) float32, L2-normalized,
                "bbox":      [x1, y1, x2, y2] ints,
                "confidence": float,
            }
        Returns [] on unreadable images.
        """
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return []

        try:
            faces = self._app.get(img)
        except Exception:
            return []

        results = []
        for face in faces:
            if face.det_score < self._threshold:
                continue
            emb = face.embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            bbox = [int(v) for v in face.bbox.tolist()]
            results.append(
                {
                    "embedding": emb,
                    "bbox": bbox,
                    "confidence": float(face.det_score),
                }
            )
        return results
