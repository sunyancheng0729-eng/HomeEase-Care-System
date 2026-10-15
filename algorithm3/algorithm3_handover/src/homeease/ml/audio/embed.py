"""Pretrained sound embedding: Google YAMNet (MobileNetV1 on AudioSet), ONNX, 16 MB, Apache-2.0.

The ONNX file is fetched with ``homeease models fetch --audio`` (pinned repo revision, sha256 in models/audio/MODELS.json). The
mel front end is inside the graph: the input is the raw 16 kHz mono waveform. A 2 s window gives 4 frames of 0.96 s (hop 0.48 s);
the window embedding is the mean and the max over frames of the 1024-d frame embeddings (2048 numbers). AudioSet ontology classes
(including Screaming) are not used directly: a small classifier is trained on the embedding.

The model was trained on web video audio (AudioSet), not on household or older-adult audio, and it is third-party: the checksum
of the file is verified against the value recorded at fetch time when a manifest entry exists.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from homeease.core.paths import models_dir
from homeease.core.util import get_logger, sha256_file

log = get_logger(__name__)

YAMNET_REL = "audio/yamnet/yamnet.onnx"
EMBED_DIM = 2 * 1024


class YamnetEmbedder:
    """Thread-safe (onnxruntime sessions are) window embedder; one instance per process is enough."""

    def __init__(self, path: Path | None = None, verify: bool = True) -> None:
        import onnxruntime as ort

        p = Path(path) if path else models_dir() / YAMNET_REL
        if not p.is_file():
            raise FileNotFoundError(f"YAMNet model missing: {p} (run `homeease models fetch --audio`)")
        self.path, self.sha256 = p, sha256_file(p) if verify else None
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1  # the monitor scores one small window at a time; do not fight the other threads
        self.sess = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def frames(self, x: np.ndarray) -> np.ndarray:
        """(frames, 1024) embeddings of a mono 16 kHz float waveform in [-1, 1]."""
        return self.sess.run(None, {self.inp: np.ascontiguousarray(x, dtype=np.float32)})[1]

    def window_embedding(self, x: np.ndarray) -> np.ndarray:
        e = self.frames(x)
        return np.concatenate([e.mean(axis=0), e.max(axis=0)]).astype(np.float32)


_shared: YamnetEmbedder | None = None
_lock = threading.Lock()


def shared_embedder() -> YamnetEmbedder:
    global _shared
    with _lock:
        if _shared is None:
            _shared = YamnetEmbedder()
        return _shared
