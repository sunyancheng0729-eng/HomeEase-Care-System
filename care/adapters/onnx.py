"""Execution-only adapter for an original OpenCV DNN ONNX network."""
from pathlib import Path

import onnxruntime as ort


class OnnxNetwork:
    def __init__(self, path, threads=2, device="cpu"):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            import torch
            if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA ONNX Runtime is unavailable")
            ort.preload_dlls(directory=str(Path(torch.__file__).parent / "lib"))
            providers.insert(0, ("CUDAExecutionProvider", {"use_tf32": "0", "cudnn_conv_algo_search": "HEURISTIC"}))
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if device == "cuda" and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("CUDA ONNX Runtime could not initialize")
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.blob = None

    def setInput(self, blob):
        self.blob = blob

    def getUnconnectedOutLayersNames(self):
        return self.output_names

    def forward(self, names):
        try:
            return self.session.run(list(names), {self.input_name: self.blob})
        finally:
            self.blob = None
