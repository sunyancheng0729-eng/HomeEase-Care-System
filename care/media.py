import threading
from fractions import Fraction

from care.config import RUNTIME

_lock = threading.Lock()


def browser_clip(original):
    """Encode a playback derivative; original evidence is left byte-for-byte intact."""
    import av
    import cv2
    output = RUNTIME / "playback" / original.name
    output.parent.mkdir(exist_ok=True)
    with _lock:
        if output.exists() and output.stat().st_mtime >= original.stat().st_mtime:
            return output
        temporary = output.with_suffix(".partial.mp4")
        cap = cv2.VideoCapture(str(original))
        container = None
        try:
            if not cap.isOpened():
                raise ValueError("原证据视频不可读取")
            fps = Fraction(float(cap.get(cv2.CAP_PROP_FPS) or 15)).limit_denominator(1000)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            container = av.open(str(temporary), "w", options={"movflags": "+faststart"})
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height = w // 2 * 2, h // 2 * 2
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "20", "preset": "veryfast"}
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if image.shape[1] != stream.width or image.shape[0] != stream.height:
                    image = cv2.resize(image, (stream.width, stream.height))
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
            temporary.replace(output)
            return output
        finally:
            cap.release()
            if container:
                container.close()
            temporary.unlink(missing_ok=True)
