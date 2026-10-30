"""
Reel minigame detection with ExoPanda's YOLO11s model (classes `fish-line`
and `catch-bar`; 1x3x640x640 RGB in, 1x6x8400 out, no NMS baked in), and the
thread that keeps the newest boxes available to the reel controller, the
counterpart of ExoPanda's GetAILatestBoxes bridge.

Backends: ncnn on Vulkan (~15 ms per frame on an RX 7600) when the converted
model exists, otherwise onnxruntime on the CPU (~100 ms).
"""

import json
import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import screen

INPUT_SIZE = 640
LETTERBOX_FILL = 114
PUBLISH_MIN_CONF = 0.25          # ExoPanda inference.rs CONF_THRESH
FISH_CLASS = "fish-line"
BAR_CLASS = "catch-bar"

# Default detection region as fractions of the game window. The reel track
# sits at x 0.29-0.71, y 0.83-0.90 of a 1920x1080 client; this wider strip
# around it scored best for both classes on recorded gameplay.
DEFAULT_REGION = (0.21, 0.76, 0.79, 0.96)
RECENTER_SAMPLES = 15            # confident bar sightings before the strip re-centres on them
RECENTER_MIN_CONF = 0.6

# Box tracker: one-euro smoothing of each class's centre, plus a jump gate
# that holds back a single-frame teleport until the next frame confirms it.
EURO_MIN_CUTOFF_HZ = 3.0
EURO_BETA = 0.05
EURO_D_CUTOFF_HZ = 1.0
MAX_SPEED_PX_S = 6000.0
TRACK_RESET_S = 0.5


@dataclass(frozen=True)
class Box:
    """Top-left x/y, width, height and confidence."""
    x: float
    y: float
    w: float
    h: float
    conf: float

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


@dataclass(frozen=True)
class Boxes:
    """Newest detections in layout coordinates. `bar`/`fish` come from the
    newest processed frame (None = not seen in it); `bar_t`/`fish_t` are the
    monotonic capture times of the last frame that saw each class and
    `frame_t` that of the newest frame."""
    bar: Box | None = None
    fish: Box | None = None
    bar_t: float = -math.inf
    fish_t: float = -math.inf
    frame_t: float = -math.inf


def letterbox(rgb):
    """Ultralytics-style letterbox: keep aspect, pad to a centred 640x640 with
    grey 114. Returns the CHW float32 tensor, the scale and the padding."""
    h, w = rgb.shape[:2]
    scale = INPUT_SIZE / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), LETTERBOX_FILL, dtype=np.uint8)
    pad_x, pad_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)
    tensor *= 1 / 255
    return tensor, scale, pad_x, pad_y


class _NcnnBackend:
    name = "ncnn (Vulkan GPU)"

    def __init__(self, param_path, bin_path):
        import ncnn
        if ncnn.get_gpu_count() == 0:
            raise RuntimeError("ncnn sees no Vulkan GPU")
        self._ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True
        if self.net.load_param(str(param_path)) != 0 or self.net.load_model(str(bin_path)) != 0:
            raise RuntimeError(f"ncnn could not load {param_path}")

    def run(self, chw):
        # ncnn.Mat wraps the numpy buffer without copying: `chw` must stay
        # alive until extract() returns.
        mat = self._ncnn.Mat(chw)
        ex = self.net.create_extractor()
        ex.input("in0", mat)
        ret, out = ex.extract("out0")
        if ret != 0:
            raise RuntimeError(f"ncnn extract failed ({ret})")
        return np.array(out)


class _OnnxBackend:
    name = "onnxruntime (CPU)"

    def __init__(self, onnx_path):
        import os

        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def run(self, chw):
        return self.session.run(None, {self.input: chw[None]})[0][0]


class ReelModel:
    def __init__(self, model_dir, backend="auto"):
        meta = json.loads((model_dir / "model.json").read_text())
        index = {name: int(i) for i, name in meta["names"].items()}
        self.fish_row = 4 + index[FISH_CLASS]
        self.bar_row = 4 + index[BAR_CLASS]
        self.backend = self._open(model_dir, backend)

    @staticmethod
    def _open(model_dir, backend):
        param, weights, onnx = model_dir / "best.ncnn.param", model_dir / "best.ncnn.bin", model_dir / "best.onnx"
        if backend in ("auto", "ncnn") and param.is_file():
            try:
                return _NcnnBackend(param, weights)
            except Exception:
                if backend == "ncnn":
                    raise
        return _OnnxBackend(onnx)

    def predict(self, rgb):
        """Best (bar, fish) boxes in `rgb` pixel coordinates; a class below the
        publish floor comes back as None."""
        tensor, scale, pad_x, pad_y = letterbox(rgb)
        out = self.backend.run(tensor)
        return self._best(out, self.bar_row, scale, pad_x, pad_y), self._best(out, self.fish_row, scale, pad_x, pad_y)

    @staticmethod
    def _best(out, row, scale, pad_x, pad_y):
        i = int(out[row].argmax())
        conf = float(out[row, i])
        if conf < PUBLISH_MIN_CONF:
            return None
        cx, cy, w, h = (float(v) for v in out[:4, i])
        return Box((cx - w / 2 - pad_x) / scale, (cy - h / 2 - pad_y) / scale, w / scale, h / scale, conf)


class _CenterTrack:
    """One-euro filter on a box centre plus a teleport gate: a jump faster than
    MAX_SPEED_PX_S is held back until a second frame lands near it."""

    def __init__(self):
        self.t = None
        self.x = self.y = self.dx = 0.0
        self.pending = None

    @staticmethod
    def _alpha(dt, cutoff):
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))

    def update(self, box, t):
        if box is None:
            return None
        cx, cy = box.cx, box.cy
        if self.t is None or t - self.t > TRACK_RESET_S:
            self.t, self.x, self.y, self.dx, self.pending = t, cx, cy, 0.0, None
            return box
        dt = max(t - self.t, 1e-3)
        if abs(cx - self.x) / dt > MAX_SPEED_PX_S:
            confirmed = self.pending is not None and abs(cx - self.pending) / dt <= MAX_SPEED_PX_S
            if not confirmed:
                self.pending = cx
                return None
            self.t, self.x, self.y, self.dx, self.pending = t, cx, cy, 0.0, None
            return box
        self.pending = None
        a_d = self._alpha(dt, EURO_D_CUTOFF_HZ)
        self.dx += a_d * ((cx - self.x) / dt - self.dx)
        a = self._alpha(dt, EURO_MIN_CUTOFF_HZ + EURO_BETA * abs(self.dx))
        self.x += a * (cx - self.x)
        self.y += a * (cy - self.y)
        self.t = t
        return Box(self.x - box.w / 2, self.y - box.h / 2, box.w, box.h, box.conf)


class DetectionRegion:
    """The strip of the window the model looks at, as window fractions. Unless
    the user fixed it in the config, the strip re-centres vertically on the
    bar once enough confident sightings say where the track really is."""

    def __init__(self, fractions=None):
        self.fixed = fractions is not None
        self.fractions = tuple(fractions or DEFAULT_REGION)
        self._bar_rows = []

    def rect(self, win_rect):
        wx, wy, ww, wh = win_rect
        x1, y1, x2, y2 = self.fractions
        return round(wx + x1 * ww), round(wy + y1 * wh), max(1, round((x2 - x1) * ww)), max(1, round((y2 - y1) * wh))

    def observe_bar(self, bar, win_rect):
        if self.fixed or len(self._bar_rows) >= RECENTER_SAMPLES or bar is None or bar.conf < RECENTER_MIN_CONF:
            return False
        self._bar_rows.append((bar.cy - win_rect[1]) / win_rect[3])
        if len(self._bar_rows) < RECENTER_SAMPLES:
            return False
        x1, y1, x2, y2 = self.fractions
        half = (y2 - y1) / 2
        centre = float(np.median(self._bar_rows))
        centre = min(max(centre, half), 1 - half)
        self.fractions = (x1, centre - half, x2, centre + half)
        return True


class AiBridge(threading.Thread):
    """Captures the detection strip and runs the model in a loop, publishing
    the newest Boxes. Runs at `fps` while the macro is in the shake or reel
    phase and at `idle_fps` otherwise."""

    def __init__(self, model, window_rect, region, fps, idle_fps, log):
        super().__init__(daemon=True, name="reel-detector")
        self.model = model
        self.window_rect = window_rect      # callable -> (x, y, w, h) or None
        self.region = region
        self.fps = fps
        self.idle_fps = idle_fps
        self.log = log
        self.active = False
        self.latest = Boxes()
        self.frame_ms = 0.0
        self.error = None
        self._stop_event = threading.Event()
        self._bar_track = _CenterTrack()
        self._fish_track = _CenterTrack()

    def stop(self):
        self._stop_event.set()

    def run(self):
        bar_t = fish_t = -math.inf
        while not self._stop_event.is_set():
            period = 1.0 / (self.fps if self.active else self.idle_fps)
            win = self.window_rect()
            if win is None:
                self._stop_event.wait(period)
                continue
            rx, ry, rw, rh = self.region.rect(win)
            t0 = time.monotonic()
            try:
                bar, fish = self.model.predict(screen.grab(rx, ry, rw, rh))
            except Exception as e:  # keep the thread alive; the macro reports it
                if str(e) != self.error:
                    self.error = str(e)
                    self.log(f"detector error: {e}")
                self._stop_event.wait(0.5)
                continue
            self.error = None
            bar = self._bar_track.update(_shift(bar, rx, ry), t0)
            fish = self._fish_track.update(_shift(fish, rx, ry), t0)
            if self.region.observe_bar(bar, win):
                self.log("detector: re-centred the detection strip on the reel bar")
            if bar is not None:
                bar_t = t0
            if fish is not None:
                fish_t = t0
            self.latest = Boxes(bar, fish, bar_t, fish_t, t0)
            elapsed = time.monotonic() - t0
            self.frame_ms += 0.1 * (elapsed * 1000 - self.frame_ms)
            self._stop_event.wait(max(0.0, period - elapsed))


def _shift(box, dx, dy):
    if box is None:
        return None
    return Box(box.x + dx, box.y + dy, box.w, box.h, box.conf)
