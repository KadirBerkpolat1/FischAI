"""
Runtime shared by `fischai run` and the settings window: the log, the loaded
model and templates, one macro Session at a time, and the Controller that
starts and stops sessions and sends the Discord notices.
"""

import fcntl
import queue
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

from . import hypr
from .config import DATA_DIR, STATE_DIR

MODEL_DIR = DATA_DIR / "model"
TEMPLATE_DIR = DATA_DIR / "templates"
IDLE_DETECT_FPS = 5


class Log:
    """Timestamped lines to fischai.log, the terminal (with one live status
    line at the bottom when interactive) and any registered listeners."""

    def __init__(self, path, echo=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", buffering=1)
        self.echo = echo
        self.tty = echo and sys.stdout.isatty()
        self.lock = threading.Lock()
        self.status_text = ""
        self.listeners = []

    def __call__(self, message):
        line = f"{time.strftime('%H:%M:%S')} {message}"
        with self.lock:
            if self.tty:
                sys.stdout.write("\r\033[K")
            if self.echo:
                print(line, flush=True)
            self.file.write(line + "\n")
            if self.tty and self.status_text:
                sys.stdout.write(self.status_text)
                sys.stdout.flush()
        for listener in self.listeners:
            listener(line)

    def show_status(self, text):
        if not self.tty:
            return
        with self.lock:
            self.status_text = text[:max(10, shutil.get_terminal_size().columns - 1)]
            sys.stdout.write("\r\033[K" + self.status_text)
            sys.stdout.flush()

    def clear_status(self):
        self.show_status("")


@dataclass
class Vision:
    shake: object
    cycle: object
    weather_event: object
    aquarium: object


def load_vision():
    from .vision import AquariumPanelDetector, CycleDetector, ShakeDetector, WeatherEventDetector
    return Vision(ShakeDetector(TEMPLATE_DIR), CycleDetector(TEMPLATE_DIR),
                  WeatherEventDetector(TEMPLATE_DIR), AquariumPanelDetector(TEMPLATE_DIR))


def assets_installed():
    return (MODEL_DIR / "model.json").is_file() and TEMPLATE_DIR.is_dir()


ASSETS_MISSING = ("The ExoPanda model and templates are not installed yet. Run:\n"
                  "  fischai setup /path/to/fisch-<version>.zip")


@contextmanager
def single_instance():
    """Two instances would both answer the hotkey and fight over the mouse."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "instance.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("FischAI is already running (`fischai run` or the settings window). "
                             "Close that one first.") from None
        yield


class Session:
    """One start/stop cycle of the macro: window lookup, focus watcher,
    detector thread and the fishing loop."""

    def __init__(self, cfg, model, vision, io, log, on_exit):
        from .detector import AiBridge, DetectionRegion
        from .macro import Macro, MacroRunner, Stats

        win = hypr.find_window(cfg.general.window)
        self.focus = hypr.FocusWatcher(win.address)
        self.focus.start()
        self.stop_event = threading.Event()
        self.stats = Stats()
        self.phase, self.detail = "starting", ""
        self.region = DetectionRegion(cfg.region_fractions)
        self.macro = None
        self.bridge = AiBridge(model, lambda: self.macro.rect if self.macro else win.rect,
                               self.region, cfg.reel, IDLE_DETECT_FPS, log)
        self.macro = Macro(cfg, win, io, self.focus, self.bridge, vision, STATE_DIR, log,
                           self._set_status, self.stop_event, self.stats)
        self.runner = MacroRunner(self.macro, on_exit)

    def _set_status(self, phase, detail):
        self.phase, self.detail = phase, detail

    def start(self):
        self.bridge.start()
        self.runner.start()

    def stop(self):
        self.stop_event.set()
        if self.runner.is_alive():
            self.runner.join(10)
        self.bridge.stop()
        if self.bridge.is_alive():
            self.bridge.join(2)
        self.focus.stop()

    def status_line(self):
        s = self.stats
        return (f"[{self.phase}] {self.detail} | reels {s.reels} casts {s.casts} "
                f"| detector {self.bridge.frame_ms:.0f} ms/frame")


class Controller:
    """Owns the long-lived pieces (model, templates, virtual input) and at
    most one running Session. Not thread-safe: call it from one thread."""

    def __init__(self, cfg, log):
        from .detector import ReelModel
        from .inputdev import VirtualInput

        self.cfg = cfg
        self.log = log
        self.model = ReelModel(MODEL_DIR, cfg.reel.backend)
        self.vision = load_vision()
        self.io = VirtualInput()
        self.session = None
        self.last_error = None
        self._webhook = None
        self._next_summary = 0.0
        self._exited = queue.Queue()

    def start(self):
        """Start fishing; False (with last_error set) if the game window is missing."""
        from .webhook import Webhook
        if self.session is not None:
            return True
        try:
            self.session = Session(self.cfg, self.model, self.vision, self.io, self.log, self._exited.put)
        except hypr.WindowError as e:
            self.last_error = str(e)
            self.log(self.last_error)
            return False
        self.last_error = None
        self._webhook = Webhook(self.cfg.webhook, self.log)
        self.session.start()
        self._next_summary = time.time() + self.cfg.webhook.summary_min * 60
        self.log("started")
        self._webhook.send("Started fishing.")
        return True

    def stop(self, reason):
        from .webhook import summary
        session, self.session = self.session, None
        if session is None:
            return
        error = session.runner.error
        session.stop()
        text = summary(session.stats)
        self.last_error = error
        self.log(f"stopped ({reason}): {text}" + (f" - {error}" if error else ""))
        self._webhook.send(f"Stopped ({error or reason}): {text}")

    def toggle(self):
        if self.session is None:
            self.start()
        else:
            self.stop("hotkey")

    def poll(self):
        """Ends a session whose macro stopped on its own and sends the periodic summary."""
        from .webhook import summary
        while not self._exited.empty():
            runner = self._exited.get_nowait()
            if self.session is not None and runner is self.session.runner:
                self.stop("the macro ended")
        if self.session is not None and time.time() >= self._next_summary:
            self._next_summary = time.time() + self.cfg.webhook.summary_min * 60
            self._webhook.send(summary(self.session.stats))

    def close(self):
        self.stop("quit")
        self.io.close()
