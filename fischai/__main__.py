"""Command line: fischai setup | run | detect | record | doctor."""

import argparse
import os
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass

from . import __version__, config, hypr, screen
from .config import DATA_DIR, STATE_DIR

MODEL_DIR = DATA_DIR / "model"
TEMPLATE_DIR = DATA_DIR / "templates"
IDLE_DETECT_FPS = 5
STATUS_REFRESH_S = 0.5


class Log:
    """Timestamped lines to the terminal and fischai.log, plus one live status
    line at the bottom of an interactive terminal."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", buffering=1)
        self.tty = sys.stdout.isatty()
        self.lock = threading.Lock()
        self.status_text = ""

    def __call__(self, message):
        line = f"{time.strftime('%H:%M:%S')} {message}"
        with self.lock:
            if self.tty:
                sys.stdout.write("\r\033[K")
            print(line, flush=True)
            self.file.write(line + "\n")
            if self.tty and self.status_text:
                sys.stdout.write(self.status_text)
                sys.stdout.flush()

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


def _load_vision():
    from .vision import AquariumPanelDetector, CycleDetector, ShakeDetector, WeatherEventDetector
    return Vision(ShakeDetector(TEMPLATE_DIR), CycleDetector(TEMPLATE_DIR),
                  WeatherEventDetector(TEMPLATE_DIR), AquariumPanelDetector(TEMPLATE_DIR))


def _load_config():
    try:
        return config.load()
    except config.ConfigError as e:
        raise SystemExit(f"Config error: {e}") from None


def _require_assets():
    if not (MODEL_DIR / "model.json").is_file() or not TEMPLATE_DIR.is_dir():
        raise SystemExit("The ExoPanda model and templates are not installed yet. Run:\n"
                         "  fischai setup /path/to/fisch-<version>.zip")


class Session:
    """One start/stop cycle of the macro: window lookup, focus watcher,
    detector thread and the fishing loop."""

    def __init__(self, cfg, model, vision, io, log, on_exit):
        from .detector import AiBridge, DetectionRegion
        from .macro import Macro, MacroRunner, Stats

        self.log = log
        win = hypr.find_window(cfg.general.window)
        self.focus = hypr.FocusWatcher(win.address)
        self.focus.start()
        self.stop_event = threading.Event()
        self.stats = Stats()
        self.phase, self.detail = "starting", ""
        self.region = DetectionRegion(cfg.region_fractions)
        self.macro = None
        self.bridge = AiBridge(model, lambda: self.macro.rect if self.macro else win.rect,
                               self.region, cfg.reel.fps, IDLE_DETECT_FPS, log)
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


def cmd_setup(args):
    from . import assets
    assets.install(args.source, DATA_DIR)
    print(f"Done. Settings live in {config.CONFIG_PATH} (created on the first `fischai run`).")


def cmd_run(args):
    hypr.require_hyprland()
    cfg = _load_config()
    _require_assets()
    from .detector import ReelModel
    from .inputdev import HotkeyListener, VirtualInput
    from .webhook import Webhook, summary

    log = Log(STATE_DIR / "fischai.log")
    model = ReelModel(MODEL_DIR, cfg.reel.backend)
    log(f"FischAI {__version__}: reel detector on {model.backend.name}")
    vision = _load_vision()
    io = VirtualInput()
    events = queue.Queue()
    hotkey = cfg.general.hotkey
    HotkeyListener([hotkey], lambda _key: events.put("toggle")).start()
    log(f"Ready. In Sober, stand facing water with the rod in hotbar slot 1 and press {hotkey.upper()} "
        f"to start or stop. Ctrl+C quits.")

    session = None
    webhook = None
    next_summary = 0.0

    def finish(reason):
        nonlocal session
        error = session.runner.error
        session.stop()
        log.clear_status()
        text = summary(session.stats)
        log(f"stopped ({reason}): {text}" + (f" - {error}" if error else ""))
        webhook.send(f"Stopped ({error or reason}): {text}")
        session = None

    try:
        while True:
            try:
                event = events.get(timeout=STATUS_REFRESH_S)
            except queue.Empty:
                event = None
            if event == "toggle" and session is None:
                try:
                    cfg = _load_config()
                    webhook = Webhook(cfg.webhook.url, log)
                    session = Session(cfg, model, vision, io, log, lambda runner: events.put(runner))
                except (hypr.WindowError, SystemExit) as e:
                    log(str(e))
                    session = None
                    continue
                session.start()
                next_summary = time.time() + cfg.webhook.summary_min * 60
                log("started")
                webhook.send("Started fishing.")
            elif event == "toggle":
                finish("hotkey")
            elif session is not None and event is session.runner:
                finish("the macro ended")
            if session is not None:
                log.show_status(session.status_line())
                if time.time() >= next_summary:
                    next_summary = time.time() + cfg.webhook.summary_min * 60
                    webhook.send(summary(session.stats))
    except KeyboardInterrupt:
        pass
    finally:
        if session is not None:
            finish("quit")
        log.clear_status()
        io.close()


def _annotate(rgb, region, bar, fish):
    import cv2
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    x, y, w, h = region
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
    for box, color, name in ((bar, (0, 220, 0), "bar"), (fish, (0, 0, 255), "fish")):
        if box is not None:
            p1, p2 = (round(box.x), round(box.y)), (round(box.x + box.w), round(box.y + box.h))
            cv2.rectangle(out, p1, p2, color, 2)
            cv2.putText(out, f"{name} {box.conf:.2f}", (p1[0], p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def _describe(box):
    if box is None:
        return "-"
    return f"{box.conf:.2f} x={box.cx:.0f} w={box.w:.0f}"


def cmd_detect(args):
    import cv2

    from .detector import Box, DetectionRegion, ReelModel
    cfg = _load_config()
    _require_assets()
    model = ReelModel(MODEL_DIR, args.backend or cfg.reel.backend)
    region = DetectionRegion(cfg.region_fractions)
    print(f"backend: {model.backend.name}")

    def shift(box, dx, dy):
        return None if box is None else Box(box.x + dx, box.y + dy, box.w, box.h, box.conf)

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"cannot read {args.image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rx, ry, rw, rh = region.rect((0, 0, rgb.shape[1], rgb.shape[0]))
        t0 = time.perf_counter()
        bar, fish = model.predict(rgb[ry:ry + rh, rx:rx + rw])
        dt = (time.perf_counter() - t0) * 1000
        bar, fish = shift(bar, rx, ry), shift(fish, rx, ry)
        print(f"region {rx},{ry} {rw}x{rh}  bar {_describe(bar)}  fish {_describe(fish)}  ({dt:.0f} ms)")
        if args.save:
            cv2.imwrite(args.save, _annotate(rgb, (rx, ry, rw, rh), bar, fish))
            print(f"saved {args.save}")
        return

    hypr.require_hyprland()
    win = hypr.find_window(cfg.general.window)
    rx, ry, rw, rh = region.rect(win.rect)
    print(f"window {win.width}x{win.height} at {win.x},{win.y}; detection strip {rw}x{rh} at {rx},{ry}")
    times, last = [], None
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        t0 = time.perf_counter()
        frame = screen.grab(rx, ry, rw, rh)
        bar, fish = model.predict(frame)
        times.append((time.perf_counter() - t0) * 1000)
        bar, fish = shift(bar, rx, ry), shift(fish, rx, ry)
        last = (bar, fish)
        print(f"bar {_describe(bar):24s} fish {_describe(fish):24s} {times[-1]:.0f} ms")
    if times:
        print(f"{len(times)} frames, median {sorted(times)[len(times) // 2]:.0f} ms per capture + inference")
    if args.save and last is not None:
        full = screen.grab(*win.rect)
        local = [shift(b, -win.x, -win.y) for b in last]
        cv2.imwrite(args.save, _annotate(full, (rx - win.x, ry - win.y, rw, rh), *local))
        print(f"saved {args.save}")


def cmd_record(args):
    """Save game-window frames while the user plays, as material for detectors
    of screens ExoPanda only reads from memory (Pinion's Aria notes, appraisal
    results, the cast power bar)."""
    import cv2
    hypr.require_hyprland()
    cfg = _load_config()
    win = hypr.find_window(cfg.general.window)
    out = STATE_DIR / "recordings" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    print(f"recording {win.width}x{win.height} every {1 / args.fps:.2f} s for {args.seconds:.0f} s into {out}")
    count = 0
    end = time.monotonic() + args.seconds
    try:
        while time.monotonic() < end:
            t0 = time.monotonic()
            win = hypr.refresh(win)
            frame = screen.grab(*win.rect)
            cv2.imwrite(str(out / f"{count:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            count += 1
            time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    print(f"saved {count} frames")


def cmd_doctor(args):
    ok = True

    def check(passed, label, hint=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"{'ok ' if passed else 'FAIL'} {label}" + ("" if passed or not hint else f"\n     {hint}"))

    check(os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"), "Hyprland session")
    check(shutil.which("grim"), "grim installed", "install grim")
    check(os.access("/dev/uinput", os.W_OK), "/dev/uinput writable", "add a udev rule giving your user access to /dev/uinput")
    from .inputdev import _keyboard_event_paths
    readable = [p for p in _keyboard_event_paths() if os.access(p, os.R_OK)]
    check(readable, "keyboard readable for the hotkey", "sudo usermod -aG input $USER, then log in again")
    try:
        cfg = config.load()
        check(True, f"config {config.CONFIG_PATH}")
    except config.ConfigError as e:
        check(False, "config", str(e))
        cfg = config.Config()
    assets_ok = (MODEL_DIR / "model.json").is_file() and TEMPLATE_DIR.is_dir()
    check(assets_ok, f"ExoPanda model and templates in {DATA_DIR}", "fischai setup /path/to/fisch-<version>.zip")
    if assets_ok:
        import numpy as np

        from .detector import INPUT_SIZE, ReelModel
        model = ReelModel(MODEL_DIR, cfg.reel.backend)
        tensor = np.zeros((3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        model.backend.run(tensor)
        t0 = time.perf_counter()
        for _ in range(10):
            model.backend.run(tensor)
        ms = (time.perf_counter() - t0) * 100
        check(True, f"reel detector: {model.backend.name}, {ms:.0f} ms per frame")
        _load_vision()
        check(True, "vision templates load")
    if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        try:
            win = hypr.find_window(cfg.general.window)
            check(True, f"game window: {win.title!r} {win.width}x{win.height}")
            if (win.width, win.height) != (1920, 1080):
                print("     note: ExoPanda's tuning targets a 1920x1080 game window")
        except hypr.WindowError as e:
            check(False, "game window", str(e))
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser(prog="fischai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("setup", help="install ExoPanda's reel model and templates from its Fisch module zip")
    p.add_argument("source", help="fisch-<version>.zip (Windows module) or the ExoPanda_Fischer.exe inside it")
    p.set_defaults(func=cmd_setup)
    p = sub.add_parser("run", help="wait for the hotkey and fish")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("detect", help="test the reel detector on the live game or an image")
    p.add_argument("--image", help="a 1920x1080-style game screenshot instead of the live game")
    p.add_argument("--seconds", type=float, default=5.0, help="how long to watch the live game")
    p.add_argument("--backend", choices=("auto", "ncnn", "onnxruntime"))
    p.add_argument("--save", help="write an annotated PNG here")
    p.set_defaults(func=cmd_detect)
    p = sub.add_parser("record", help="save game-window frames while you play (material for new detectors)")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--fps", type=float, default=4.0)
    p.set_defaults(func=cmd_record)
    p = sub.add_parser("doctor", help="check the environment")
    p.set_defaults(func=cmd_doctor)
    args = parser.parse_args()
    try:
        args.func(args)
    except hypr.WindowError as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
