"""
Getting back to fishing after Sober closes or Roblox drops the connection:
restart Sober straight into Fisch, wait until the HUD shows, then walk the
recorded route from the spawn point (the last Innkeeper bed, or Moosewood)
to the fishing spot.

The route is the user's own key presses (W A S D, space, arrow keys) with
their timing, recorded once from the spawn point; replaying them from the
same spawn walks the same path.
"""

import json
import os
import select
import subprocess
import threading
import time

import cv2

from . import hypr, inputdev, screen

FISCH_PLACE_ID = 16732694052
JOIN_URI = f"roblox://experience?placeId={FISCH_PLACE_ID}"
SOBER_APP = "org.vinegarhq.Sober"

KILL_WAIT_S = 15.0
LOAD_WAIT_S = 150.0             # window, Roblox loading screen and Fisch's own loading
LOAD_POLL_S = 2.0
SPAWN_SETTLE_S = 3.0            # the character lands and the camera settles after the HUD appears
ATTEMPTS = 3
PROMPT_CROP = (0.20, 0.25, 0.80, 0.75)   # Roblox centres its error prompt over the dimmed game
PROMPT_WORDS = ("error code", "disconnected")
ROUTE_KEYS = ("w", "a", "s", "d", "space", "left", "right", "up", "down")
ROUTE_FILE = "route.json"       # in the state directory


class GameLost(Exception):
    """Sober closed or Roblox showed its error prompt."""


def error_prompt(frame):
    """True when Roblox's error prompt ("Disconnected … (Error Code: 277)") is on screen."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = PROMPT_CROP
    crop = cv2.cvtColor(frame[int(h * y1):int(h * y2), int(w * x1):int(w * x2)], cv2.COLOR_RGB2GRAY)
    ok, png = cv2.imencode(".png", crop)
    if not ok:
        return False
    out = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "11"], input=png.tobytes(),
                         capture_output=True, check=False)
    text = out.stdout.decode(errors="replace").lower()
    return any(word in text for word in PROMPT_WORDS)


def restart_sober(window_match, sleep):
    """Kill Sober if it still runs and start it again joining Fisch."""
    subprocess.run(["flatpak", "kill", SOBER_APP], capture_output=True, check=False)
    deadline = time.monotonic() + KILL_WAIT_S
    while time.monotonic() < deadline:
        try:
            hypr.find_window(window_match)
        except hypr.WindowError:
            break
        sleep(0.5)
    subprocess.Popen(["flatpak", "run", SOBER_APP, JOIN_URI], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def wait_for_fisch(window_match, cycle_detector, sleep):
    """The game window once Fisch's day/night HUD icon shows, or None after LOAD_WAIT_S."""
    deadline = time.monotonic() + LOAD_WAIT_S
    while time.monotonic() < deadline:
        sleep(LOAD_POLL_S)
        try:
            win = hypr.find_window(window_match)
            hypr.focus(win)     # grim only sees what is on screen
        except hypr.WindowError:
            continue
        frame = screen.grab(*win.rect)
        if cycle_detector.detect(frame) is not None:
            return win
        if error_prompt(frame):
            return None
    return None


def load_route(path):
    """[(seconds, key, down), ...] or None when no route was recorded."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    events = [(float(t), str(k), bool(d)) for t, k, d in data.get("events", []) if k in ROUTE_KEYS]
    return events or None


def walk(io, events, sleep):
    """Replay a recorded route through the virtual keyboard."""
    start = time.monotonic()
    try:
        for t, key, down in events:
            wait = start + t - time.monotonic()
            if wait > 0:
                sleep(wait)
            io.key(key, down)
    finally:
        for key in ROUTE_KEYS:
            io.release_key(key)


class RouteRecorder:
    """Records the route keys from the real keyboards between two presses of
    the hotkey. Key repeats are dropped; only presses and releases count."""

    def __init__(self, hotkey):
        self.hotkey_code = inputdev.keycode(hotkey)
        self.codes = {inputdev.keycode(k): k for k in ROUTE_KEYS}
        self.fds = inputdev.open_keyboards()
        self._cancel_r, self._cancel_w = os.pipe()
        self._lock = threading.Lock()

    def cancel(self):
        """Ends record() from another thread; harmless once the recorder is closed."""
        with self._lock:
            if self._cancel_w is not None:
                os.write(self._cancel_w, b"x")

    def _events(self):
        size = inputdev.INPUT_EVENT.size
        while True:
            ready, _, _ = select.select([*self.fds, self._cancel_r], [], [])
            if self._cancel_r in ready:
                return
            for fd in ready:
                try:
                    data = os.read(fd, size * 64)
                except BlockingIOError:
                    continue
                for off in range(0, len(data) - size + 1, size):
                    _, _, etype, code, value = inputdev.INPUT_EVENT.unpack_from(data, off)
                    if etype == inputdev.EV_KEY and value in (0, 1):
                        yield time.monotonic(), code, value == 1

    def record(self, on_start=None):
        """Waits for the hotkey, records until it is pressed again and returns
        [(seconds, key, down), ...]; None if cancelled."""
        events, start, held = [], None, set()
        for t, code, down in self._events():
            if code == self.hotkey_code:
                if down and start is not None:
                    end = round(t - start, 4)
                    return events + [(end, self.codes[c], False) for c in sorted(held)]
                if down:
                    start = t
                    if on_start:
                        on_start()
                continue
            if start is None or code not in self.codes or down == (code in held):
                continue
            (held.add if down else held.discard)(code)
            events.append((round(t - start, 4), self.codes[code], down))
        return None

    def close(self):
        with self._lock:
            if self._cancel_w is None:
                return
            for fd in (*self.fds, self._cancel_r, self._cancel_w):
                os.close(fd)
            self.fds = []
            self._cancel_r = self._cancel_w = None


def save_route(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"events": events}, indent=1))
    tmp.replace(path)
