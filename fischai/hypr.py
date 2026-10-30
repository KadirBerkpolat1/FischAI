"""
The game window on Hyprland: lookup, focus, geometry and the window state
changes the aquarium sweep needs, all through `hyprctl` (Hyprland >= 0.57
Lua dispatchers). FocusWatcher follows focus changes on the event socket so
the macro pauses the moment Sober loses focus instead of clicking into
whatever window took it.
"""

import json
import os
import socket
import subprocess
import threading
from dataclasses import dataclass

SOBER_CLASS = "org.vinegarhq.Sober"


class WindowError(Exception):
    pass


@dataclass(frozen=True)
class Window:
    address: str
    title: str
    x: int
    y: int
    width: int
    height: int
    floating: bool
    fullscreen: int
    monitor: int

    @property
    def rect(self):
        return self.x, self.y, self.width, self.height

    @property
    def center(self):
        return self.x + self.width // 2, self.y + self.height // 2


def require_hyprland():
    if not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        raise SystemExit("FischAI needs a running Hyprland session (HYPRLAND_INSTANCE_SIGNATURE is unset).")


def _hyprctl_json(*args):
    out = subprocess.run(["hyprctl", "-j", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def dispatch(expr):
    out = subprocess.run(["hyprctl", "dispatch", expr], capture_output=True, text=True)
    reply = out.stdout.strip()
    if out.returncode != 0 or reply != "ok":
        raise WindowError(f"hyprctl dispatch {expr!r} failed: {reply or out.stderr.strip()}")


def norm_address(address):
    return address.lower().removeprefix("0x")


def _to_window(client):
    return Window(
        address=client["address"],
        title=client["title"],
        x=client["at"][0],
        y=client["at"][1],
        width=client["size"][0],
        height=client["size"][1],
        floating=bool(client.get("floating")),
        fullscreen=int(client.get("fullscreen") or 0),
        monitor=client["monitor"],
    )


def find_window(match=SOBER_CLASS):
    """The game window: exact class match first, then a case-insensitive title substring."""
    needle = match.lower()
    clients = [c for c in _hyprctl_json("clients") if c.get("mapped") and not c.get("hidden")]
    clients.sort(key=lambda c: c.get("focusHistoryID", 1 << 30))
    for client in clients:
        if client["class"].lower() == needle:
            return _to_window(client)
    for client in clients:
        if needle in client["title"].lower():
            return _to_window(client)
    raise WindowError(f"No window with class or title matching '{match}'. Is Sober running?")


def refresh(win):
    for client in _hyprctl_json("clients"):
        if client["address"] == win.address:
            return _to_window(client)
    raise WindowError("The game window was closed.")


def active_address():
    active = _hyprctl_json("activewindow")
    return norm_address(active.get("address", "")) if isinstance(active, dict) else ""


def focus(win):
    if active_address() != norm_address(win.address):
        dispatch(f'hl.dsp.focus({{ window = "address:{win.address}" }})')


def set_fullscreen(win, on):
    action = "set" if on else "unset"
    dispatch(f'hl.dsp.window.fullscreen({{ window = "address:{win.address}", action = "{action}", mode = "fullscreen" }})')


def set_floating(win, on):
    action = "set" if on else "unset"
    dispatch(f'hl.dsp.window.float({{ window = "address:{win.address}", action = "{action}" }})')


def resize_exact(win, width, height):
    dispatch(f'hl.dsp.window.resize({{ window = "address:{win.address}", x = {int(width)}, y = {int(height)}, exact = true }})')


def move_exact(win, x, y):
    dispatch(f'hl.dsp.window.move({{ window = "address:{win.address}", x = {int(x)}, y = {int(y)}, exact = true }})')


def monitor_rect(monitor_id):
    """Logical (x, y, width, height) of a monitor."""
    for m in _hyprctl_json("monitors"):
        if m["id"] == monitor_id:
            scale = m["scale"] or 1
            return m["x"], m["y"], round(m["width"] / scale), round(m["height"] / scale)
    raise WindowError(f"Monitor {monitor_id} not found.")


def layout_bounds():
    """Logical bounding box (x, y, width, height) of all monitors."""
    boxes = []
    for m in _hyprctl_json("monitors"):
        scale = m["scale"] or 1
        boxes.append((m["x"], m["y"], m["x"] + round(m["width"] / scale), m["y"] + round(m["height"] / scale)))
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    return min_x, min_y, max(b[2] for b in boxes) - min_x, max(b[3] for b in boxes) - min_y


def _event_socket_path():
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return os.path.join(runtime, "hypr", os.environ["HYPRLAND_INSTANCE_SIGNATURE"], ".socket2.sock")


class FocusWatcher(threading.Thread):
    """Tracks the focused window from `activewindowv2` events, so checking
    focus before every click costs no subprocess."""

    def __init__(self, address):
        super().__init__(daemon=True, name="focus-watcher")
        self.address = norm_address(address)
        self.closed = False
        self._active = active_address()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(_event_socket_path())

    def is_focused(self):
        if not self.is_alive():
            return active_address() == self.address
        return self._active == self.address

    def run(self):
        buf = b""
        while True:
            try:
                chunk = self._sock.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                event, _, data = line.partition(b">>")
                if event == b"activewindowv2":
                    self._active = norm_address(data.decode(errors="replace"))
                elif event == b"closewindow" and norm_address(data.decode(errors="replace")) == self.address:
                    self.closed = True

    def stop(self):
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
