"""
Mouse and keyboard as two virtual devices on /dev/uinput, addressed in
Hyprland layout coordinates (input lands in the focused window), plus a
read-only global hotkey listener on the real keyboards in /dev/input.
"""

import fcntl
import os
import select
import struct
import threading
import time

from . import hypr

VIRTUAL_DEVICE_PREFIX = "FischAI virtual"

KEYCODES = {
    "esc": 1, "escape": 1, "tab": 15, "enter": 28, "return": 28, "space": 57, "backspace": 14,
    "leftshift": 42, "shift": 42, "leftctrl": 29, "ctrl": 29, "leftalt": 56, "alt": 56,
    "pause": 119, "home": 102, "end": 107, "insert": 110, "delete": 111, "f11": 87, "f12": 88,
    "up": 103, "left": 105, "right": 106, "down": 108,
}
KEYCODES.update({f"f{n}": 58 + n for n in range(1, 11)})
KEYCODES.update({str(n): 1 + n for n in range(1, 10)})
KEYCODES["0"] = 11
for _row, _first in (("qwertyuiop", 16), ("asdfghjkl", 30), ("zxcvbnm", 44)):
    KEYCODES.update({ch: _first + i for i, ch in enumerate(_row)})


def keycode(name):
    try:
        return KEYCODES[name.strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown key '{name}'. Try something like: f3, f8, pause, m.") from None


EV_SYN, EV_KEY, EV_REL, EV_ABS = 0x00, 0x01, 0x02, 0x03
SYN_REPORT = 0
REL_WHEEL = 0x08
ABS_X, ABS_Y = 0x00, 0x01
ABS_MAX = 32767
BTN_LEFT, BTN_RIGHT, BTN_MIDDLE = 0x110, 0x111, 0x112
BUS_USB, BUS_VIRTUAL = 0x03, 0x06

INPUT_EVENT = struct.Struct("llHHi")


def _iow(nr, size):
    return (1 << 30) | (size << 16) | (ord("U") << 8) | nr


UI_DEV_CREATE = 0x5501
UI_DEV_SETUP = _iow(3, struct.calcsize("4H80sI"))
UI_ABS_SETUP = _iow(4, struct.calcsize("H2x6i"))
UI_SET_EVBIT = _iow(100, 4)
UI_SET_KEYBIT = _iow(101, 4)
UI_SET_RELBIT = _iow(102, 4)
UI_SET_ABSBIT = _iow(103, 4)


class _UInputDevice:
    def __init__(self, name, product, keys, absolute=False):
        try:
            self.fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
        except PermissionError:
            raise SystemExit("No write access to /dev/uinput. Add a udev rule or ACL for your user.") from None
        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_KEY)
        for code in keys:
            fcntl.ioctl(self.fd, UI_SET_KEYBIT, code)
        bus = BUS_VIRTUAL
        if absolute:
            # QEMU usb-tablet profile (ABS 0..32767 + wheel): libinput ignores
            # motion from a bare ABS_X/ABS_Y + BTN_LEFT device.
            bus = BUS_USB
            fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_REL)
            fcntl.ioctl(self.fd, UI_SET_RELBIT, REL_WHEEL)
            fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_ABS)
            for axis in (ABS_X, ABS_Y):
                fcntl.ioctl(self.fd, UI_SET_ABSBIT, axis)
                fcntl.ioctl(self.fd, UI_ABS_SETUP, struct.pack("H2x6i", axis, 0, 0, ABS_MAX, 0, 0, 0))
        fcntl.ioctl(self.fd, UI_DEV_SETUP, struct.pack("4H80sI", bus, 0x1209, product, 1, name.encode()[:79], 0))
        fcntl.ioctl(self.fd, UI_DEV_CREATE)

    def emit(self, *events):
        buf = b"".join(INPUT_EVENT.pack(0, 0, t, c, v) for t, c, v in events)
        os.write(self.fd, buf + INPUT_EVENT.pack(0, 0, EV_SYN, SYN_REPORT, 0))

    def close(self):
        if self.fd is not None:
            os.close(self.fd)  # closing the fd destroys the device
            self.fd = None


class VirtualInput:
    """Absolute pointer + keyboard. Tracks held buttons/keys so release_all()
    can unstick them when the macro stops or pauses."""

    def __init__(self):
        self.origin_x, self.origin_y, self.width, self.height = hypr.layout_bounds()
        self.pointer = _UInputDevice(f"{VIRTUAL_DEVICE_PREFIX} pointer", 0x0001,
                                     [BTN_LEFT, BTN_RIGHT, BTN_MIDDLE], absolute=True)
        self.keyboard = _UInputDevice(f"{VIRTUAL_DEVICE_PREFIX} keyboard", 0x0002,
                                      sorted(set(KEYCODES.values())))
        self._held_buttons = set()
        self._held_keys = set()
        self._lock = threading.Lock()
        # The compositor needs a moment to pick up new devices before events count.
        time.sleep(0.5)

    @property
    def mouse_down(self):
        return BTN_LEFT in self._held_buttons

    def move(self, x, y):
        ax = round((int(x) - self.origin_x + 0.5) * ABS_MAX / self.width)
        ay = round((int(y) - self.origin_y + 0.5) * ABS_MAX / self.height)
        with self._lock:
            self.pointer.emit((EV_ABS, ABS_X, min(max(ax, 0), ABS_MAX)), (EV_ABS, ABS_Y, min(max(ay, 0), ABS_MAX)))

    def button(self, down, button=BTN_LEFT):
        with self._lock:
            self.pointer.emit((EV_KEY, button, 1 if down else 0))
            (self._held_buttons.add if down else self._held_buttons.discard)(button)

    def scroll(self, notches):
        """Positive scrolls up, negative down."""
        with self._lock:
            self.pointer.emit((EV_REL, REL_WHEEL, int(notches)))

    def key(self, name, down):
        code = keycode(name)
        with self._lock:
            self.keyboard.emit((EV_KEY, code, 1 if down else 0))
            (self._held_keys.add if down else self._held_keys.discard)(code)

    def press(self, name, hold_s=0.03):
        self.key(name, True)
        time.sleep(hold_s)
        self.key(name, False)

    def release_key(self, name):
        code = keycode(name)
        with self._lock:
            if code in self._held_keys:
                self.keyboard.emit((EV_KEY, code, 0))
                self._held_keys.discard(code)

    def release_all(self):
        with self._lock:
            for b in list(self._held_buttons):
                self.pointer.emit((EV_KEY, b, 0))
            for k in list(self._held_keys):
                self.keyboard.emit((EV_KEY, k, 0))
            self._held_buttons.clear()
            self._held_keys.clear()

    def close(self):
        self.release_all()
        self.pointer.close()
        self.keyboard.close()


def _keyboard_event_paths():
    """Event nodes of real keyboards (EV_KEY + EV_REP, 'kbd' handler), excluding
    our own virtual devices."""
    with open("/proc/bus/input/devices") as f:
        blocks = f.read().strip().split("\n\n")
    paths = []
    for block in blocks:
        name, handlers, ev = "", [], 0
        for line in block.splitlines():
            if line.startswith("N: Name="):
                name = line[len("N: Name="):].strip('"')
            elif line.startswith("H: Handlers="):
                handlers = line[len("H: Handlers="):].split()
            elif line.startswith("B: EV="):
                ev = int(line[len("B: EV="):], 16)
        if name.startswith(VIRTUAL_DEVICE_PREFIX) or "kbd" not in handlers:
            continue
        if not (ev & (1 << EV_KEY) and ev & (1 << 0x14)):
            continue
        paths += [f"/dev/input/{h}" for h in handlers if h.startswith("event")]
    return paths


def open_keyboards():
    """Non-blocking read fds of every real keyboard; the devices are not grabbed."""
    fds = []
    for path in _keyboard_event_paths():
        try:
            fds.append(os.open(path, os.O_RDONLY | os.O_NONBLOCK))
        except OSError:
            pass
    if not fds:
        raise PermissionError("Cannot read any keyboard in /dev/input. Add your user to the 'input' group "
                              "(sudo usermod -aG input $USER) and log in again.")
    return fds


class HotkeyListener(threading.Thread):
    """Calls on_press(key_name) when one of `keys` is pressed on any keyboard,
    whichever window is focused."""

    def __init__(self, keys, on_press):
        super().__init__(daemon=True, name="hotkeys")
        self.codes = {keycode(k): k for k in keys}
        self.on_press = on_press
        self.fds = open_keyboards()

    def run(self):
        size = INPUT_EVENT.size
        while self.fds:
            ready, _, _ = select.select(self.fds, [], [])
            for fd in ready:
                try:
                    data = os.read(fd, size * 64)
                except BlockingIOError:
                    continue
                except OSError:
                    self.fds.remove(fd)  # keyboard unplugged
                    continue
                for off in range(0, len(data) - size + 1, size):
                    _, _, etype, code, value = INPUT_EVENT.unpack_from(data, off)
                    if etype == EV_KEY and value == 1 and code in self.codes:
                        self.on_press(self.codes[code])
