"""
Fisch's UI state read straight from the Roblox client's memory in Sober,
read-only, with process_vm_readv. The Roblox process lives in Sober's
sandbox, whose user namespace belongs to this same user, so the kernel lets
us read it without root even with Yama's ptrace_scope = 1.

Layout of the Android Roblox client, as measured on 2026-10-31 (attach()
re-derives every offset from known relationships, so a shifted layout after
a Roblox update is found again instead of read wrongly):

    Instance   +0x08  pointer to itself
               +0x18  ClassDescriptor*: class name std::string* at +0x08
               +0x68  parent
               +0x70  name: shared name object, std::string at +0x08
               +0x78  children: vector<shared_ptr<Instance>>* (16-byte elements)
    GuiBase2d  +0xfc  AbsolutePosition (float x, y)
               +0x104 AbsoluteSize (float w, h), in client pixels; a ScreenGui
                      that keeps the 58 px top-bar inset reads (0, 58) 1880x940
                      in a 1880x998 window, one that ignores it (0, 0) 1880x998
    TextLabel  +0xb78, +0xdc0, +0xe08, +0xe20  its text (std::string), the
                      latter two with rich-text tags; found again from the
                      HUD's "Level N" label

Strings are libc++ std::string: short ones inline with the length doubled in
the first byte, long ones as {capacity|1, size, data*}.
"""

import ctypes
import ctypes.util
import os
import re
import struct
import threading
import time

import numpy as np

from .config import CAST_GRADES

SOBER_COMM = "sober"
GAME_COMM = "Main"
READ_CHUNK = 64 << 20
MAX_HEAP_MAP = 1 << 30
IOV_MAX = 1024
MAX_CHILDREN = 4096
SCAN_OFFSETS = range(0x20, 0x140, 8)    # where name/parent/children may sit in an Instance
GEOMETRY_SPAN = 0x400                   # bytes of a GuiBase2d searched for AbsoluteSize
ALIVE_CHECK_S = 1.0
ATTACH_RETRY_S = 10.0                   # an attach scans ~2 GB of the game's heap (~5 s)
TEXT_SCAN = range(0x80, 0x1200, 8)      # where a TextLabel may keep its text
LEVEL_TEXT = re.compile(r"Level \d+")

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


class _IoVec(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("len", ctypes.c_size_t)]


_libc.process_vm_readv.argtypes = [ctypes.c_int, ctypes.POINTER(_IoVec), ctypes.c_ulong,
                                   ctypes.POINTER(_IoVec), ctypes.c_ulong, ctypes.c_ulong]
_libc.process_vm_readv.restype = ctypes.c_ssize_t


class LayoutError(Exception):
    pass


def find_game_pid():
    """PID of Sober's Roblox process ("Main", child of "sober") owned by this user, or None."""
    uid = os.getuid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                stat = f.read()
            comm = stat[stat.index("(") + 1:stat.rindex(")")]
            ppid = int(stat[stat.rindex(")") + 2:].split()[1])
            if comm != GAME_COMM or os.stat(f"/proc/{entry}").st_uid not in (uid, 0):
                continue
            with open(f"/proc/{ppid}/comm") as f:
                if f.read().strip() == SOBER_COMM:
                    return int(entry)
        except (OSError, ValueError):
            continue
    return None


class Process:
    """Read-only view of another process's memory."""

    def __init__(self, pid):
        self.pid = pid

    def read(self, addr, n):
        buf = ctypes.create_string_buffer(n)
        local, remote = _IoVec(ctypes.cast(buf, ctypes.c_void_p), n), _IoVec(addr, n)
        got = _libc.process_vm_readv(self.pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0)
        return buf.raw[:got] if got > 0 else b""

    def read_many(self, addrs, n):
        """n bytes at each address; unreadable ones come back short or empty."""
        out = []
        for start in range(0, len(addrs), IOV_MAX):
            batch = addrs[start:start + IOV_MAX]
            buf = ctypes.create_string_buffer(n * len(batch))
            local = (_IoVec * len(batch))(*[_IoVec(ctypes.addressof(buf) + i * n, n) for i in range(len(batch))])
            remote = (_IoVec * len(batch))(*[_IoVec(a, n) for a in batch])
            got = _libc.process_vm_readv(self.pid, local, len(batch), remote, len(batch), 0)
            if got == n * len(batch):
                out += [buf.raw[i * n:(i + 1) * n] for i in range(len(batch))]
            else:   # a bad address stops the batch: fall back to one read each
                out += [self.read(a, n) for a in batch]
        return out

    def u64(self, addr):
        b = self.read(addr, 8)
        return struct.unpack("<Q", b)[0] if len(b) == 8 else 0

    def maps(self):
        rows = []
        with open(f"/proc/{self.pid}/maps") as f:
            for line in f:
                p = line.split()
                lo, hi = (int(x, 16) for x in p[0].split("-"))
                rows.append((lo, hi, p[1], p[5] if len(p) > 5 else ""))
        return rows

    def resident_heap(self):
        """Resident, writable anonymous mappings up to 1 GiB: where instances live."""
        out, cur = [], None
        with open(f"/proc/{self.pid}/smaps") as f:
            for line in f:
                p = line.split()
                if len(p) >= 5 and "-" in p[0] and ":" in p[3]:
                    lo, hi = (int(x, 16) for x in p[0].split("-"))
                    name = p[5] if len(p) > 5 else ""
                    cur = [lo, hi, p[1].startswith("rw") and (not name or name.startswith("[anon")), 0]
                    out.append(cur)
                elif p and p[0] == "Rss:" and cur is not None:
                    cur[3] = int(p[1])
        return [(lo, hi) for lo, hi, rw, rss in out if rw and rss and hi - lo <= MAX_HEAP_MAP]

    def string(self, addr):
        """libc++ std::string at addr, or None."""
        b = self.read(addr, 24)
        if len(b) < 24:
            return None
        if b[0] & 1 == 0:
            n = b[0] >> 1
            return b[1:1 + n].decode(errors="replace") if n <= 22 else None
        size, data = struct.unpack("<QQ", b[8:24])
        if size > 1024:
            return None
        raw = self.read(data, size)
        return raw.decode(errors="replace") if len(raw) == size else None


def _short_string(text):
    """The leading bytes of a libc++ short std::string holding text, terminator included."""
    raw = text.encode()
    return bytes([len(raw) << 1]) + raw + b"\x00"


class RobloxMemory:
    """Attached to one Roblox process: resolves PlayerGui paths and reads GUI geometry."""

    def __init__(self, log):
        self.log = log
        self.proc = None
        self.player_gui = None
        self.player_name = None
        self.client_h = None
        self.off = {}
        self._cache = {}

    # -- attaching -----------------------------------------------------------

    def attach(self):
        """Find the game, the PlayerGui and the layout. False (logged) when any step fails."""
        pid = find_game_pid()
        if pid is None:
            self.log("memory: Sober's Roblox process is not running")
            return False
        self.proc = Process(pid)
        if not self.proc.read(self.proc.maps()[0][0], 8):
            self.log("memory: the kernel does not let this user read the Roblox process")
            return False
        desc = self._class_descriptor("PlayerGui")
        if desc is None:
            self.log("memory: no PlayerGui class descriptor found")
            return False
        for inst, desc_off in self._instances_of(desc):
            if self._learn_layout(inst, desc_off):
                self.player_gui = inst
                self._cache.clear()
                self.log(f"memory: attached to Roblox (pid {pid}), player {self.player_name}, "
                         f"layout {', '.join(f'{k} {v:#x}' for k, v in self.off.items())}")
                return True
        self.log("memory: found PlayerGui objects but could not work out their layout")
        return False

    def _lib_ranges(self):
        """(read-only data, writable data) address ranges of the Roblox library, which
        Sober maps from an anonymous memfd ("/memfd:", unlike its own "/memfd:sober")."""
        maps = self.proc.maps()
        lib = [(lo, hi, perm) for lo, hi, perm, name in maps if name == "/memfd:"]
        if not lib:
            raise LayoutError("the Roblox library mapping was not found")
        ro = [(lo, hi) for lo, hi, perm in lib if perm.startswith("r--")]
        end = max(hi for _, hi, _ in lib)
        data = [(lo, hi) for lo, hi, perm, name in maps
                if perm.startswith("rw") and (name in ("", "/memfd:") or name.startswith("[anon")) and end - (32 << 20) <= lo < end + (64 << 20)]
        return ro, data

    def _class_descriptor(self, class_name):
        """A ClassDescriptor is {vtable in the library's read-only data, std::string* name}."""
        try:
            ro, data = self._lib_ranges()
        except LayoutError as e:
            self.log(f"memory: {e}")
            return None
        want = _short_string(class_name)
        for lo, hi in data:
            raw = self.proc.read(lo, hi - lo)
            arr = np.frombuffer(raw[:len(raw) // 8 * 8], dtype=np.uint64)
            if arr.size < 2:
                continue
            vt = np.zeros(arr.size, dtype=bool)
            for rlo, rhi in ro:
                vt |= (arr >= rlo) & (arr < rhi)
            idx = np.flatnonzero(vt[:-1] & (arr[1:] > 0x10000))
            names = [int(arr[i + 1]) for i in idx]
            for i, blob in zip(idx, self.proc.read_many(names, len(want))):
                if blob == want:
                    return lo + int(i) * 8
        return None

    def _instances_of(self, desc):
        """(instance, descriptor offset) for every object pointing at desc whose
        word at +0x08 points back at itself."""
        found = []
        for lo, hi in self.proc.resident_heap():
            for start in range(lo, hi, READ_CHUNK):
                raw = self.proc.read(start, min(READ_CHUNK, hi - start))
                arr = np.frombuffer(raw[:len(raw) // 8 * 8], dtype=np.uint64)
                for i in np.flatnonzero(arr == np.uint64(desc)):
                    loc = start + int(i) * 8
                    for desc_off in (0x10, 0x18, 0x20):
                        if self.proc.u64(loc - desc_off + 8) == loc - desc_off:
                            found.append((loc - desc_off, desc_off))
        return found

    def _name_at(self, obj, off):
        p = self.proc.u64(obj + off)
        return self.proc.string(p + 8) if p else None

    def _learn_layout(self, inst, desc_off):
        p = self.proc
        off = {"desc": desc_off}
        name_off = next((o for o in SCAN_OFFSETS if self._name_at(inst, o) == "PlayerGui"), None)
        if name_off is None:
            return False
        off["name"] = name_off

        def class_of(obj):
            d = p.u64(obj + desc_off)
            return p.string(p.u64(d + 8)) if d else None

        parent_off = next((o for o in SCAN_OFFSETS if o != name_off and class_of(p.u64(inst + o) or 1) == "Player"), None)
        if parent_off is None:
            return False
        off["parent"] = parent_off
        for o in SCAN_OFFSETS:
            vec = p.u64(inst + o)
            begin, end = (p.u64(vec), p.u64(vec + 8)) if vec else (0, 0)
            if not begin or end <= begin or end - begin > 16 * MAX_CHILDREN or (end - begin) % 16:
                continue
            kids = [p.u64(begin + i) for i in range(0, min(end - begin, 16 * 8), 16)]
            if kids and all(k and p.u64(k + parent_off) == inst for k in kids):
                off["children"] = o
                break
        else:
            return False
        self.off = off
        self.player_name = self._name_at(p.u64(inst + parent_off), name_off)
        return self._learn_geometry(inst)

    def _learn_geometry(self, player_gui):
        """AbsolutePosition/AbsoluteSize: every enabled ScreenGui spans the client
        width and ends at the client bottom, starting at y 0 or below the top-bar
        inset. Disabled ones read all zeros."""
        guis = [c for c in self.children(player_gui) if self.class_name(c) == "ScreenGui"][:12]
        blobs = [self.proc.read(g, GEOMETRY_SPAN) for g in guis]
        for o in range(0x88, GEOMETRY_SPAN - 8, 4):
            rects = [struct.unpack_from("<4f", b, o - 8) for b in blobs if len(b) >= o + 8]
            rects = [r for r in rects if any(r)]
            if len(rects) < 3 or any(x != 0 for x, _, _, _ in rects):
                continue
            widths = {w for _, _, w, _ in rects}
            bottoms = {y + h for _, y, _, h in rects}
            if len(widths) == 1 and len(bottoms) == 1:
                w, bottom = widths.pop(), bottoms.pop()
                if 320 <= w <= 8000 and 240 <= bottom <= 5000 and w == int(w) and bottom == int(bottom):
                    self.off["abs_pos"], self.off["abs_size"] = o - 8, o
                    self.client_h = bottom
                    return True
        return False

    # -- reading ---------------------------------------------------------------

    def alive(self):
        return (self.player_gui is not None and self.proc is not None
                and self.name(self.player_gui) == "PlayerGui")

    def name(self, inst):
        return self._name_at(inst, self.off["name"])

    def class_name(self, inst):
        d = self.proc.u64(inst + self.off["desc"])
        return self.proc.string(self.proc.u64(d + 8)) if d else None

    def parent(self, inst):
        return self.proc.u64(inst + self.off["parent"])

    def children(self, inst):
        vec = self.proc.u64(inst + self.off["children"])
        if not vec:
            return []
        begin, end = self.proc.u64(vec), self.proc.u64(vec + 8)
        if not begin or end <= begin or end - begin > 16 * MAX_CHILDREN:
            return []
        raw = self.proc.read(begin, end - begin)
        return [v for v in struct.unpack(f"<{len(raw) // 8}Q", raw[:len(raw) // 8 * 8])[::2] if v]

    def child(self, inst, wanted):
        for c in self.children(inst):
            if self.name(c) == wanted:
                return c
        return None

    def gui(self, *names):
        """Instance at PlayerGui.<names...>, or None. Resolved paths are cached
        and re-checked (name and parent) on every use."""
        key = names
        hit = self._cache.get(key)
        if hit and self.name(hit) == names[-1] and self.parent(hit) == (self._cache.get(names[:-1]) or self.player_gui):
            return hit
        inst = self.player_gui
        for i, n in enumerate(names):
            inst = self.child(inst, n) if inst else None
            if inst:
                self._cache[names[:i + 1]] = inst
        return inst

    def rect(self, inst):
        """(x, y, w, h) in client pixels, or None when unreadable."""
        raw = self.proc.read(inst + self.off["abs_pos"], 16)
        return struct.unpack("<4f", raw) if len(raw) == 16 else None


class FischUI(RobloxMemory):
    """Fisch's fishing UI, as observed in Sober on 2026-10-31:

        PlayerGui.shakeui.safezone.<ImageButton>        the shake button
        PlayerGui.reel.bar.{playerbar, fish}            the reel minigame;
                    .bar.progress.bar                   progress = its width / progress's width;
                                                        the reel slides in from below the window
        Workspace.<player>.HumanoidRootPart
            .power.powerbar.{bar, green}                cast power bar: bar is the fill, its
                                                        height over powerbar's how full it is
            .powerfeedback.title                        the cast's grade ("Meh..", "Fine.",
                                                        "Good!", "Great!", "Amazing!!", "PERFECT!")

    Each cast creates a new power and powerfeedback; the previous cast's linger
    beside them for a while, so the newest is the live one.
    """

    def __init__(self, log):
        super().__init__(log)
        self._character = None
        self._text_offsets = None

    def _newest_child(self, inst, wanted):
        found = None
        for c in self.children(inst):
            if self.name(c) == wanted:
                found = c
        return found

    def text(self, inst):
        """A TextLabel's text, or None."""
        if self._text_offsets is None:
            lvl = self.gui("hud", "safezone", "lvl")
            self._text_offsets = [o for o in TEXT_SCAN if lvl and LEVEL_TEXT.fullmatch(self.proc.string(lvl + o) or "")]
            if not self._text_offsets:
                self.log("memory: could not find where text labels keep their text")
        for o in self._text_offsets:
            s = self.proc.string(inst + o)
            if s:
                return s
        return None

    def shake_button(self):
        """(x, y, w, h) of the shake button in client pixels, or None."""
        zone = self.gui("shakeui", "safezone")
        if not zone:
            return None
        for c in self.children(zone):
            if self.class_name(c) == "ImageButton":
                r = self.rect(c)
                if r and r[2] > 0 and r[3] > 0:
                    return r
        return None

    def reel(self):
        """(playerbar rect, fish rect, progress 0..1) while the reel is on screen, or None."""
        bar = self.gui("reel", "bar")
        if not bar:
            return None
        frame = self.rect(bar)
        if not frame or frame[1] + frame[3] > self.client_h:     # still sliding in, or out
            return None
        player, fish = self.gui("reel", "bar", "playerbar"), self.gui("reel", "bar", "fish")
        track, fill = self.gui("reel", "bar", "progress"), self.gui("reel", "bar", "progress", "bar")
        if not player or not fish:
            return None
        p, f = self.rect(player), self.rect(fish)
        t, b = (self.rect(track), self.rect(fill)) if track and fill else (None, None)
        progress = b[2] / t[2] if t and b and t[2] > 0 else None
        return p, f, progress

    def _hrp(self):
        if self._character is None or self.name(self._character) != self.player_name:
            players = self.parent(self.parent(self.player_gui))
            workspace = self.child(self.parent(players), "Workspace")
            self._character = self.child(workspace, self.player_name) if workspace else None
        return self.child(self._character, "HumanoidRootPart") if self._character else None

    def power_fill(self):
        """How full the cast power bar is (0..1) while it shows, or None."""
        hrp = self._hrp()
        power = self._newest_child(hrp, "power") if hrp else None
        bar = self.child(power, "powerbar") if power else None
        fill = self.child(bar, "bar") if bar else None
        if not fill:
            return None
        f, b = self.rect(fill), self.rect(bar)
        if not f or not b or b[3] <= 0:
            return None
        return f[3] / b[3]

    def grade_labels(self):
        """The cast grade labels under the character now; pass them to cast_grade()."""
        hrp = self._hrp()
        return {c for c in self.children(hrp) if self.name(c) == "powerfeedback"} if hrp else set()

    def cast_grade(self, earlier):
        """The grade ("meh" … "perfect") on a grade label not in `earlier`, or None."""
        hrp = self._hrp()
        for c in self.children(hrp) if hrp else ():
            if c in earlier or self.name(c) != "powerfeedback":
                continue
            title = self.child(c, "title")
            text = (self.text(title) or "").lower() if title else ""
            grade = next((g for g in CAST_GRADES if g in text), None)
            if grade:
                return grade
        return None


class MemoryLink:
    """The FischUI attachment, owned by the app and shared by every session:
    attached in the background, dropped when the game goes away and attached
    again (to the new process) when it is back."""

    def __init__(self, log):
        self.log = log
        self.ui = None
        self._thread = None
        self._checked = -float("inf")
        self._last_try = -float("inf")

    def get(self):
        """The attached FischUI, or None while there is none (an attach may be under way)."""
        now = time.monotonic()
        if self.ui is not None and now - self._checked >= ALIVE_CHECK_S:
            self._checked = now
            if not self.ui.alive():
                self.log("memory: lost the game's memory, reading the screen until it is back")
                self.ui = None
        if self.ui is None:
            self.ensure()
        return self.ui

    def ensure(self):
        """Start attaching unless attached, attaching, or tried within ATTACH_RETRY_S."""
        now = time.monotonic()
        if self.ui is not None or (self._thread and self._thread.is_alive()) or now - self._last_try < ATTACH_RETRY_S:
            return
        self._last_try = now

        def attach():
            ui = FischUI(self.log)
            if ui.attach():
                self.ui = ui
        self._thread = threading.Thread(target=attach, daemon=True, name="memory-attach")
        self._thread.start()
