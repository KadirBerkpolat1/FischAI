"""
The fishing loop, ported from ExoPanda Fischer's AI mode (fisch_main.lua and
its feature modules). Fishing is an ordered state machine: cast -> shake ->
reel -> cast. Between catches the idle chain runs one action per pass, in
priority order: aquarium, potions, totem, cast.

Coordinates are Hyprland layout pixels. Input only goes out while Sober has
focus: losing it pauses the loop and releases every held button and key.
"""

import math
import queue
import threading
import time
from dataclasses import dataclass, field

from . import hypr, recovery, screen
from .detector import Box, Boxes
from .reeler import REEL_LOCK_S, Reeler
from .vision import PowerBar, appraise_row, cast_grade, cast_text_crop

# Reel entry (fisch_data.lua, Windows values).
STALE_S = 0.250
HOLDOVER_S = 0.500
ENTRY_MIN_FISH_CONF = 0.45
ENTRY_FAST_FISH_CONF = 0.70
DUAL_ENTRY_MIN_CONF = 0.45
ENTRY_DEBOUNCE_TICKS = 2
SHAKE_TO_REEL_HINT_TICKS = 3
MIN_COUNTED_REEL_S = 1.5        # shorter "reels" are detector blips, not catches
MAX_SNAPSHOTS = 400
PROMPT_CHECK_AFTER_S = 60.0     # no reel for this long: start looking for Roblox's error prompt
PROMPT_CHECK_EVERY_S = 30.0
PROGRESS_EPS = 0.002            # catch progress read from memory: smaller steps back are rounding

REEL_TICK_S = 0.016
SHAKE_TICK_S = 0.008
IDLE_TICK_FAST_S = 0.100
IDLE_TICK_SLOW_S = 0.250
SILENCE_RAMP_S = 1.5
WINDOW_REFRESH_S = 1.0

SHAKE_TIMEOUT_S = 16.0          # without a successful click
SHAKE_FIRST_TIMEOUT_S = 30.0    # before the first button: in Sober it came 1-19 s after the cast
# The press spans two game frames at 60 fps so the down and the up are never
# read in the same frame.
SHAKE_SETTLE_S = 0.016
SHAKE_HOLD_S = 0.035
SHAKE_JIGGLE_DY_PX = 3
SHAKE_JIGGLE_GAP_S = 0.008
SHAKE_SAME_BUTTON_PX = 30
SHAKE_REARM_S = 0.300           # same spot, never seen gone: the click missed, retry
SHAKE_DEAD_AFTER_CLICKS = 3     # clicks a "button" may ignore before its spot is left alone

# The rod lives in hotbar slot 1. Alternating 2 -> 1 presses always end with
# the rod in hand; pressing 1 alone would put an already-held rod away.
ROD_SLOT = "1"
OTHER_SLOT = "2"
EQUIP_STEP_S = 0.120

# Casting on the power bar. Fisch grades a cast by how full the bar is when the
# fill stops, at the first game frame after the release. Read from memory on 26
# casts: Meh.. up to 0.16, Fine. 0.22-0.39, Good! 0.57, Great! 0.63-0.77,
# Amazing!! 0.81-0.92, PERFECT! 0.95 and up; the fill turns back down at 1.
# The release goes out when the rising fill, projected ahead at its speed,
# would stop in the middle of the chosen grade's band.
CAST_STOP_AT = {"meh": 0.10, "fine": 0.30, "good": 0.50, "great": 0.70, "amazing": 0.875, "perfect": 1.0}
CAST_BAR_WAIT_S = 0.5           # the bar shows ~0.1 s after the press
CAST_MAX_S = 2.0                # the fill rises, falls and rises again in ~1.3 s
CAST_MEMORY_POLL_S = 0.001
GAME_FRAME_S = 1 / 60
CAST_MEMORY_LEAD_S = 1.5 * GAME_FRAME_S     # memory has each frame's fill as it is drawn; it stops at the next one
CAST_SCREEN_LEAD_S = 0.051      # the screen shows it ~43 ms late, and it stops half a frame after the release
CAST_SCREEN_READ_S = 0.025      # one screen read; a release due sooner is waited out instead
CAST_FILL_START_S = 0.085       # the fill starts rising this long after the press,
CAST_TYPICAL_SPEED = 2.0        # at 1-3 bars per second
CAST_MAX_SPEED = 4.0
CAST_SCREEN_FIRST_S = 0.17      # and is not readable on the screen any earlier
CAST_GRADE_READ_S = 1.4         # the grade text stays up about a second
CAST_GRADE_POLL_S = 0.1         # OCR
CAST_GRADE_MEMORY_POLL_S = 0.01

ROD_REEQUIP_INTERVAL_S = 3600

TOTEM_EQUIP_SETTLE_S = 0.65
TOTEM_POST_USE_S = 0.70
TOTEM_SAME_CYCLE_RETRY_S = 12.0  # day totem re-fires while it is still day
TOTEM_INTER_FIRE_S = 8.0
ROD_EQUIP_SETTLE_S = 0.40

APPRAISE_SETTLE_S = 1.5         # after choosing the dialog's appraise option, lets the re-roll write back the fish's name and weight
APPRAISE_WRITEBACK_S = 3.0      # give up waiting for that write-back after this long

POTION_EQUIP_SETTLE_S = 0.80
POTION_POST_USE_S = 1.00

WORLD_CACHE_S = 5.0
CYCLE_STALE_FALLBACK_S = 300.0

AQUARIUM_CLIENT = (1680, 900)
AQUARIUM_BUTTONS = {"open": (944, 30), "use": (811, 664), "close": (1345, 109)}   # client px at AQUARIUM_CLIENT
AQUARIUM_INTERVAL_S = 65 * 60   # food lasts 60 min
AQUARIUM_RETRY_S = 30
AQUARIUM_BLIND_AFTER_MISSES = 3
AQUARIUM_RESIZE_SETTLE_S = 0.6
AQUARIUM_OPEN_SETTLE_S = 0.7
AQUARIUM_MOVE_SETTLE_S = 0.08
AQUARIUM_JIGGLE_DY_PX = 2
AQUARIUM_JIGGLE_GAP_S = 0.008
AQUARIUM_CLICK_PERIOD_S = 0.05
AQUARIUM_CLICK_HOLD_S = 0.012
AQUARIUM_SCROLL_STEP_S = 0.12
AQUARIUM_SCROLL_DOWN_S = 1.5
AQUARIUM_SCROLL_UP_S = 3.0      # Roblox drops wheel events; over-scroll back to the top


class Stopped(Exception):
    pass


class FocusLost(Exception):
    pass


class GuardedInput:
    """Refuses to move, press or click unless Sober is focused and the macro is
    running. Releases always go through."""

    def __init__(self, io, focus, stop_event):
        self.io = io
        self.focus = focus
        self.stop_event = stop_event

    def check(self):
        if self.stop_event.is_set():
            raise Stopped
        if self.focus.closed:
            raise hypr.WindowClosed
        if not self.focus.is_focused():
            raise FocusLost

    def move(self, x, y):
        self.check()
        self.io.move(x, y)

    def button(self, down):
        if down:
            self.check()
        self.io.button(down)

    def scroll(self, notches):
        self.check()
        self.io.scroll(notches)

    def press(self, key):
        self.check()
        self.io.press(key)

    def key(self, name, down):
        if down:
            self.check()
        self.io.key(name, down)

    def release_key(self, name):
        self.io.release_key(name)

    def release_all(self):
        self.io.release_all()


class Pointer:
    """ExoPanda's cursor idioms (fisch_util.lua): a 7-step glide from the last
    position, and a 50 px jiggle that makes Roblox re-read the hover state."""

    def __init__(self, io):
        self.io = io
        self.pos = None

    def warp(self, x, y):
        self.io.move(x, y)
        self.pos = (x, y)

    def glide(self, x, y):
        if self.pos is None:
            self.warp(x, y)
            return
        sx, sy = self.pos
        for i in range(1, 8):
            t = i / 7
            self.io.move(round(sx + (x - sx) * t), round(sy + (y - sy) * t))
            time.sleep(0.001)
        self.pos = (x, y)

    def use_click(self, x, y):
        """Totem/potion use: glide to (x, y), jiggle, click."""
        self.glide(x, y)
        time.sleep(0.08)
        self.glide(x, y + 50)
        time.sleep(0.02)
        self.glide(x, y)
        time.sleep(0.05)
        self.io.button(True)
        time.sleep(0.06)
        self.io.button(False)

    def nudge_click(self, x, y, hold_s=AQUARIUM_CLICK_HOLD_S):
        """Warp, settle, nudge down and back so the new position registers, click."""
        self.warp(x, y)
        time.sleep(AQUARIUM_MOVE_SETTLE_S)
        self.io.move(x, y + AQUARIUM_JIGGLE_DY_PX)
        time.sleep(AQUARIUM_JIGGLE_GAP_S)
        self.io.move(x, y)
        time.sleep(AQUARIUM_JIGGLE_GAP_S)
        self.io.button(True)
        time.sleep(hold_s)
        self.io.button(False)


class Snapshots:
    """Debug evidence: the frame a decision was made on, marked up and
    written as JPEG by a background thread. Keeps the newest MAX_SNAPSHOTS."""

    def __init__(self, folder):
        self.folder = folder
        self.queue = queue.Queue(maxsize=16)
        threading.Thread(target=self._write_loop, daemon=True, name="snapshots").start()

    def save(self, name, frame, points=(), rects=()):
        try:
            self.queue.put_nowait((time.time(), name, frame, points, rects))
        except queue.Full:
            pass

    def _write_loop(self):
        import cv2
        self.folder.mkdir(parents=True, exist_ok=True)
        while True:
            stamp, name, frame, points, rects = self.queue.get()
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for x, y, w, h in rects:
                cv2.rectangle(image, (round(x), round(y)), (round(x + w), round(y + h)), (0, 220, 255), 2)
            for x, y in points:
                cv2.drawMarker(image, (round(x), round(y)), (0, 0, 255), cv2.MARKER_CROSS, 44, 3)
            image = cv2.resize(image, (image.shape[1] * 2 // 3, image.shape[0] * 2 // 3), interpolation=cv2.INTER_AREA)
            ms = int(stamp * 1000) % 1000
            path = self.folder / f"{time.strftime('%H%M%S', time.localtime(stamp))}.{ms:03d}-{name}.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            for old in sorted(self.folder.glob("*.jpg"))[:-MAX_SNAPSHOTS]:
                old.unlink(missing_ok=True)


class ReelTelemetry:
    """How well the detector saw one reel, for the log line at its end. With
    the game's memory it also follows the catch progress: a reel that never
    lost any is a perfect catch."""

    def __init__(self, start):
        self.start = start
        self.frames = self.bar = self.fish = 0
        self.both = [0, 0]      # frames with both boxes: while the minigame ignores input, after
        self.inside = [0, 0]
        self.bar_conf = self.fish_conf = 0.0
        self.last_frame_t = None
        self.last_progress = None
        self.progress_lost = False

    def progress(self, value):
        if self.last_progress is not None and value < self.last_progress - PROGRESS_EPS:
            self.progress_lost = True
        self.last_progress = value

    def add(self, boxes):
        if boxes.frame_t == self.last_frame_t:
            return
        self.last_frame_t = boxes.frame_t
        self.frames += 1
        bar, fish = boxes.bar, boxes.fish
        if bar is not None:
            self.bar += 1
            self.bar_conf += bar.conf
        if fish is not None:
            self.fish += 1
            self.fish_conf += fish.conf
        if bar is not None and fish is not None:
            k = int(boxes.frame_t - self.start >= REEL_LOCK_S)
            self.both[k] += 1
            self.inside[k] += fish.x <= bar.x + bar.w and bar.x <= fish.x + fish.w   # progress grows on any overlap

    def summary(self):
        if not self.frames:
            return "no detector frames"

        def seen(name, n, conf):
            return f"{name} in {n / self.frames:.0%} (avg {conf / n:.2f})" if n else f"{name} never seen"

        text = f"{self.frames} frames, {seen('bar', self.bar, self.bar_conf)}, {seen('fish', self.fish, self.fish_conf)}"
        if self.last_progress is not None:
            text += f", progress {self.last_progress:.0%}" + (" with losses" if self.progress_lost else " never lost")
        if sum(self.both):
            text += f", fish inside the bar {sum(self.inside) / sum(self.both):.0%}"
        if all(self.both):
            text += (f" (first {REEL_LOCK_S:g} s {self.inside[0] / self.both[0]:.0%}, "
                     f"then {self.inside[1] / self.both[1]:.0%})")
        return text


@dataclass
class Stats:
    started: float = field(default_factory=time.time)
    casts: int = 0
    reels: int = 0
    shake_clicks: int = 0
    shake_timeouts: int = 0
    totems: int = 0
    appraisals: int = 0
    cast_grades: dict = field(default_factory=dict)     # grade -> casts that got it
    potion_rounds: int = 0
    aquarium_sweeps: int = 0
    rejoins: int = 0


class Shaker:
    """One click per shake button, after fisch_shaker.lua's vision path. A
    detection within SHAKE_SAME_BUTTON_PX of the last click is that same
    button unless a frame without it came in between: the game can respawn
    the next button in the very same spot, and that one is clicked at once.
    A button that never left is clicked again after SHAKE_REARM_S (the click
    missed); one that sits through SHAKE_DEAD_AFTER_CLICKS clicks is not a
    button, and since a click beside the button cancels the cast in Fisch,
    that spot is left alone for the rest of the cast. The watchdog recasts
    after 16 s without a successful click, or 30 s while no button has come
    up yet."""

    def __init__(self, io, pointer, log):
        self.io = io
        self.pointer = pointer
        self.log = log
        self.reset()

    def reset(self):
        self.last_click = None
        self.last_click_t = -math.inf
        self.gone_since_click = True
        self.last_progress = None
        self.repeats = 0
        self.spots = []         # distinct button positions clicked this cast
        self.dead_spots = []    # layout px of SHAKE-like matches that ignored clicks

    def _note_spot(self, x, y):
        if all(math.dist((x, y), s) > SHAKE_SAME_BUTTON_PX for s in self.spots):
            self.spots.append((x, y))

    def _timed_out(self, now):
        if self.last_progress is None:
            self.last_progress = now
        limit = SHAKE_TIMEOUT_S if self.last_click is not None else SHAKE_FIRST_TIMEOUT_S
        if now - self.last_progress > limit:
            self.log(f"shake: nothing to click for {limit:.0f} s, recasting")
            self.reset()
            return True
        return False

    def wait(self, now):
        """A frame without a button."""
        self.gone_since_click = True
        return "timeout" if self._timed_out(now) else "waiting"

    def click(self, x, y, suppress, now, trusted=False):
        """`trusted`: the position comes from the game's memory, so it is the
        button even if it ignores clicks for a while."""
        if self._timed_out(now):
            return "timeout"
        if suppress or any(math.dist((x, y), d) <= SHAKE_SAME_BUTTON_PX for d in self.dead_spots):
            return "waiting"
        same = (self.last_click is not None and not self.gone_since_click
                and math.dist((x, y), self.last_click) <= SHAKE_SAME_BUTTON_PX)
        if same and now - self.last_click_t < SHAKE_REARM_S:
            return "waiting"
        self.repeats = self.repeats + 1 if same else 0
        if self.repeats >= SHAKE_DEAD_AFTER_CLICKS and not trusted:
            self.dead_spots.append((x, y))
            self.repeats = 0
            self.log(f"shake: ignoring a SHAKE-like spot at {x:.0f},{y:.0f} that did not react to clicks")
            return "waiting"
        # Warp, nudge down and back, then click: Roblox can miss a same-frame
        # warp + click and click at the old cursor position.
        x, y = round(x), round(y)
        self.pointer.warp(x, y)
        time.sleep(SHAKE_JIGGLE_GAP_S)
        self.io.move(x, y + SHAKE_JIGGLE_DY_PX)
        time.sleep(SHAKE_JIGGLE_GAP_S)
        self.io.move(x, y)
        time.sleep(SHAKE_SETTLE_S)
        self.io.button(True)
        time.sleep(SHAKE_HOLD_S)
        self.io.button(False)
        self.last_click, self.last_click_t = (x, y), time.monotonic()
        self.gone_since_click = False
        self.last_progress = self.last_click_t
        self._note_spot(x, y)
        return "clicked"


class WorldState:
    """Day/night cycle and weather event read from the HUD, cached 5 s
    (fisch_weather.lua). When a read fails, cycle_with_fallback() serves the
    last good cycle for up to 5 minutes."""

    def __init__(self, grab_window, cycle_detector, event_detector):
        self.grab_window = grab_window
        self.cycle_detector = cycle_detector
        self.event_detector = event_detector
        self._cycle = self._good_cycle = self._weather = None
        self._cycle_at = self._good_at = self._weather_at = -math.inf

    def invalidate(self):
        self._cycle_at = self._weather_at = -math.inf

    def cycle(self, now):
        if now - self._cycle_at >= WORLD_CACHE_S:
            self._cycle = self.cycle_detector.detect(self.grab_window())
            self._cycle_at = now
            if self._cycle is not None:
                self._good_cycle, self._good_at = self._cycle, now
        return self._cycle

    def cycle_with_fallback(self, now):
        cycle = self.cycle(now)
        if cycle is None and now - self._good_at < CYCLE_STALE_FALLBACK_S:
            return self._good_cycle
        return cycle

    def weather(self, now):
        if now - self._weather_at >= WORLD_CACHE_S:
            self._weather = self.event_detector.detect(self.grab_window())
            self._weather_at = now
        return self._weather


def _read_float(path):
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return 0.0


def _write_float(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value:.0f}\n")


class Macro:
    def __init__(self, cfg, win, io, focus, bridge, vision, memory, state_dir, log, status, notify, stop_event, stats):
        self.cfg = cfg
        self.memory = memory                # robloxmem.MemoryLink
        self.win = win
        self.raw_io = io
        self.io = GuardedInput(io, focus, stop_event)
        self.focus = focus
        self.bridge = bridge
        self.reeler = Reeler(self.io, cfg.reel, state_dir / "learned_gains.json", log)
        self.vision = vision
        self.log = log
        self.status = status
        self.notify = notify                # Discord message
        self.stop_event = stop_event
        self.stats = stats
        self.pointer = Pointer(self.io)
        self.shaker = Shaker(self.io, self.pointer, log)
        self.world = WorldState(self.grab_window, vision.cycle, vision.weather_event)
        self.state_dir = state_dir
        self.snapshots = None
        self.aquarium_timer = state_dir / "aquarium_next_run"
        self.potion_timer = state_dir / "potions_last_use"
        self.aquarium_misses = 0
        self.window_checked = 0.0

        self.phase = "cast"
        self.reeling = False
        self.reel_started = -math.inf
        self.reel_telemetry = ReelTelemetry(self.reel_started)
        self.cast_shake_clicks = 0
        self.cast_t = -math.inf
        self.last_bar_seen = -math.inf
        self.last_fish_seen = -math.inf
        self.consec_fish = 0
        self.shake_hint_frames = 0
        self.waiting_for_bite = False
        self.recast_anchor = None
        self.last_activity = time.monotonic()
        self.last_reel_activity = time.monotonic()
        self.prompt_checked = -math.inf
        self.ui = None                      # FischUI while the game's memory is readable, per step
        self.memory_key = None
        self.memory_obs_t = -math.inf
        self.last_equip = -math.inf
        self.totem_fired_cycle = None
        self.totem_last_fire = -math.inf
        self.totem_next_fire = -math.inf
        self.totem_depleted = set()     # cycles whose totem ran out this run; that half stays off
        self.appraise_sig = None        # the appraise settings the counters below belong to
        self.appraise_attempts = 0
        self.appraise_spent = 0
        self.appraise_over = None       # why appraising stopped ("goal met: ..."), once it has

    # -- helpers -------------------------------------------------------------

    @property
    def rect(self):
        return self.win.rect

    def grab_window(self):
        return screen.grab(*self.rect)

    def sleep(self, seconds):
        if self.stop_event.wait(seconds):
            raise Stopped

    def _refresh_window(self, now, force=False):
        if not force and now - self.window_checked < WINDOW_REFRESH_S:
            return
        self.window_checked = now
        self.win = hypr.refresh(self.win)

    def _mark_activity(self):
        self.last_activity = time.monotonic()

    def _snap(self, name, frame=None, points=(), with_boxes=False):
        """Debug snapshot of what the macro is acting on, when enabled."""
        if not self.cfg.general.debug_snapshots:
            return
        if self.snapshots is None:
            self.snapshots = Snapshots(self.state_dir / "debug" / time.strftime("%Y%m%d-%H%M%S"))
        if frame is None:
            frame = self.grab_window()
        rects = []
        if with_boxes:
            ox, oy = self.win.x, self.win.y
            rx, ry, rw, rh = self.bridge.region.rect(self.rect)
            rects.append((rx - ox, ry - oy, rw, rh))
            boxes = self.bridge.latest
            rects += [(b.x - ox, b.y - oy, b.w, b.h) for b in (boxes.bar, boxes.fish) if b is not None]
        self.snapshots.save(name, frame, points, rects)

    def _equip_rod(self, full):
        presses = (OTHER_SLOT, ROD_SLOT, OTHER_SLOT, ROD_SLOT) if full else (OTHER_SLOT, ROD_SLOT)
        for key in presses:
            self.io.press(key)
            self.sleep(EQUIP_STEP_S)
        self.last_equip = time.monotonic()

    def _end_reel(self, now, why):
        self._snap("reel-end", with_boxes=True)
        self.reeling = False
        self.reeler.reset(now)
        self.shaker.reset()
        self.cast_shake_clicks = 0      # counted together with shaker.spots
        self.waiting_for_bite = False
        self.recast_anchor = None
        self.phase = "cast"
        self.last_reel_activity = now
        duration = now - self.reel_started
        if duration >= MIN_COUNTED_REEL_S:
            self.stats.reels += 1
        self.log(f"reel over after {duration:.1f} s ({why}): {self.reel_telemetry.summary()}; "
                 f"reels so far: {self.stats.reels}")

    def _start_reel(self, now, how):
        self.reeling = True
        self.reel_started = now
        self.reel_telemetry = ReelTelemetry(now)
        self.phase = "reel"
        since_cast = f" in {now - self.cast_t:.1f} s" if self.cast_t > 0 else ""
        # The button jumps after every shake the game accepts, so many clicks
        # on few positions means the clicks are not landing.
        self.log(f"reel: {how} after {self.cast_shake_clicks} shake click(s) on "
                 f"{len(self.shaker.spots)} button position(s){since_cast}")
        self._snap("reel-start", with_boxes=True)

    # -- main loop -------------------------------------------------------------

    def run(self):
        hypr.focus(self.win)
        self.sleep(0.3)
        equipped = paused = walk_pending = False
        lost = None
        try:
            while True:
                try:
                    if lost:
                        self._recover(lost)
                        lost, equipped, walk_pending = None, False, True
                    if paused:
                        self._pause_until_focused()
                        paused = False
                    if walk_pending:
                        walk_pending = False     # a walk cut short by a focus change is not restarted
                        self._walk_route()
                    if not equipped:
                        self._equip_rod(full=True)
                        equipped = True
                    self._step()
                except FocusLost:
                    paused = True
                except (hypr.WindowClosed, recovery.GameLost) as e:
                    if not self.cfg.recovery.enabled:
                        raise
                    lost, paused = str(e), False
        except Stopped:
            pass
        finally:
            self.reeler.release()
            self.raw_io.release_all()
            self.bridge.phase = "idle"
            if self.reeler.learner.rods:
                try:
                    self.reeler.learner.save()
                except OSError as e:
                    self.log(f"reel learner: could not save gains: {e}")

    # -- reading the game's memory ------------------------------------------------

    def _memory_boxes(self, now):
        """The reel bar and fish from memory as detector boxes. Their capture time
        is when the values last changed: the game moves them once per frame."""
        reel = self.ui.reel()
        if reel is None:
            return Boxes(frame_t=now)
        (bx, by, bw, bh), (fx, fy, fw, fh), progress = reel
        if (bx, fx) != self.memory_key:
            self.memory_key, self.memory_obs_t = (bx, fx), now
        if progress is not None:
            self.reel_telemetry.progress(progress)
        ox, oy = self.win.x, self.win.y
        return Boxes(Box(ox + bx, oy + by, bw, bh, 1.0), Box(ox + fx, oy + fy, fw, fh, 1.0),
                     self.memory_obs_t, self.memory_obs_t, now)

    def _pause_until_focused(self):
        self.reeler.release()
        self.raw_io.release_all()
        self.bridge.phase = "idle"
        self.status("paused", "Sober is not focused")
        self.log("paused: Sober lost focus")
        while not self.focus.is_focused():
            if self.focus.closed:
                raise hypr.WindowClosed
            self.sleep(0.1)
        self.log("resumed")
        self.pointer.pos = None
        self.sleep(0.2)

    # -- recovery --------------------------------------------------------------

    def _check_error_prompt(self, now):
        """Fishing stalled: look for Roblox's error prompt (OCR, ~0.2 s)."""
        if now - self.last_reel_activity < PROMPT_CHECK_AFTER_S or now - self.prompt_checked < PROMPT_CHECK_EVERY_S:
            return
        self.prompt_checked = now
        if recovery.error_prompt(self.grab_window()):
            raise recovery.GameLost("Roblox showed its error prompt")

    def _recover(self, why):
        why = why.rstrip(".")
        self.reeler.release()
        self.raw_io.release_all()
        self.bridge.phase = "idle"
        self.stats.rejoins += 1
        self.log(f"recovery: {why}; restarting Sober into Fisch")
        self.notify(f"Lost the game ({why}), rejoining Fisch.")
        match = self.cfg.general.window
        for attempt in range(1, recovery.ATTEMPTS + 1):
            self.status("recovery", f"rejoining Fisch, attempt {attempt}")
            recovery.restart_sober(match, self.sleep)
            win = recovery.wait_for_fisch(match, self.vision.cycle, self.sleep)
            if win is not None:
                break
            self.log(f"recovery: Fisch did not load (attempt {attempt} of {recovery.ATTEMPTS})")
        else:
            raise hypr.WindowError(f"Could not get back into Fisch after {recovery.ATTEMPTS} attempts.")
        self.win = win
        self.focus.retarget(win.address)
        hypr.focus(win)
        self.sleep(recovery.SPAWN_SETTLE_S)
        self.phase = "cast"
        self.reeling = False
        self.waiting_for_bite = False
        self.recast_anchor = None
        self.consec_fish = self.shake_hint_frames = 0
        self.shaker.reset()
        self.pointer.pos = None
        self.last_reel_activity = time.monotonic()
        self.log("recovery: back in Fisch")
        self.notify("Back in Fisch.")

    def _walk_route(self):
        route = recovery.load_route(self.state_dir / recovery.ROUTE_FILE)
        if route is None:
            self.log("recovery: no route recorded, fishing from the spawn point")
            return
        self.status("recovery", "walking to the fishing spot")
        self.log(f"recovery: walking the recorded route ({route[-1][0]:.1f} s)")
        recovery.walk(self.io, route, self.sleep)

    def _step(self):
        now = time.monotonic()
        self._refresh_window(now)
        self.ui = self.memory.get() if self.cfg.general.memory else None
        boxes = self._memory_boxes(now) if self.ui else self.bridge.latest
        fresh = now - boxes.frame_t < STALE_S
        if fresh and boxes.bar is not None:
            self.last_bar_seen = now
        if fresh and boxes.fish is not None:
            self.last_fish_seen = now
        both_now = fresh and boxes.bar is not None and boxes.fish is not None

        # Reel entry: STRICT to start (a debounced or confident fish box, or both
        # classes at once). Staying needs both classes within the holdover
        # window: ExoPanda stayed on any single box, but in Fisch a lone box is
        # far more often on-screen text than a reel that lost one class (real
        # reels show both in ~96% of frames).
        fish_now = fresh and boxes.fish is not None and boxes.fish.conf >= ENTRY_MIN_FISH_CONF
        enter_strict = False
        if fish_now:
            self.consec_fish += 1
            dual = boxes.bar is not None and boxes.bar.conf >= DUAL_ENTRY_MIN_CONF
            enter_strict = dual or self.consec_fish >= ENTRY_DEBOUNCE_TICKS or boxes.fish.conf >= ENTRY_FAST_FISH_CONF
        else:
            self.consec_fish = 0
        stay_reel = (self.reeling and now - self.last_bar_seen < HOLDOVER_S
                     and now - self.last_fish_seen < HOLDOVER_S)
        reel_live = stay_reel or enter_strict

        if self.phase == "reel" or reel_live:
            self.bridge.phase = "idle" if self.ui else "reel"
            if not reel_live:
                self._end_reel(now, "minigame closed")
                return
            self._mark_activity()
            if not self.reeling:
                self._start_reel(now, "minigame detected")
            self.reel_telemetry.add(boxes)
            self.status("reel", self._reel_detail(boxes))
            if self.reeler.tick(boxes, now) == "ended":
                self._end_reel(now, "bar and fish stopped moving")
            self.sleep(REEL_TICK_S)
        elif self.phase == "shake":
            self.bridge.phase = "idle" if self.ui else "shake"
            self._check_error_prompt(now)
            reel_hint = both_now and boxes.bar.conf >= DUAL_ENTRY_MIN_CONF
            self._shake_step(now, fish_now or both_now, reel_hint)
            self.sleep(SHAKE_TICK_S)
        else:
            self.bridge.phase = "idle"
            self.reeler.release()
            self._check_error_prompt(now)
            self._idle_step(now)
            silent = time.monotonic() - self.last_activity
            fast = self.waiting_for_bite or silent < SILENCE_RAMP_S
            self.sleep(IDLE_TICK_FAST_S if fast else IDLE_TICK_SLOW_S)

    @staticmethod
    def _reel_detail(boxes):
        parts = []
        if boxes.bar is not None:
            parts.append(f"bar {boxes.bar.conf:.2f}")
        if boxes.fish is not None:
            parts.append(f"fish {boxes.fish.conf:.2f}")
        return " ".join(parts) or "holding"

    def _shake_step(self, now, suppress, reel_hint):
        self.reeling = False
        # Both reel classes persisting while still in the shake phase means the
        # strict entry under-detected the reel: hand over instead of waiting
        # for the 16 s watchdog to recast mid-reel.
        self.shake_hint_frames = self.shake_hint_frames + 1 if reel_hint else 0
        if self.shake_hint_frames >= SHAKE_TO_REEL_HINT_TICKS:
            self.shake_hint_frames = 0
            self._start_reel(now, "entered from a persistent box during shake")
            return
        hit = frame = None
        trusted = self.ui is not None
        if self.cfg.fishing.auto_shake and trusted:
            r = self.ui.shake_button()
            hit = (r[0] + r[2] / 2, r[1] + r[3] / 2) if r else None
        elif self.cfg.fishing.auto_shake:
            frame = self.grab_window()
            dead = [(x - self.win.x, y - self.win.y) for x, y in self.shaker.dead_spots]
            hit = self.vision.shake.find(frame, exclude=dead)
        if hit is not None:
            self._mark_activity()
            result = self.shaker.click(self.win.x + hit[0], self.win.y + hit[1], suppress, now, trusted)
            if result == "clicked":
                self.stats.shake_clicks += 1
                self.cast_shake_clicks += 1
                self.status("shake", "clicked (memory)" if trusted else f"clicked (match {self.vision.shake.last_score:.2f})")
                self._snap("shake", frame, [hit])
        else:
            result = self.shaker.wait(now)
            if result == "waiting":
                self.status("shake", "waiting for the button")
        if result == "timeout":
            self._snap("shake-timeout", frame)
            self.stats.shake_timeouts += 1
            self.waiting_for_bite = False
            self.recast_anchor = None
            self.phase = "cast"

    def _idle_step(self, now):
        if self.cfg.appraise.enabled:
            self._appraise_step()
            self._mark_activity()
            return
        if self.cfg.aquarium.enabled and time.time() >= _read_float(self.aquarium_timer):
            self._aquarium_sweep()
            self.shaker.reset()
            self._mark_activity()
        elif self._potions_due(now):
            self._use_potions()
            self._mark_activity()
        elif self._totem_enabled() and not self.waiting_for_bite and self._totem_tick() == "used":
            self._mark_activity()
        elif self.cfg.fishing.auto_cast and not self.waiting_for_bite:
            if self.recast_anchor is None:
                self.recast_anchor = now
            if now - self.recast_anchor < self.cfg.fishing.recast_delay_ms / 1000:
                self.status("cast", "waiting to recast")
                return
            if now - self.last_equip >= ROD_REEQUIP_INTERVAL_S:
                self._equip_rod(full=False)
            self._cast()
        else:
            self.status("idle", "auto cast is off" if not self.cfg.fishing.auto_cast else "waiting")

    def _cast(self):
        self.status("cast", "casting")
        self._snap("cast")
        x, y = self.win.center
        self.pointer.warp(x, y)
        target, ui = self.cfg.fishing.cast_grade, self.ui
        earlier = ui.grade_labels() if ui else None
        bar = None if ui else PowerBar(screen.grab(*PowerBar.crop(self.rect)))
        self.io.button(True)
        try:
            how = self._hold_for_cast(CAST_STOP_AT[target], ui, bar)
        finally:
            self.io.button(False)
        self.stats.casts += 1
        self.log(f"cast #{self.stats.casts}: aiming for {target.upper()}, {how}")
        threading.Thread(target=self._read_cast_grade, args=(self.stats.casts, target, self.rect, ui, earlier),
                         daemon=True, name="cast-grade").start()
        self.cast_t = time.monotonic()
        self.cast_shake_clicks = 0
        self.waiting_for_bite = True
        self.recast_anchor = None
        self.shake_hint_frames = 0
        self.shaker.reset()
        self.phase = "shake"

    def _hold_for_cast(self, stop_at, ui, bar):
        """Keep the button held until the fill will stop at `stop_at`, read
        from the game's memory or, without it, from the screen with `bar`, a
        PowerBar set up before the press. Returns how the release was timed.
        When no power bar can be read, the timed hold."""
        hold_s = self.cfg.fishing.cast_hold_ms / 1000
        start = time.monotonic()
        if ui is not None:
            read, source, lead, poll = ui.power_fill, "memory", CAST_MEMORY_LEAD_S, CAST_MEMORY_POLL_S
        else:
            timed_s = CAST_FILL_START_S + stop_at / CAST_TYPICAL_SPEED - GAME_FRAME_S / 2
            if timed_s < CAST_SCREEN_FIRST_S:
                self.sleep(timed_s)
                return f"held {timed_s:.2f} s, a throw too short to see on the screen"
            crop = PowerBar.crop(self.rect)
            read, source, lead, poll = (lambda: bar.fill(screen.grab(*crop))), "screen", CAST_SCREEN_LEAD_S, 0.0
        horizon = CAST_SCREEN_READ_S if source == "screen" else 0.0
        last = rise = None              # rise: the fill and time the current rise was first read at
        while True:
            if self.stop_event.is_set():
                raise Stopped
            fill = read()
            now = time.monotonic()
            t = now - start
            if t > CAST_MAX_S:
                return f"held {t:.2f} s, the fill never rose to {stop_at:.0%} ({source})"
            if fill is None:
                if last is None and t >= max(CAST_BAR_WAIT_S, hold_s):
                    return f"held {t:.2f} s, no power bar ({source})"
            elif fill != last:
                # Memory keeps the last cast's bar until the new one shows; the screen shows only the new one.
                if last is not None and last < fill:
                    speed = min((fill - rise[0]) / (now - rise[1]), CAST_MAX_SPEED)
                else:
                    speed = CAST_TYPICAL_SPEED if last is None and source == "screen" else None
                    rise = (fill, now)
                wait = (stop_at - fill) / speed - lead if speed else None
                if wait is not None and wait <= horizon:
                    if wait > 0:
                        self.sleep(wait)
                    return (f"released at {t + max(wait, 0):.2f} s with the bar {fill:.0%} full "
                            f"rising {speed:.1f}/s ({source})")
                last = fill
            if poll:
                time.sleep(poll)

    def _read_cast_grade(self, number, target, rect, ui, earlier):
        """Log the grade Fisch writes over the character after a cast, read
        from the game's memory or, without it, with OCR."""
        crop = cast_text_crop(rect)
        end = time.monotonic() + CAST_GRADE_READ_S
        while time.monotonic() < end and not self.stop_event.is_set():
            grade = ui.cast_grade(earlier) if ui else cast_grade(screen.grab(*crop))
            if grade is not None:
                self.stats.cast_grades[grade] = self.stats.cast_grades.get(grade, 0) + 1
                missed = "" if grade == target else f", aimed for {target.upper()}"
                self.log(f"cast #{number}: {grade.upper()}{missed}")
                return
            time.sleep(CAST_GRADE_MEMORY_POLL_S if ui else CAST_GRADE_POLL_S)

    # -- totem (fisch_totem.lua) ---------------------------------------------------

    def _totem_enabled(self):
        t = self.cfg.totem
        return t.enabled and bool(t.day_name.strip() or t.night_name.strip() or t.day_slot or t.night_slot)

    def _totem_target(self, cycle):
        """("fire", label, slot) for this cycle's totem, ("gone", name, None)
        when the hotbar can be read and holds none, ("wait", name, None) when
        the hotbar cannot be read now and no fallback slot is set. The totem
        is found by name in the hotbar; the slot settings are the fallback
        for when the game's memory is not readable."""
        t = self.cfg.totem
        name = (t.day_name if cycle == "day" else t.night_name).strip()
        fallback = t.day_slot if cycle == "day" else t.night_slot
        if name and self.ui is not None:
            items = self.ui.hotbar()
            if items is None:
                return "wait", name, None
            match = next(((s, n, c) for s, (n, c, _w) in sorted(items.items()) if name.lower() in n.lower()), None)
            if match is None:
                return "gone", name, None
            slot, item, count = match
            return "fire", item + (f" x{count}" if count else ""), slot
        if 1 <= fallback <= 9:
            return "fire", name or "totem", fallback
        return "wait", name, None

    def _totem_tick(self):
        now = time.monotonic()
        cycle = self.world.cycle_with_fallback(now)
        if cycle is None:
            self.status("totem", "day/night unknown")
            return "wait"
        if cycle in self.totem_depleted:
            return "wait"
        kind, label, slot = self._totem_target(cycle)
        if kind == "gone" and self._totem_target(cycle)[0] == "gone":
            # Confirmed twice, so a hotbar being rebuilt cannot switch it off.
            self.totem_depleted.add(cycle)
            self.log(f"totem: no '{label}' left in the hotbar, the {cycle} totem is off for this run")
            self.notify(f"Out of totems: no '{label}' in the hotbar, the {cycle} totem is off; fishing continues.")
            return "wait"
        if kind != "fire":
            return "wait"
        if cycle == self.totem_fired_cycle and (cycle == "night" or now < self.totem_next_fire):
            return "wait"   # night totems once per night; day totems retry until night shows
        if now - self.totem_last_fire < TOTEM_INTER_FIRE_S:
            return "wait"
        # Cached reads can lag a cycle flip; fire only on a fresh agreeing read.
        self.world.invalidate()
        if self.world.cycle(now) != cycle:
            return "wait"
        self.status("totem", f"using the {cycle} totem (slot {slot})")
        self.log(f"totem: {cycle}, using {label} (slot {slot})")
        self.io.press(str(slot))
        self.sleep(TOTEM_EQUIP_SETTLE_S)
        self.pointer.use_click(*self.win.center)
        self.sleep(TOTEM_POST_USE_S)
        self.io.press(ROD_SLOT)
        self.sleep(ROD_EQUIP_SETTLE_S)
        self.stats.totems += 1
        self.totem_fired_cycle = cycle
        self.world.invalidate()
        self.totem_last_fire = time.monotonic()
        self.totem_next_fire = self.totem_last_fire + TOTEM_SAME_CYCLE_RETRY_S
        self.recast_anchor = None
        return "used"

    # -- appraise (the Appraiser NPC's dialog) --------------------------------------

    def _appraise_fish(self):
        """(name, weight in kg or None) of the fish in the appraise slot, or None."""
        items = self.ui.hotbar() if self.ui else None
        entry = items.get(self.cfg.appraise.slot) if items else None
        return (entry[0], entry[2]) if entry else None

    def _appraise_goal(self, name, kg):
        a = self.cfg.appraise
        mut = a.mutation.strip().lower()
        if mut and mut not in name.lower():
            return False
        return a.min_weight_kg <= 0 or (kg is not None and kg >= a.min_weight_kg)

    def _appraise_stop(self, why):
        self.appraise_over = why
        self.log(f"appraise: {why}")
        self.notify(f"Appraise stopped: {why}.")

    def _appraise_step(self):
        """One appraisal: click the dialog's appraise option (found on screen
        with OCR — the dialog is a BillboardGui on the NPC, so its place
        follows the camera), then read the re-rolled name and weight back
        from the hotbar. The player
        walks to the Appraiser and opens the dialog with the fish in hand;
        while appraise is enabled it owns the idle step, so nothing is cast.
        Stops for good when the goal is met or the attempts run out — every
        attempt costs C$, so it never rolls without a goal or a cap."""
        a = self.cfg.appraise
        sig = (a.slot, a.mutation.strip().lower(), a.min_weight_kg, a.max_attempts)
        if sig != self.appraise_sig:    # retargeting mid-run starts a fresh count
            self.appraise_sig = sig
            self.appraise_attempts = self.appraise_spent = 0
            self.appraise_over = None
        if self.appraise_over is not None:
            self.status("appraise", self.appraise_over)
            return
        if self.ui is None:
            self.status("appraise", "needs the game's memory")
            return
        if not a.mutation.strip() and a.min_weight_kg <= 0:
            self.status("appraise", "set a mutation or a minimum weight first")
            return
        fish = self._appraise_fish()
        if fish is None:
            self.status("appraise", f"no fish readable in hotbar slot {a.slot}")
            return
        name, kg = fish
        if self._appraise_goal(name, kg):
            self._appraise_stop(f"goal met: {name}" + (f", {kg:g} kg" if kg is not None else "")
                                + f" after {self.appraise_attempts} attempt(s), {self.appraise_spent} C$ spent")
            return
        if self.appraise_attempts >= a.max_attempts:
            self._appraise_stop(f"{a.max_attempts} attempts without the goal (last: {name}"
                                + (f", {kg:g} kg" if kg is not None else "") + f"), {self.appraise_spent} C$ spent")
            return
        price = self.ui.appraise_option()
        if price is None:
            self.status("appraise", "open the appraiser dialog and hold the fish")
            return
        row = None
        for _ in range(2):      # a missed OCR read costs a second look, never a blind click
            row = appraise_row(self.grab_window())
            if row is not None:
                break
        if row is None:
            self.status("appraise", "the dialog's appraise option is not readable on screen")
            return
        self.pointer.use_click(self.win.x + row[0], self.win.y + row[1])
        self.appraise_attempts += 1
        self.stats.appraisals += 1
        self.appraise_spent += price
        self.sleep(APPRAISE_SETTLE_S)
        deadline = time.monotonic() + APPRAISE_WRITEBACK_S
        while self._appraise_fish() in (fish, None) and time.monotonic() < deadline:
            self.sleep(0.1)
        name, kg = self._appraise_fish() or fish
        weight = f"{kg:g} kg" if kg is not None else "weight unreadable"
        self.log(f"appraise #{self.appraise_attempts}: {name}, {weight} "
                 f"(paid {price} C$, {self.appraise_spent} C$ in all)")
        self.status("appraise", f"attempt {self.appraise_attempts}: {name}, {weight}")

    # -- potions (fisch_potions.lua) -------------------------------------------------

    def _potion_slots(self):
        return [s for s in self.cfg.potions.slots if 1 <= s <= 9]

    def _potions_due(self, now):
        p = self.cfg.potions
        if not p.enabled or not self._potion_slots():
            return False
        if time.time() - _read_float(self.potion_timer) < p.interval_min * 60:
            return False
        if p.only_during_weather_event and self.world.weather(now) in (None, "clear"):
            return False
        return True

    def _use_potions(self):
        slots = self._potion_slots()
        self.log(f"potions: using hotbar slot(s) {', '.join(map(str, slots))}")
        for slot in slots:
            self.status("potions", f"slot {slot}")
            self.io.press(str(slot))
            self.sleep(POTION_EQUIP_SETTLE_S)
            self.pointer.use_click(*self.win.center)
            self.sleep(POTION_POST_USE_S)
        self.io.press(ROD_SLOT)
        self.sleep(ROD_EQUIP_SETTLE_S)
        _write_float(self.potion_timer, time.time())
        self.stats.potion_rounds += 1
        self.recast_anchor = None

    # -- aquarium (fisch_aquarium.lua) -----------------------------------------------

    def _aquarium_sweep(self):
        self.status("aquarium", "feeding")
        self.log("aquarium: hourly feed")
        before = hypr.refresh(self.win)
        try:
            if not self._aquarium_resize(before):
                _write_float(self.aquarium_timer, time.time() + AQUARIUM_RETRY_S * 10)
                return
            cx, cy = self.win.x, self.win.y

            def at(name):
                bx, by = AQUARIUM_BUTTONS[name]
                return cx + bx, cy + by

            self.pointer.nudge_click(*at("open"))
            self.sleep(AQUARIUM_OPEN_SETTLE_S)
            if not self.vision.aquarium.is_open(self.grab_window()):
                self.aquarium_misses += 1
                score = self.vision.aquarium.last_score
                if self.aquarium_misses < AQUARIUM_BLIND_AFTER_MISSES:
                    self.log(f"aquarium: panel not recognised (score {score:.2f}), retrying in {AQUARIUM_RETRY_S} s")
                    self.pointer.nudge_click(*at("close"))
                    _write_float(self.aquarium_timer, time.time() + AQUARIUM_RETRY_S)
                    return
                self.log(f"aquarium: panel still not recognised (score {score:.2f}), feeding blind like ExoPanda")
            self.aquarium_misses = 0
            use = at("use")
            self.pointer.warp(*use)
            self.sleep(AQUARIUM_MOVE_SETTLE_S)
            self.io.move(use[0], use[1] + AQUARIUM_JIGGLE_DY_PX)
            time.sleep(AQUARIUM_JIGGLE_GAP_S)
            self.io.move(*use)
            self._spam_scroll(-1, AQUARIUM_SCROLL_DOWN_S)
            self._spam_scroll(1, AQUARIUM_SCROLL_UP_S)
            self.pointer.nudge_click(*at("close"))
            _write_float(self.aquarium_timer, time.time() + AQUARIUM_INTERVAL_S)
            self.stats.aquarium_sweeps += 1
        finally:
            self._aquarium_restore(before)

    def _spam_scroll(self, direction, duration_s):
        start = time.monotonic()
        last_scroll = -math.inf
        while time.monotonic() - start < duration_s:
            self.io.button(True)
            time.sleep(AQUARIUM_CLICK_HOLD_S)
            self.io.button(False)
            if time.monotonic() - last_scroll >= AQUARIUM_SCROLL_STEP_S:
                self.io.scroll(direction)
                last_scroll = time.monotonic()
            self.sleep(AQUARIUM_CLICK_PERIOD_S - AQUARIUM_CLICK_HOLD_S)

    def _aquarium_resize(self, before):
        """Float the window at exactly 1680x900 so the panel's buttons land on
        the mapped pixels. False (and logged) if Hyprland would not comply."""
        w, h = AQUARIUM_CLIENT
        mx, my, mw, mh = hypr.monitor_rect(before.monitor)
        if mw < w or mh < h:
            self.log(f"aquarium: the monitor is smaller than {w}x{h}; feeding disabled for now")
            return False
        if before.fullscreen:
            hypr.set_fullscreen(before, False)
        if not before.floating:
            hypr.set_floating(before, True)
        hypr.resize_exact(before, w, h)
        hypr.move_exact(before, mx + (mw - w) // 2, my + (mh - h) // 2)
        self.sleep(AQUARIUM_RESIZE_SETTLE_S)
        self._refresh_window(time.monotonic(), force=True)
        if (self.win.width, self.win.height) != (w, h):
            self.log(f"aquarium: the window stayed {self.win.width}x{self.win.height} instead of {w}x{h}")
            return False
        return True

    def _aquarium_restore(self, before):
        try:
            current = hypr.refresh(self.win)
            if before.floating:
                hypr.resize_exact(current, before.width, before.height)
                hypr.move_exact(current, before.x, before.y)
            elif current.floating:
                hypr.set_floating(current, False)
            if before.fullscreen:
                hypr.set_fullscreen(current, True)
            time.sleep(AQUARIUM_RESIZE_SETTLE_S)
            self._refresh_window(time.monotonic(), force=True)
        except hypr.WindowError as e:
            self.log(f"aquarium: could not restore the window: {e}")


class MacroRunner(threading.Thread):
    """Runs one Macro session in the background so the hotkey stays responsive."""

    def __init__(self, macro, on_exit):
        super().__init__(daemon=True, name="macro")
        self.macro = macro
        self.on_exit = on_exit
        self.error = None

    def run(self):
        try:
            self.macro.run()
        except hypr.WindowError as e:
            self.error = str(e)
        except Exception as e:  # report instead of dying silently in a thread
            self.error = f"{type(e).__name__}: {e}"
            import traceback
            self.macro.log(traceback.format_exc())
        finally:
            self.on_exit(self)
