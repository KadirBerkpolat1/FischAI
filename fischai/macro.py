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
from .reeler import Reeler

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

# The rod lives in hotbar slot 1. Alternating 2 -> 1 presses always end with
# the rod in hand; pressing 1 alone would put an already-held rod away.
ROD_SLOT = "1"
OTHER_SLOT = "2"
EQUIP_STEP_S = 0.120
ROD_REEQUIP_INTERVAL_S = 3600

TOTEM_EQUIP_SETTLE_S = 0.65
TOTEM_POST_USE_S = 0.70
TOTEM_SAME_CYCLE_RETRY_S = 12.0  # day totem re-fires while it is still day
TOTEM_INTER_FIRE_S = 8.0
ROD_EQUIP_SETTLE_S = 0.40

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
    """How well the detector saw one reel, for the log line at its end."""

    def __init__(self):
        self.frames = self.bar = self.fish = self.both = self.inside = 0
        self.bar_conf = self.fish_conf = 0.0
        self.last_frame_t = None

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
            self.both += 1
            self.inside += bar.x <= fish.cx <= bar.x + bar.w

    def summary(self):
        if not self.frames:
            return "no detector frames"

        def seen(name, n, conf):
            return f"{name} in {n / self.frames:.0%} (avg {conf / n:.2f})" if n else f"{name} never seen"
        text = f"{self.frames} frames, {seen('bar', self.bar, self.bar_conf)}, {seen('fish', self.fish, self.fish_conf)}"
        if self.both:
            text += f", fish inside the bar {self.inside / self.both:.0%}"
        return text


@dataclass
class Stats:
    started: float = field(default_factory=time.time)
    casts: int = 0
    reels: int = 0
    shake_clicks: int = 0
    shake_timeouts: int = 0
    totems: int = 0
    potion_rounds: int = 0
    aquarium_sweeps: int = 0
    rejoins: int = 0


class Shaker:
    """One click per shake button, after fisch_shaker.lua's vision path. A
    detection within SHAKE_SAME_BUTTON_PX of the last click is that same
    button unless a frame without it came in between: the game can respawn
    the next button in the very same spot, and that one is clicked at once.
    A button that never left is clicked again after SHAKE_REARM_S (the click
    missed). The watchdog recasts after 16 s without a successful click, or
    30 s while no button has come up yet."""

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
        self.spots = []     # distinct button positions clicked this cast

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

    def click(self, x, y, suppress, now):
        if self._timed_out(now):
            return "timeout"
        if suppress:
            return "waiting"
        if (self.last_click is not None and not self.gone_since_click
                and math.dist((x, y), self.last_click) <= SHAKE_SAME_BUTTON_PX
                and now - self.last_click_t < SHAKE_REARM_S):
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
    def __init__(self, cfg, win, io, focus, bridge, vision, state_dir, log, status, notify, stop_event, stats):
        self.cfg = cfg
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
        self.reel_telemetry = ReelTelemetry()
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
        self.last_equip = -math.inf
        self.totem_fired_cycle = None
        self.totem_last_fire = -math.inf
        self.totem_next_fire = -math.inf

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
        win = hypr.refresh(self.win)
        if win.rect != self.win.rect:
            self.reeler.forget_walls()
        self.win = win

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
        self.reel_telemetry = ReelTelemetry()
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
            self.bridge.active = False
            if self.reeler.learner.rods:
                try:
                    self.reeler.learner.save()
                except OSError as e:
                    self.log(f"reel learner: could not save gains: {e}")

    def _pause_until_focused(self):
        self.reeler.release()
        self.raw_io.release_all()
        self.bridge.active = False
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
        self.bridge.active = False
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
        self.reeler.forget_walls()
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
        boxes = self.bridge.latest
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
            self.bridge.active = True
            if not reel_live:
                self._end_reel(now, "minigame closed")
                return
            self._mark_activity()
            if not self.reeling:
                self._start_reel(now, "minigame detected")
            self.reel_telemetry.add(boxes)
            self.status("reel", self._reel_detail(boxes))
            if self.reeler.tick(boxes, now) == "ended":
                self._end_reel(now, "bar stopped moving")
            self.sleep(REEL_TICK_S)
        elif self.phase == "shake":
            self.bridge.active = True
            self._check_error_prompt(now)
            reel_hint = both_now and boxes.bar.conf >= DUAL_ENTRY_MIN_CONF
            self._shake_step(now, fish_now or both_now, reel_hint)
            self.sleep(SHAKE_TICK_S)
        else:
            self.bridge.active = False
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
        if self.cfg.fishing.auto_shake:
            frame = self.grab_window()
            hit = self.vision.shake.find(frame)
        if hit is not None:
            self._mark_activity()
            result = self.shaker.click(self.win.x + hit[0], self.win.y + hit[1], suppress, now)
            if result == "clicked":
                self.stats.shake_clicks += 1
                self.cast_shake_clicks += 1
                self.status("shake", f"clicked (match {self.vision.shake.last_score:.2f})")
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
        self.io.button(True)
        try:
            self.sleep(self.cfg.fishing.cast_hold_ms / 1000)
        finally:
            self.io.button(False)
        self.stats.casts += 1
        self.log(f"cast #{self.stats.casts}")
        self.cast_t = time.monotonic()
        self.cast_shake_clicks = 0
        self.waiting_for_bite = True
        self.recast_anchor = None
        self.shake_hint_frames = 0
        self.shaker.reset()
        self.phase = "shake"

    # -- totem (fisch_totem.lua) ---------------------------------------------------

    def _totem_enabled(self):
        t = self.cfg.totem
        return t.enabled and (1 <= t.day_slot <= 9 or 1 <= t.night_slot <= 9)

    def _totem_tick(self):
        now = time.monotonic()
        cycle = self.world.cycle_with_fallback(now)
        if cycle is None:
            self.status("totem", "day/night unknown")
            return "wait"
        slot = self.cfg.totem.day_slot if cycle == "day" else self.cfg.totem.night_slot
        if not 1 <= slot <= 9:
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
        self.log(f"totem: {cycle}, using hotbar slot {slot}")
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
