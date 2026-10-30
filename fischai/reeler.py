"""
Reel minigame controller, ported from ExoPanda's AI-mode reeler
(fisch_reeler.lua). Each ~16 ms tick turns the latest catch-bar / fish-line
boxes into a left-button press or release:

    err_pred = err + v_hat * latency              latency-compensated error
    v_des    = clamp(-KP * err_pred, -VMAX, VMAX)  outer loop: desired err rate
    neutral  = 0.5 + integral(KI * err/100 * dt)   gravity compensation
    duty     = clamp(neutral + KV * (v_hat - v_des), 0, 1)

The duty becomes a first-order sigma-delta pulse train. When the target sits
within half a bar of a track wall the bar is pinned to that wall (hold for
the right one, release for the left) instead of oscillating against it.
v_hat only advances on genuinely new observations, never on holdover frames.

A perturb-and-observe learner tunes KP/KV per rod (keyed by bar width) from
each reel's cost J = mean(err^2) + 300 * flips/s + 20 * max overshoot, and
persists the table. The Requiem rod, which fails the catch when mouse events
come closer than ~300 ms, gets ExoPanda's separate gated bang-bang law.
"""

import json
import math
from dataclasses import asdict, dataclass

TICK_S = 0.016

STALE_S = 0.250               # AI_STALE_MS
HOLDOVER_S = 0.500            # AI_HOLDOVER_MS
MIN_CONF = 0.25               # AI_MIN_CONF_RELEEL
STUCK_TICKS = 120             # AI_STUCK_FRAMES: ~2 s of a static bar ends the reel
STUCK_DELTA_PX = 2.0

CAS_KP = 6.0                  # px/s of desired closing speed per px of error
CAS_KV = 0.0015               # duty per px/s of velocity error
CAS_VMAX = 700.0
CAS_KI = 0.12                 # neutral-duty integral rate (duty/s per 100 px err)
CAS_NEUTRAL_CLAMP = 0.25
VEL_EMA_ALPHA = 0.55
VEL_GAP_RESET_S = 0.250
ACTUATION_S = 0.030           # game input + render delay on top of the observation age
LATENCY_RANGE_S = (0.020, 0.200)
HOLD_STATION_PX = 40          # AI_HOVER_BAND_PX: a slow bar this close is holding, not stuck
HOLD_STATION_VEL = 60

WALL_MARGIN_PX = 6.0
WALL_HYST_PX = 24.0
WALL_MIN_TRACK_PX = 120.0

SMART_TARGET_OVERLAP = 0.15   # share of the bar kept over the fish when chasing an override

REQUIEM_MIN_INTERVAL_S = 0.300
REQUIEM_DEADBAND_PX = 6
REQUIEM_FALL_THRESH_PX = 3
REQUIEM_ARM_TIMEOUT_S = 2.0

LEARN_BUCKET_PX = 8
LEARN_SAVE_MIN_S = 15.0
LEARN_STEP_KP = 0.6
LEARN_STEP_KV = 0.00018
LEARN_ANNEAL = 0.8
LEARN_STEP_MIN_FRAC = 0.25
LEARN_EMA_ALPHA = 0.4
LEARN_KP_RANGE = (2.0, 14.0)
LEARN_KV_RANGE = (0.0005, 0.004)
LEARN_W_ERR, LEARN_W_FLIP, LEARN_W_OVER = 1.0, 300.0, 20.0
LEARN_J_DIVERGE = 60000.0
LEARN_MIN_TICKS = 30
LEARN_CLOSE_PX = 40.0


def _clamp(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


@dataclass
class RodGains:
    kp: float
    kv: float
    jbest: float | None
    step_kp: float
    step_kv: float


class GainLearner:
    """Per-rod (bar-width bucket) coordinate descent on KP/KV with a
    sign-alternating probe; a probe that lowers the reel cost pulls the best
    gains toward it, a worse one anneals that gain's step."""

    def __init__(self, path, log):
        self.path = path
        self.log = log
        self.rods = {}
        self.param = "kp"
        self.sign = 1
        self.last_save = -math.inf
        self.load()
        self._new_reel()

    def _new_reel(self):
        self.bucket = None
        self.best_kp = self.best_kv = self.kp = self.kv = None
        self.jbest = None
        self.step_kp = self.step_kv = None
        self.err2_sum = 0.0
        self.ticks = 0
        self.flips = 0
        self.overshoot = 0.0
        self.was_close = False
        self.t0 = None
        self.diverged = False

    def load(self):
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        for bucket, entry in data.items():
            try:
                self.rods[int(bucket)] = RodGains(**entry)
            except (TypeError, ValueError):
                continue
        if self.rods:
            self.log(f"reel learner: loaded {len(self.rods)} rod(s)")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({str(b): asdict(g) for b, g in self.rods.items()}, indent=1))
        tmp.replace(self.path)

    def gains(self, bar_width, now):
        """Active (KP, KV) for this reel, picked on its first cascade tick."""
        if self.t0 is None:
            self.t0 = now
            self.bucket = int(bar_width // LEARN_BUCKET_PX)
            entry = self.rods.get(self.bucket)
            if entry is None:
                nearest = min(self.rods.items(), key=lambda kv: abs(kv[0] - self.bucket), default=None)
                seed_kp, seed_kv = (nearest[1].kp, nearest[1].kv) if nearest else (CAS_KP, CAS_KV)
                entry = RodGains(seed_kp, seed_kv, None, LEARN_STEP_KP, LEARN_STEP_KV)
                self.rods[self.bucket] = entry
            self.best_kp, self.best_kv, self.jbest = entry.kp, entry.kv, entry.jbest
            self.step_kp, self.step_kv = entry.step_kp, entry.step_kv
            self.kp, self.kv = self.best_kp, self.best_kv
            if self.jbest is not None:
                if self.param == "kp":
                    self.kp = self.best_kp + self.sign * self.step_kp
                else:
                    self.kv = self.best_kv + self.sign * self.step_kv
            self.kp = _clamp(self.kp, *LEARN_KP_RANGE)
            self.kv = _clamp(self.kv, *LEARN_KV_RANGE)
        if self.diverged:
            return CAS_KP, CAS_KV
        return self.kp, self.kv

    def observe(self, err_px):
        self.ticks += 1
        self.err2_sum += err_px * err_px
        ae = abs(err_px)
        if ae <= LEARN_CLOSE_PX:
            self.was_close = True
        elif self.was_close and ae > self.overshoot:
            self.overshoot = ae
        if not self.diverged and self.ticks >= 10 and LEARN_W_ERR * self.err2_sum / self.ticks > LEARN_J_DIVERGE:
            self.diverged = True

    def _anneal(self):
        if self.param == "kp":
            self.step_kp = max(self.step_kp * LEARN_ANNEAL, LEARN_STEP_KP * LEARN_STEP_MIN_FRAC)
        else:
            self.step_kv = max(self.step_kv * LEARN_ANNEAL, LEARN_STEP_KV * LEARN_STEP_MIN_FRAC)

    def end_reel(self, now):
        if self.t0 is not None and self.ticks >= LEARN_MIN_TICKS:
            duration = max(now - self.t0, 0.1)
            cost = (LEARN_W_ERR * self.err2_sum / self.ticks
                    + LEARN_W_FLIP * self.flips / duration
                    + LEARN_W_OVER * self.overshoot)
            if self.diverged:
                action = "revert"
                self._anneal()
            elif self.jbest is None:
                action = "baseline"
                self.jbest = cost
            elif cost < self.jbest:
                action = "accept"
                if self.param == "kp":
                    self.best_kp += LEARN_EMA_ALPHA * (self.kp - self.best_kp)
                else:
                    self.best_kv += LEARN_EMA_ALPHA * (self.kv - self.best_kv)
                self.jbest = cost
            else:
                action = "reject"
                self._anneal()
            self.rods[self.bucket] = RodGains(self.best_kp, self.best_kv, self.jbest, self.step_kp, self.step_kv)
            best = cost if self.jbest is None else self.jbest
            self.log(f"reel learner: rod {self.bucket * LEARN_BUCKET_PX}px kp={self.best_kp:.2f} "
                     f"kv={self.best_kv:.5f} cost={cost:.0f} best={best:.0f} {action}")
            if now - self.last_save >= LEARN_SAVE_MIN_S:
                self.last_save = now
                try:
                    self.save()
                except OSError as e:
                    self.log(f"reel learner: could not save gains: {e}")
            self.param = "kv" if self.param == "kp" else "kp"
            self.sign = -self.sign
        self._new_reel()


class Reeler:
    """One instance per macro run. tick() returns "active" while the reel
    continues and "ended" when the bar has sat still for ~2 s."""

    def __init__(self, io, settings, learned_gains_path, log):
        self.io = io
        self.settings = settings            # read every tick, so changes apply live
        self.log = log
        self.learner = GainLearner(learned_gains_path, log)
        self.wall_lo = self.wall_hi = None   # observed track walls, sticky across reels
        self._down = False
        self.reset(now=None)

    def _set_mouse(self, down):
        if down == self._down:
            return
        self.io.button(down)
        self._down = down
        self.learner.flips += 1

    def release(self):
        self._set_mouse(False)

    def forget_walls(self):
        self.wall_lo = self.wall_hi = None

    def reset(self, now):
        """End of a reel (or macro start): release, score the reel for the learner, clear per-reel state."""
        self.release()
        if now is not None:
            self.learner.end_reel(now)   # scores only reels that ran on learned gains
        self.last_bar = self.last_fish = None
        self.last_bar_seen = self.last_fish_seen = -math.inf
        self.prev_obs_t = None
        self.prev_err = 0.0
        self.vhat = 0.0
        self.neutral_i = 0.0
        self.pwm_acc = 0.0
        self.wall_pin = None
        self.last_tick = None
        self.stuck_ticks = 0
        self.last_bar_x = None
        self.req_armed = False
        self.req_init_x = None
        self.req_arm_start = None
        self.req_last_event = -math.inf

    def tick(self, boxes, now, target_override=None, override_strict=False):
        bar_ok = boxes.bar is not None and boxes.bar.conf >= MIN_CONF
        fish_ok = boxes.fish is not None and boxes.fish.conf >= MIN_CONF
        if bar_ok:
            self.last_bar, self.last_bar_seen = boxes.bar, now
        if fish_ok:
            self.last_fish, self.last_fish_seen = boxes.fish, now
        bar_usable = bar_ok or now - self.last_bar_seen < HOLDOVER_S
        fish_usable = fish_ok or now - self.last_fish_seen < HOLDOVER_S
        if not (bar_usable and fish_usable) or now - boxes.frame_t > STALE_S:
            return "active"
        bar = boxes.bar if bar_ok else self.last_bar
        fish = boxes.fish if fish_ok else self.last_fish
        # Older of the two per-class captures feeding err: frozen while either
        # class rides on holdover, so v_hat never differentiates a stale position.
        obs_t = min(boxes.bar_t, boxes.fish_t)

        if self.last_bar_x is not None and abs(bar.x - self.last_bar_x) <= STUCK_DELTA_PX:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        self.last_bar_x = bar.x
        if self.stuck_ticks >= STUCK_TICKS:
            self.log("reel: bar static for 2 s, treating the reel as over")
            return "ended"

        if not self.settings.auto_reel:
            return "active"

        bar_w = max(1.0, bar.w)
        bar_center = bar.x + bar_w / 2
        fish_center = fish.x + fish.w / 2
        target = self._target(fish_center, bar_w, target_override, override_strict)
        err = target - bar_center

        if self.settings.rod == "requiem":
            self._requiem(bar.x, err, now)
            return "active"
        self._cascade(err, bar_center, bar_w, target, obs_t, now)
        return "active"

    @staticmethod
    def _target(fish_center, bar_w, override, strict):
        if override is None:
            return fish_center
        if strict:
            return override
        # Keep the bar's near edge just over the fish (catch progress keeps
        # accruing) while the rest of the bar reaches toward the override.
        diff = override - fish_center
        overlap = bar_w * SMART_TARGET_OVERLAP
        if abs(diff) < 1.0:
            return fish_center
        if abs(diff) <= bar_w - 2 * overlap:
            return fish_center + math.copysign(bar_w / 2 - overlap, diff)
        return override

    def _requiem(self, bar_x, err, now):
        if not self.req_armed:
            if self.req_init_x is None:
                self.req_init_x, self.req_arm_start = bar_x, now
            if abs(bar_x - self.req_init_x) > REQUIEM_FALL_THRESH_PX or now - self.req_arm_start >= REQUIEM_ARM_TIMEOUT_S:
                self.req_armed = True
                self.req_last_event = now   # the first transition still waits out the gate
            return
        if err > REQUIEM_DEADBAND_PX:
            want = True
        elif err < -REQUIEM_DEADBAND_PX:
            want = False
        else:
            want = self._down
        if want != self._down and now - self.req_last_event >= REQUIEM_MIN_INTERVAL_S:
            self._set_mouse(want)
            self.req_last_event = now

    def _cascade(self, err, bar_center, bar_w, target, obs_t, now):
        s = self.settings
        learning = s.learn or self.learner.t0 is not None   # a reel keeps the gains it started with
        if learning:
            kp, kv = self.learner.gains(bar_w, now)
        else:
            kp, kv = CAS_KP, CAS_KV
        kp *= s.aggressiveness_pct / 100
        kv /= max(s.damping_pct / 100, 0.1)
        vmax = CAS_VMAX * (1 + s.balance_pct / 200)

        if obs_t != self.prev_obs_t:
            if self.prev_obs_t is None:
                self.vhat = 0.0
            else:
                gap = obs_t - self.prev_obs_t
                if gap <= 0 or gap > VEL_GAP_RESET_S:
                    self.vhat = 0.0
                else:
                    raw_v = (err - self.prev_err) / max(gap, 0.01)
                    self.vhat += VEL_EMA_ALPHA * (raw_v - self.vhat)
            self.prev_obs_t = obs_t
            self.prev_err = err

        if s.latency_ms > 0:
            latency = s.latency_ms / 1000
        else:
            latency = _clamp(now - obs_t + ACTUATION_S, *LATENCY_RANGE_S)
        dt = TICK_S if self.last_tick is None else _clamp(now - self.last_tick, 0.004, 0.05)
        self.last_tick = now

        err_pred = err + self.vhat * latency
        v_des = _clamp(-kp * err_pred, -vmax, vmax)
        self.neutral_i = _clamp(self.neutral_i + CAS_KI * (err / 100) * dt, -CAS_NEUTRAL_CLAMP, CAS_NEUTRAL_CLAMP)
        duty = _clamp(0.5 + self.neutral_i + kv * (self.vhat - v_des), 0.0, 1.0)

        self.wall_lo = bar_center if self.wall_lo is None else min(self.wall_lo, bar_center)
        self.wall_hi = bar_center if self.wall_hi is None else max(self.wall_hi, bar_center)
        self._update_wall_pin(target, bar_w)

        if self.wall_pin == "right":
            want = True
        elif self.wall_pin == "left":
            want = False
        else:
            self.pwm_acc += duty
            want = self.pwm_acc >= 1.0
            if want:
                self.pwm_acc -= 1.0
        self._set_mouse(want)

        # A wall pin's large err is the fish's position, not a gain failure.
        if learning and self.wall_pin is None:
            self.learner.observe(err)
        if self.wall_pin is not None or (abs(err) < HOLD_STATION_PX and abs(self.vhat) < HOLD_STATION_VEL):
            self.stuck_ticks = 0

    def _update_wall_pin(self, target, bar_w):
        if self.wall_hi - self.wall_lo < WALL_MIN_TRACK_PX:
            self.wall_pin = None
            return
        enter = bar_w / 2 + WALL_MARGIN_PX
        leave = enter + WALL_HYST_PX
        d_right = self.wall_hi - target
        d_left = target - self.wall_lo
        if self.wall_pin == "right":
            if d_right > leave:
                self.wall_pin = None
        elif self.wall_pin == "left":
            if d_left > leave:
                self.wall_pin = None
        elif d_right <= enter and d_right <= d_left:
            self.wall_pin = "right"
        elif d_left <= enter:
            self.wall_pin = "left"
