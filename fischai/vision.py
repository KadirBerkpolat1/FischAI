"""
OpenCV detectors on full game-window frames ((h, w, 3) uint8 RGB), using
ExoPanda's templates from the data directory:

    templates/shake/*.png                  the SHAKE wordmark at three sizes
    templates/weather/cycle_day|night/*    the HUD day/night icon
    templates/weather/event/aurora/*       the Aurora weather-event icon
    templates/aquarium/list/*              the open aquarium's food list

All results are in frame pixels; the caller adds the window origin.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .config import CAST_GRADES

# Shake: the SHAKE wordmark on the round button, 43 px tall at a 1080 px
# client (measured in Sober). The button pops in and wobbles when clicked,
# so its size varies (0.7-1.06x in a recording) and every frame tries a few
# sizes. Matching runs at half resolution with the smallest of ExoPanda's
# templates, which correlates best with Sober's rendering (0.91 vs 0.83-0.86).
# The ROI stops short of the bottom-right HUD, which scores 0.57 at 0.82x.
SHAKE_ROI = (0.04, 0.10, 0.92, 0.82)
SHAKE_WORD_H = 43 / 1080
SHAKE_SIZE_STEPS = (0.85, 0.93, 1.0, 1.08)
SHAKE_MATCH_SCALE = 0.5
SHAKE_THRESH = 0.60             # recorded buttons mostly 0.6-0.93; static scenery up to 0.57
SHAKE_EXCLUDE_PX = 40           # radius around a spot the macro gave up on

# Cast power bar: an 8 px wide, ~212 px tall bar right of the character with
# a green marker at its top. While the button is held a white fill rises from
# the bottom, reaches the top, falls back and rises again; how full it is when
# the release lands sets the cast's grade. The empty part is a translucent
# blue tint, dark by night and light by day, so the fill is told by its own
# colour and by differing from the background beside the bar. Measured in
# Sober at 1880x998 by night and by day.
POWER_ROI = (0.48, 0.25, 0.24, 0.50)        # x, y, w, h as fractions of the client
POWER_GREEN_HSV = ((35, 110, 90), (85, 255, 255))
POWER_MARKER_W = (6, 48)                    # px: a short, thin horizontal line
POWER_MARKER_H = (2, 10)
POWER_HALF_W = 2                            # half-width of the column read in the middle of the bar
POWER_SIDE_PX = (9, 13)                     # the background beside the bar, this far from its middle
POWER_FILL_MAX_S = 30                       # the fill is near white (S 11-12, V 217-231); the empty
POWER_FILL_MIN_V = 170                      # bar has S 50-80, clouds below the bar V ~160
POWER_FILL_CONTRAST = 20                    # the fill differs from both sides by 44+, clouds by up to 12
POWER_BAR_ROWS = (160, 265)                 # the fill ends at the bar's bottom, ~212 rows below the marker's top
POWER_FILL_GAP = 3                          # rows a fill may skip and still be one fill
POWER_SCENERY_PX = 6                        # a green blob this close to one seen before the press is scenery

# The cast's grade ("Meh..", "Fine.", "Good!", "Great!", "Amazing!!",
# "PERFECT!") is written over the character for about a second.
CAST_TEXT_ROI = (0.38, 0.28, 0.24, 0.18)

# HUD status icons, bottom-right quadrant.
STATUS_ROI = (0.55, 0.50, 0.45, 0.48)
CYCLE_LOCATE_FLOOR = 0.80       # best template correlation needed to say "an icon is here"
CYCLE_MARGIN = 0.05             # min day/night score gap before committing
CYCLE_REFINE_PX = 2
WEATHER_EVENT_THRESH = 0.70
EVENTS = ("aurora",)

AQUARIUM_LIST_THRESH = 0.70


def _roi(shape, fractions):
    h, w = shape[:2]
    fx, fy, fw, fh = fractions
    return int(w * fx), int(h * fy), int(w * fw), int(h * fh)


def load_gray_templates(folder: Path):
    """Grayscale templates. A transparent PNG is composited over black, so a
    flat glyph whose shape lives only in its alpha channel (the white sun icon)
    becomes a usable bright-on-dark template."""
    templates = []
    for path in sorted(folder.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 2:
            templates.append(image)
            continue
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        if image.shape[2] == 4:
            gray = (gray.astype(np.float32) * (image[:, :, 3] / 255.0)).astype(np.uint8)
        templates.append(gray)
    if not templates:
        raise RuntimeError(f"no templates in {folder}; run `fischai setup` first")
    return templates


def best_match(gray, templates):
    """(score, x, y, w, h) of the best TM_CCOEFF_NORMED match over all templates, or None."""
    best = None
    for template in templates:
        th, tw = template.shape[:2]
        if th > gray.shape[0] or tw > gray.shape[1]:
            continue
        scores = np.nan_to_num(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED), nan=-1.0)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if best is None or score > best[0]:
            best = (score, x, y, tw, th)
    return best


class ShakeDetector:
    """Finds the SHAKE button by its wordmark alone. ExoPanda's primary
    locator (HoughCircles plus a near-white pixel count) also accepts the
    bobber and bright scenery in Sober, and clicking those does nothing."""

    def __init__(self, template_root):
        self.source = min(load_gray_templates(Path(template_root) / "shake"), key=lambda t: t.shape[0])
        self._sized = {}        # client height -> templates at match scale, one per size step
        self.last_score = None

    def _templates(self, client_h):
        if client_h not in self._sized:
            t = self.source
            heights = (SHAKE_WORD_H * client_h * f * SHAKE_MATCH_SCALE for f in SHAKE_SIZE_STEPS)
            self._sized[client_h] = [
                cv2.resize(t, (max(1, round(t.shape[1] * h / t.shape[0])), max(1, round(h))), interpolation=cv2.INTER_AREA)
                for h in heights
            ]
        return self._sized[client_h]

    def find(self, frame, exclude=()):
        """Shake button centre (frame px) or None, ignoring matches centred
        within SHAKE_EXCLUDE_PX of the `exclude` points; `last_score` keeps
        the best correlation."""
        rx, ry, rw, rh = _roi(frame.shape, SHAKE_ROI)
        gray = cv2.cvtColor(frame[ry:ry + rh, rx:rx + rw], cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, None, fx=SHAKE_MATCH_SCALE, fy=SHAKE_MATCH_SCALE, interpolation=cv2.INTER_AREA)
        r = round(SHAKE_EXCLUDE_PX * SHAKE_MATCH_SCALE)
        best = None
        for template in self._templates(frame.shape[0]):
            th, tw = template.shape
            if th > small.shape[0] or tw > small.shape[1]:
                continue
            scores = np.nan_to_num(cv2.matchTemplate(small, template, cv2.TM_CCOEFF_NORMED), nan=-1.0, posinf=-1.0, neginf=-1.0)
            for ex, ey in exclude:
                x0 = round((ex - rx) * SHAKE_MATCH_SCALE - tw / 2)
                y0 = round((ey - ry) * SHAKE_MATCH_SCALE - th / 2)
                scores[max(0, y0 - r):max(0, y0 + r + 1), max(0, x0 - r):max(0, x0 + r + 1)] = -1.0
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            if best is None or score > best[0]:
                best = (score, x + tw / 2, y + th / 2)
        self.last_score = None if best is None else best[0]
        if best is None or best[0] < SHAKE_THRESH:
            return None
        _, x, y = best
        return rx + x / SHAKE_MATCH_SCALE, ry + y / SHAKE_MATCH_SCALE


class PowerBar:
    """Reads the cast power bar from crops of POWER_ROI (see `crop`), one cast
    at a time, with fill() on every frame. `before` is a crop taken before the
    press: green marker-like blobs already in it are scenery."""

    def __init__(self, before=None):
        self.marker = None      # (column x, marker top y) in crop px, where the bar was last found
        self.scenery = [] if before is None else [(x, y) for x, y, _ in self._markers(before)]

    @staticmethod
    def crop(win_rect):
        """Layout-pixel rectangle of the power bar search area."""
        x, y, w, h = win_rect
        fx, fy, fw, fh = POWER_ROI
        return round(x + fx * w), round(y + fy * h), round(fw * w), round(fh * h)

    @staticmethod
    def _markers(rgb):
        """(middle x, top y, bottom y) of every marker-shaped green blob."""
        green = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), *POWER_GREEN_HSV)
        n, _, stats, _ = cv2.connectedComponentsWithStats(green)
        return [(int(x + w // 2), int(y), int(y + h)) for x, y, w, h, _area in stats[1:n]
                if POWER_MARKER_W[0] <= w <= POWER_MARKER_W[1] and POWER_MARKER_H[0] <= h <= POWER_MARKER_H[1]]

    @staticmethod
    def _fill_rows(rgb, cx, top):
        """(first, end) rows of the fill below a marker at (cx, top), counted from
        the marker's top, when a fill run ends where the bar's bottom can be."""
        x0, x1 = cx - POWER_SIDE_PX[1], cx + POWER_SIDE_PX[1] + 1
        if x0 < 0 or x1 > rgb.shape[1]:
            return None
        strip = rgb[top:top + POWER_BAR_ROWS[1], x0:x1]
        mid = slice(POWER_SIDE_PX[1] - POWER_HALF_W, POWER_SIDE_PX[1] + POWER_HALF_W + 1)
        left = slice(0, POWER_SIDE_PX[1] - POWER_SIDE_PX[0] + 1)
        right = slice(POWER_SIDE_PX[1] + POWER_SIDE_PX[0], 2 * POWER_SIDE_PX[1] + 1)
        hsv = cv2.cvtColor(strip[:, mid], cv2.COLOR_RGB2HSV).mean(axis=1)
        gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY).astype(np.float32)
        middle = gray[:, mid].mean(axis=1)
        fill = ((hsv[:, 1] <= POWER_FILL_MAX_S) & (hsv[:, 2] >= POWER_FILL_MIN_V)
                & (np.abs(middle - gray[:, left].mean(axis=1)) >= POWER_FILL_CONTRAST)
                & (np.abs(middle - gray[:, right].mean(axis=1)) >= POWER_FILL_CONTRAST))
        rows = np.flatnonzero(fill)
        if not rows.size:
            return None
        breaks = np.flatnonzero(np.diff(rows) > POWER_FILL_GAP)
        first, last = (int(rows[breaks[-1] + 1]) if breaks.size else int(rows[0])), int(rows[-1])
        end = last + 1
        return (first, end) if POWER_BAR_ROWS[0] <= end <= POWER_BAR_ROWS[1] else None

    def fill(self, rgb):
        """How full the bar is (0..1), as the game's memory has it: the fill's
        height over the bar's, from the marker's top to the fill's bottom.
        The bar sways with the character, so it is looked for on every frame;
        near the top the fill covers the marker and its last place is used.
        None when no bar or no fill shows."""
        candidates = [(x, y) for x, y, _ in self._markers(rgb)
                      if not any(abs(x - sx) <= POWER_SCENERY_PX and abs(y - sy) <= POWER_SCENERY_PX
                                 for sx, sy in self.scenery)]
        if self.marker is not None:
            candidates.append(self.marker)
        for cx, top in candidates:
            rows = self._fill_rows(rgb, cx, top)
            if rows is not None:
                self.marker = (cx, top)
                first, end = rows
                return (end - first) / end
        return None


def cast_text_crop(win_rect):
    x, y, w, h = win_rect
    fx, fy, fw, fh = CAST_TEXT_ROI
    return round(x + fx * w), round(y + fy * h), round(fw * w), round(fh * h)


def cast_grade(rgb):
    """The cast grade word in a crop of CAST_TEXT_ROI ("perfect", …) or None (tesseract, ~0.1 s)."""
    ok, png = cv2.imencode(".png", cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None, fx=1.5, fy=1.5))
    if not ok:
        return None
    out = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "11"], input=png.tobytes(),
                         capture_output=True, check=False)
    text = out.stdout.decode(errors="replace").lower()
    return next((grade for grade in CAST_GRADES if grade in text), None)


def appraise_row(rgb):
    """(x, y) centre of the NPC dialog's appraise option in a full game-window
    frame, or None (tesseract TSV, ~1 s). The dialog rides on the NPC as a
    BillboardGui, so where it lands on screen follows the camera; only the
    text is stable."""
    gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None, fx=1.5, fy=1.5)
    ok, png = cv2.imencode(".png", gray)
    if not ok:
        return None
    out = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "11", "tsv"], input=png.tobytes(),
                         capture_output=True, check=False)
    lines = {}
    for row in out.stdout.decode(errors="replace").splitlines()[1:]:
        f = row.split("\t")
        if len(f) < 12 or not f[11].strip():
            continue
        try:
            x, y, w, h = (int(float(f[i])) for i in (6, 7, 8, 9))
        except ValueError:
            continue
        lines.setdefault(tuple(f[2:5]), []).append((x, y, w, h, f[11].lower()))
    for words in lines.values():
        text = " ".join(w[-1] for w in words)
        if "appraise" in text and ("c$" in text or "again" in text):
            return (round((min(w[0] for w in words) + max(w[0] + w[2] for w in words)) / 2 / 1.5),
                    round((min(w[1] for w in words) + max(w[1] + w[3] for w in words)) / 2 / 1.5))
    return None


class CycleDetector:
    """Two-stage day/night read of the HUD icon. Stage 1 locates the icon as the
    best match of any template; stage 2 scores day vs night only around that
    spot and commits only when one side wins by CYCLE_MARGIN, so a crescent
    that partially matches the sun glyph cannot flip the answer."""

    def __init__(self, template_root):
        root = Path(template_root) / "weather"
        self.day = load_gray_templates(root / "cycle_day")
        self.night = load_gray_templates(root / "cycle_night")
        self.last_scores = None

    @staticmethod
    def _match(image, template):
        return np.nan_to_num(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED), nan=-1.0, posinf=-1.0, neginf=-1.0)

    def _score_at(self, roi, templates, x, y, w, h):
        cx, cy = x + w / 2, y + h / 2
        best = -1.0
        for template in templates:
            th, tw = template.shape[:2]
            x0 = max(0, round(cx - tw / 2) - CYCLE_REFINE_PX)
            y0 = max(0, round(cy - th / 2) - CYCLE_REFINE_PX)
            patch = roi[y0:y0 + th + 2 * CYCLE_REFINE_PX, x0:x0 + tw + 2 * CYCLE_REFINE_PX]
            if patch.shape[0] < th or patch.shape[1] < tw:
                continue
            best = max(best, float(self._match(patch, template).max()))
        return best

    def detect(self, frame):
        """"day", "night", or None when no icon is found or the read is ambiguous."""
        rx, ry, rw, rh = _roi(frame.shape, STATUS_ROI)
        roi = cv2.cvtColor(frame[ry:ry + rh, rx:rx + rw], cv2.COLOR_RGB2GRAY)
        located = best_match(roi, self.day + self.night)
        if located is None or located[0] < CYCLE_LOCATE_FLOOR:
            self.last_scores = (located[0] if located else None, None, None)
            return None
        score, x, y, w, h = located
        day = self._score_at(roi, self.day, x, y, w, h)
        night = self._score_at(roi, self.night, x, y, w, h)
        self.last_scores = (score, day, night)
        if abs(day - night) < CYCLE_MARGIN:
            return None
        return "day" if day > night else "night"


class WeatherEventDetector:
    def __init__(self, template_root):
        root = Path(template_root) / "weather" / "event"
        self.events = {name: load_gray_templates(root / name) for name in EVENTS if (root / name).is_dir()}

    def detect(self, frame):
        """Name of the active weather event, or "clear"."""
        rx, ry, rw, rh = _roi(frame.shape, STATUS_ROI)
        roi = cv2.cvtColor(frame[ry:ry + rh, rx:rx + rw], cv2.COLOR_RGB2GRAY)
        for name, templates in self.events.items():
            hit = best_match(roi, templates)
            if hit is not None and hit[0] >= WEATHER_EVENT_THRESH:
                return name
        return "clear"


class AquariumPanelDetector:
    """Confirms the aquarium panel opened before the sweep scrolls (scrolling
    with the panel closed zooms the camera instead of the food list)."""

    def __init__(self, template_root):
        self.templates = load_gray_templates(Path(template_root) / "aquarium" / "list")
        self.last_score = None

    def is_open(self, frame):
        hit = best_match(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), self.templates)
        self.last_score = hit[0] if hit else None
        return hit is not None and hit[0] >= AQUARIUM_LIST_THRESH
