"""
OpenCV detectors on full game-window frames ((h, w, 3) uint8 RGB), using
ExoPanda's templates from the data directory:

    templates/shake/*.png                  the SHAKE wordmark at three sizes
    templates/weather/cycle_day|night/*    the HUD day/night icon
    templates/weather/event/aurora/*       the Aurora weather-event icon
    templates/aquarium/list/*              the open aquarium's food list

All results are in frame pixels; the caller adds the window origin.
"""

from pathlib import Path

import cv2
import numpy as np

# Shake (fisch_vision.lua / fisch_data.lua). Radii are client pixels at a
# 900 px tall client and scale with the frame height.
SHAKE_ROI = (0.10, 0.22, 0.80, 0.66)
SHAKE_CIRCLE_MIN_R = 28
SHAKE_CIRCLE_MAX_R = 80
SHAKE_CIRCLE_PARAM1 = 100       # Canny high threshold
SHAKE_CIRCLE_PARAM2 = 35        # accumulator threshold; lower = more circles
SHAKE_CIRCLE_MIN_WHITE = 150    # near-white pixels required inside the circle
SHAKE_WHITE_THRESH = 200
SHAKE_TEMPLATE_THRESH = 0.65
SHAKE_MISS_FULLFRAME_S = 0.6    # full-frame template sweep cadence on the miss path
REFERENCE_CLIENT_H = 900

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
    def __init__(self, template_root):
        self.templates = load_gray_templates(Path(template_root) / "shake")
        self._last_full_sweep = float("-inf")
        self.last_method = None

    def _find_circle(self, roi_rgb, scale):
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        min_r = max(4, round(SHAKE_CIRCLE_MIN_R * scale))
        max_r = max(min_r + 1, round(SHAKE_CIRCLE_MAX_R * scale))
        circles = cv2.HoughCircles(
            cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1, minDist=min_r * 2,
            param1=SHAKE_CIRCLE_PARAM1, param2=SHAKE_CIRCLE_PARAM2, minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            return None
        white = np.all(roi_rgb > SHAKE_WHITE_THRESH, axis=2).astype(np.uint8)
        min_white = SHAKE_CIRCLE_MIN_WHITE * scale * scale
        best = None
        for cx, cy, r in circles[0]:
            cx, cy, r = int(cx), int(cy), int(r)
            x0, y0 = max(0, cx - r), max(0, cy - r)
            patch = white[y0:cy + r + 1, x0:cx + r + 1]
            disk = np.zeros_like(patch)
            cv2.circle(disk, (cx - x0, cy - y0), r, 1, -1)
            count = int(np.count_nonzero(patch & disk))
            if count >= min_white and (best is None or count > best[2]):
                best = (float(cx), float(cy), count)
        return None if best is None else best[:2]

    def _find_template(self, rgb):
        hit = best_match(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), self.templates)
        if hit is None or hit[0] < SHAKE_TEMPLATE_THRESH:
            return None
        score, x, y, w, h = hit
        return x + w / 2, y + h / 2

    def find(self, frame, first_look, now):
        """Shake button centre (frame px) or None. Circle detector first, then the
        wordmark template inside the lower-centre ROI. Until the first click of a
        cast (`first_look`) every frame also sweeps the whole frame; afterwards
        the full sweep runs at most every SHAKE_MISS_FULLFRAME_S."""
        rx, ry, rw, rh = _roi(frame.shape, SHAKE_ROI)
        roi = frame[ry:ry + rh, rx:rx + rw]
        scale = frame.shape[0] / REFERENCE_CLIENT_H
        hit = self._find_circle(roi, scale)
        self.last_method = "circle"
        if hit is None:
            hit = self._find_template(roi)
            self.last_method = "template"
        if hit is not None:
            return rx + hit[0], ry + hit[1]
        if first_look or now - self._last_full_sweep >= SHAKE_MISS_FULLFRAME_S:
            self._last_full_sweep = now
            self.last_method = "full-frame template"
            return self._find_template(frame)
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
