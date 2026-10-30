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

# Shake: the SHAKE wordmark on the round button. Its height is a fixed share
# of the client height (43 px at 1080, measured in Sober); until a confident
# match pins the real size, sizes around that are tried. Matching runs at
# half resolution.
SHAKE_ROI = (0.04, 0.10, 0.92, 0.82)
SHAKE_WORD_H = 43 / 1080
SHAKE_SIZE_STEPS = (0.88, 1.0, 1.12)
SHAKE_MATCH_SCALE = 0.5
SHAKE_THRESH = 0.55             # real buttons score 0.65-0.9 at half resolution, anything else under 0.45
SHAKE_PIN_THRESH = 0.70         # a match this good fixes the wordmark size for that client height

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
        self.sources = load_gray_templates(Path(template_root) / "shake")
        self._sized = {}        # (client height, size factor) -> templates at match scale
        self._pinned = {}       # client height -> size factor of a confident match
        self.last_score = None

    def _templates(self, client_h, factor):
        key = (client_h, factor)
        if key not in self._sized:
            h = SHAKE_WORD_H * client_h * factor * SHAKE_MATCH_SCALE
            self._sized[key] = [
                cv2.resize(t, (max(1, round(t.shape[1] * h / t.shape[0])), max(1, round(h))), interpolation=cv2.INTER_AREA)
                for t in self.sources
            ]
        return self._sized[key]

    def find(self, frame):
        """Shake button centre (frame px) or None; `last_score` keeps the best correlation."""
        client_h = frame.shape[0]
        rx, ry, rw, rh = _roi(frame.shape, SHAKE_ROI)
        gray = cv2.cvtColor(frame[ry:ry + rh, rx:rx + rw], cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, None, fx=SHAKE_MATCH_SCALE, fy=SHAKE_MATCH_SCALE, interpolation=cv2.INTER_AREA)
        pinned = self._pinned.get(client_h)
        best = None
        for factor in (pinned,) if pinned else SHAKE_SIZE_STEPS:
            for template in self._templates(client_h, factor):
                th, tw = template.shape
                if th > small.shape[0] or tw > small.shape[1]:
                    continue
                scores = np.nan_to_num(cv2.matchTemplate(small, template, cv2.TM_CCOEFF_NORMED), nan=-1.0, posinf=-1.0, neginf=-1.0)
                _, score, _, (x, y) = cv2.minMaxLoc(scores)
                if best is None or score > best[0]:
                    best = (score, x + tw / 2, y + th / 2, factor)
        self.last_score = None if best is None else best[0]
        if best is None or best[0] < SHAKE_THRESH:
            return None
        score, x, y, factor = best
        if pinned is None and score >= SHAKE_PIN_THRESH:
            self._pinned[client_h] = factor
        return rx + x / SHAKE_MATCH_SCALE, ry + y / SHAKE_MATCH_SCALE


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
