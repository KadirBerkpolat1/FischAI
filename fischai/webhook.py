"""Discord webhook: start/stop notices and a periodic progress summary."""

import json
import threading
import time
import urllib.error
import urllib.request

from .config import CAST_GRADES


class Webhook:
    def __init__(self, settings, log):
        self.settings = settings            # read on every send, so the URL can change live
        self.log = log

    def send(self, text):
        url = self.settings.url.strip()
        if url:
            threading.Thread(target=self._post, args=(url, text), daemon=True, name="webhook").start()

    def _post(self, url, text):
        body = json.dumps({"content": text, "username": "FischAI"}).encode()
        request = urllib.request.Request(url, body, {"Content-Type": "application/json", "User-Agent": "FischAI"})
        try:
            urllib.request.urlopen(request, timeout=10).close()
        except (urllib.error.URLError, OSError) as e:
            self.log(f"webhook: could not post: {e}")


def summary(stats):
    minutes = int((time.time() - stats.started) // 60)
    parts = [f"{stats.reels} reels", f"{stats.casts} casts"]
    if stats.cast_grades:
        graded = ", ".join(f"{stats.cast_grades[g]} {g}" for g in reversed(CAST_GRADES) if g in stats.cast_grades)
        parts.append(f"casts graded {graded}")
    if stats.shake_timeouts:
        parts.append(f"{stats.shake_timeouts} casts without a bite")
    if stats.totems:
        parts.append(f"{stats.totems} totems")
    if stats.appraisals:
        parts.append(f"{stats.appraisals} appraisals")
    if stats.potion_rounds:
        parts.append(f"{stats.potion_rounds} potion rounds")
    if stats.aquarium_sweeps:
        parts.append(f"{stats.aquarium_sweeps} aquarium feeds")
    if stats.rejoins:
        parts.append(f"{stats.rejoins} rejoins")
    return f"{', '.join(parts)} in {minutes // 60} h {minutes % 60} min"
