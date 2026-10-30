"""Discord webhook: start/stop notices and a periodic progress summary."""

import json
import threading
import time
import urllib.error
import urllib.request


class Webhook:
    def __init__(self, url, log):
        self.url = url.strip()
        self.log = log

    def send(self, text):
        if self.url:
            threading.Thread(target=self._post, args=(text,), daemon=True, name="webhook").start()

    def _post(self, text):
        body = json.dumps({"content": text, "username": "FischAI"}).encode()
        request = urllib.request.Request(self.url, body, {"Content-Type": "application/json", "User-Agent": "FischAI"})
        try:
            urllib.request.urlopen(request, timeout=10).close()
        except (urllib.error.URLError, OSError) as e:
            self.log(f"webhook: could not post: {e}")


def summary(stats):
    minutes = int((time.time() - stats.started) // 60)
    parts = [f"{stats.reels} reels", f"{stats.casts} casts"]
    if stats.shake_timeouts:
        parts.append(f"{stats.shake_timeouts} casts without a bite")
    if stats.totems:
        parts.append(f"{stats.totems} totems")
    if stats.potion_rounds:
        parts.append(f"{stats.potion_rounds} potion rounds")
    if stats.aquarium_sweeps:
        parts.append(f"{stats.aquarium_sweeps} aquarium feeds")
    return f"{', '.join(parts)} in {minutes // 60} h {minutes % 60} min"
