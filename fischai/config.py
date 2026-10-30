"""Settings in ~/.config/fischai/config.toml, written with defaults on first run.
Defaults mirror ExoPanda Fischer's own (fisch.cfg)."""

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .inputdev import keycode


def _xdg(var, fallback):
    return Path(os.environ.get(var) or Path.home() / fallback) / "fischai"


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", ".config")
DATA_DIR = _xdg("XDG_DATA_HOME", ".local/share")
STATE_DIR = _xdg("XDG_STATE_HOME", ".local/state")
CONFIG_PATH = CONFIG_DIR / "config.toml"

DEFAULT_CONFIG = """\
# FischAI settings. Read each time the macro starts (hotkey), so edits apply
# on the next start. Stand facing water with the rod in hotbar slot 1.

[general]
hotkey = "f3"                    # start / stop
window = "org.vinegarhq.Sober"   # window class, or a title substring

[fishing]
auto_cast = true
auto_shake = true
cast_hold_ms = 500               # how long the cast is held
recast_delay_ms = 1500           # wait after a catch before casting again

[reel]
auto_reel = true
backend = "auto"                 # auto | ncnn | onnxruntime
fps = 30                         # detection rate while fishing
region = "auto"                  # or [x1, y1, x2, y2] as fractions of the game window
rod = "normal"                   # normal | requiem
aggressiveness_pct = 100         # how hard the bar chases the fish
damping_pct = 100                # higher = gentler, less overshoot
balance_pct = 0                  # raises the bar's top speed
latency_ms = 0                   # 0 = measure; else fixed latency compensation
learn = true                     # tune the reel gains per rod over time

[totem]
enabled = false
day_slot = 0                     # hotbar 1-9, 0 = off (e.g. Sundial)
night_slot = 0                   # hotbar 1-9, 0 = off (e.g. Aurora)

[potions]
enabled = false
slots = [0, 0, 0, 0]             # hotbar slots, 0 = off
interval_min = 30
only_during_weather_event = false

[aquarium]
enabled = false                  # hourly feed; briefly resizes the game window to 1680x900

[webhook]
url = ""                         # Discord webhook URL
summary_min = 30
"""


@dataclass
class General:
    hotkey: str = "f3"
    window: str = "org.vinegarhq.Sober"


@dataclass
class Fishing:
    auto_cast: bool = True
    auto_shake: bool = True
    cast_hold_ms: int = 500
    recast_delay_ms: int = 1500


@dataclass
class Reel:
    auto_reel: bool = True
    backend: str = "auto"
    fps: int = 30
    region: object = "auto"
    rod: str = "normal"
    aggressiveness_pct: int = 100
    damping_pct: int = 100
    balance_pct: int = 0
    latency_ms: int = 0
    learn: bool = True


@dataclass
class Totem:
    enabled: bool = False
    day_slot: int = 0
    night_slot: int = 0


@dataclass
class Potions:
    enabled: bool = False
    slots: list = field(default_factory=lambda: [0, 0, 0, 0])
    interval_min: int = 30
    only_during_weather_event: bool = False


@dataclass
class Aquarium:
    enabled: bool = False


@dataclass
class Webhook:
    url: str = ""
    summary_min: int = 30


@dataclass
class Config:
    general: General = field(default_factory=General)
    fishing: Fishing = field(default_factory=Fishing)
    reel: Reel = field(default_factory=Reel)
    totem: Totem = field(default_factory=Totem)
    potions: Potions = field(default_factory=Potions)
    aquarium: Aquarium = field(default_factory=Aquarium)
    webhook: Webhook = field(default_factory=Webhook)

    @property
    def region_fractions(self):
        return None if self.reel.region == "auto" else tuple(self.reel.region)


class ConfigError(Exception):
    pass


def _check(cond, message):
    if not cond:
        raise ConfigError(message)


def _validate(cfg):
    keycode(cfg.general.hotkey)
    f, r = cfg.fishing, cfg.reel
    _check(50 <= f.cast_hold_ms <= 5000, "fishing.cast_hold_ms must be 50-5000")
    _check(100 <= f.recast_delay_ms <= 10000, "fishing.recast_delay_ms must be 100-10000")
    _check(r.backend in ("auto", "ncnn", "onnxruntime"), "reel.backend must be auto, ncnn or onnxruntime")
    _check(5 <= r.fps <= 60, "reel.fps must be 5-60")
    _check(r.rod in ("normal", "requiem"), "reel.rod must be normal or requiem")
    _check(10 <= r.aggressiveness_pct <= 300, "reel.aggressiveness_pct must be 10-300")
    _check(10 <= r.damping_pct <= 300, "reel.damping_pct must be 10-300")
    _check(0 <= r.balance_pct <= 200, "reel.balance_pct must be 0-200")
    _check(0 <= r.latency_ms <= 300, "reel.latency_ms must be 0-300")
    if r.region != "auto":
        ok = isinstance(r.region, list) and len(r.region) == 4 and all(isinstance(v, (int, float)) for v in r.region)
        _check(ok and 0 <= r.region[0] < r.region[2] <= 1 and 0 <= r.region[1] < r.region[3] <= 1,
               'reel.region must be "auto" or [x1, y1, x2, y2] with 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1')
    for name in ("day_slot", "night_slot"):
        _check(0 <= getattr(cfg.totem, name) <= 9, f"totem.{name} must be 0-9")
    p = cfg.potions
    _check(isinstance(p.slots, list) and len(p.slots) <= 4 and all(isinstance(s, int) and 0 <= s <= 9 for s in p.slots),
           "potions.slots must be up to four hotbar numbers 0-9")
    _check(p.interval_min >= 10, "potions.interval_min must be at least 10")
    _check(cfg.webhook.summary_min >= 5, "webhook.summary_min must be at least 5")


def load(path=CONFIG_PATH):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG)
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
    cfg = Config()
    sections = {f.name for f in fields(Config)}
    for section, values in raw.items():
        _check(section in sections and isinstance(values, dict), f"{path}: unknown section [{section}]")
        target = getattr(cfg, section)
        known = {f.name: f for f in fields(target)}
        for key, value in values.items():
            _check(key in known, f"{path}: unknown setting {section}.{key}")
            default = getattr(target, key)
            if isinstance(default, bool):
                _check(isinstance(value, bool), f"{section}.{key} must be true or false")
            elif isinstance(default, int):
                _check(isinstance(value, int) and not isinstance(value, bool), f"{section}.{key} must be a whole number")
            elif isinstance(default, str) and key != "region":
                _check(isinstance(value, str), f"{section}.{key} must be text")
            setattr(target, key, value)
    _validate(cfg)
    return cfg
