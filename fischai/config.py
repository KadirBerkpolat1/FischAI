"""Settings in ~/.config/fischai/config.toml, written with defaults on first run.
Defaults mirror ExoPanda Fischer's own (fisch.cfg)."""

import json
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

TEMPLATE = """\
# FischAI settings. The settings window (`fischai gui`) rewrites this file and
# applies changes at once; `fischai run` reads it each time the macro starts.
# Stand facing water with the rod in hotbar slot 1.

[general]
hotkey = {general_hotkey}          # start / stop
window = {general_window}          # window class, or a title substring
debug_snapshots = {general_debug_snapshots}  # save what the macro saw at each click and phase change

[fishing]
auto_cast = {fishing_auto_cast}
auto_shake = {fishing_auto_shake}
perfect_cast = {fishing_perfect_cast}          # release when the power bar's fill reaches the green marker
cast_hold_ms = {fishing_cast_hold_ms}          # how long the cast is held when the power bar is not used
recast_delay_ms = {fishing_recast_delay_ms}    # wait after a catch before casting again

[reel]
auto_reel = {reel_auto_reel}
backend = {reel_backend}           # auto | ncnn | onnxruntime
fps = {reel_fps}                   # detection rate while fishing
region = {reel_region}             # "auto" or [x1, y1, x2, y2] as fractions of the game window
rod = {reel_rod}                   # normal | requiem
aggressiveness_pct = {reel_aggressiveness_pct}  # how hard the bar chases the fish
damping_pct = {reel_damping_pct}   # higher = gentler, less overshoot
balance_pct = {reel_balance_pct}   # raises the bar's top speed
latency_ms = {reel_latency_ms}     # 0 = measure; else fixed latency compensation
learn = {reel_learn}               # tune the reel gains per rod over time

[totem]
enabled = {totem_enabled}
day_slot = {totem_day_slot}        # hotbar 1-9, 0 = off (e.g. Sundial)
night_slot = {totem_night_slot}    # hotbar 1-9, 0 = off (e.g. Aurora)

[potions]
enabled = {potions_enabled}
slots = {potions_slots}            # hotbar slots, 0 = off
interval_min = {potions_interval_min}
only_during_weather_event = {potions_only_during_weather_event}

[aquarium]
enabled = {aquarium_enabled}       # hourly feed; briefly resizes the game window to 1680x900

[recovery]
enabled = {recovery_enabled}       # rejoin Fisch when Sober closes or Roblox disconnects, then walk the recorded route

[webhook]
url = {webhook_url}                # Discord webhook URL
summary_min = {webhook_summary_min}
"""


@dataclass
class General:
    hotkey: str = "f3"
    window: str = "org.vinegarhq.Sober"
    debug_snapshots: bool = False


@dataclass
class Fishing:
    auto_cast: bool = True
    auto_shake: bool = True
    perfect_cast: bool = True
    cast_hold_ms: int = 500
    recast_delay_ms: int = 1500


@dataclass
class Reel:
    auto_reel: bool = True
    backend: str = "auto"
    fps: int = 60
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
class Recovery:
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
    recovery: Recovery = field(default_factory=Recovery)
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


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)   # JSON strings are valid TOML basic strings
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {value!r} to TOML")


def render(cfg):
    values = {}
    for section in fields(Config):
        obj = getattr(cfg, section.name)
        for f in fields(obj):
            values[f"{section.name}_{f.name}"] = _toml_value(getattr(obj, f.name))
    return TEMPLATE.format(**values)


def save(cfg, path=CONFIG_PATH):
    _validate(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(render(cfg))
    tmp.replace(path)


def load(path=CONFIG_PATH):
    if not path.exists():
        save(Config(), path)
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
