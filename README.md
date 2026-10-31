# FischAI

A Fisch (Roblox) fishing macro for [Sober](https://sober.vinegarhq.org/) on **Hyprland**.
It casts at a grade you pick (Meh … Perfect), clicks the shake button, plays the
reel minigame with ExoPanda's detection model, cycles day/night totems, feeds
the aquarium, drinks potions, rejoins after a disconnect — and can re-appraise
a fish at the Appraiser NPC until it has the mutation and weight you want.

State is read from the game's memory (read-only, via `process_vm_readv`) when
available and from the screen otherwise.

> **Hyprland only, for now.** Window handling goes through `hyprctl` and
> screen capture through `grim` (wlr-screencopy), so KDE/GNOME sessions are
> not supported yet. The hotkey and input injection are kernel-level
> (`/dev/input`, `/dev/uinput`) and would work anywhere.

## Requirements

- Linux with a running **Hyprland** session
- **Sober** with Fisch installed (`org.vinegarhq.Sober`, e.g. from Flathub)
- System tools: `grim` and `tesseract` (Arch: `sudo pacman -S grim tesseract`)
- [uv](https://docs.astral.sh/uv/) (or any Python ≥ 3.12 environment)
- The ExoPanda Fisch module zip (`fisch-<version>.zip`) — **not included**;
  you supply your own copy for `fischai setup`
- Permissions (one-time):
  ```sh
  sudo usermod -aG input "$USER"          # read the keyboard for the hotkey
  echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger
  # then log out and back in
  ```

## Install

```sh
git clone https://github.com/KadirBerkpolat1/FischAI.git
cd FischAI
uv sync

.venv/bin/fischai setup /path/to/fisch-1.4.3.zip   # installs the reel model + templates
.venv/bin/fischai doctor                            # checks everything above
.venv/bin/fischai gui                               # settings window
```

`fischai doctor` must report `ok` on every line before you start.

## Use

1. In Sober, stand facing water with the rod in hotbar slot 1.
2. Open `fischai gui`, tune what you want (everything applies and saves at once).
3. Press **F3** (configurable) to start/stop, in the window or in the game.

Features, all optional and off unless enabled (except the fishing loop):

- **Cast grade** — Meh, Fine, Good, Great, Amazing or Perfect; the macro times
  the release on the power bar and logs the grade Fisch actually gave.
- **Shake + reel** — memory-precise when memory reading is on; the reel
  controller learns per-rod gains over time.
- **Totems by name** — the totem is found in the hotbar by its name
  (e.g. Sundial by day, Aurora by night), whichever slot it is in. When one
  runs out, that half switches off for the run and fishing continues.
- **Auto appraise** — at the Appraiser NPC with the fish in hand and the
  dialog open, re-rolls until the fish's name carries the wanted mutation
  and reaches the weight floor, or the attempt cap is spent (every roll
  costs C$, so it never runs without a goal).
- **Potions, aquarium, rejoin** — timed potion rounds, hourly aquarium
  feeding, and automatic rejoin + recorded walk back after a disconnect.
- **Discord webhook** — start/stop notices and periodic summaries.

## Commands

| Command | What it does |
|---|---|
| `fischai gui` | Settings window: start/stop and tune everything |
| `fischai run` | Same macro without the window (F3 toggles) |
| `fischai setup <zip>` | Install ExoPanda's model and templates from your module zip |
| `fischai doctor` | Check the environment, step by step |
| `fischai detect` | Test the reel detector on the live game or an image |
| `fischai record` | Save game-window frames while you play (detector material) |
| `fischai route` | Record the spawn-to-fishing-spot walk used after a rejoin |

## Configuration

`~/.config/fischai/config.toml`, created with defaults on first run and
rewritten by the settings window. Logs and state live under
`~/.local/state/fischai/`.

## License

No license file yet — all rights reserved by the author until one is added.
