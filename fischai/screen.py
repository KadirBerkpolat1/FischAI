"""Screen capture through `grim` (wlr-screencopy), in Hyprland layout coordinates."""

import subprocess

import numpy as np


def _parse_ppm(data):
    if data[:2] != b"P6":
        raise ValueError("grim did not return a binary PPM image")
    fields = []
    i = 2
    while len(fields) < 3:
        while data[i:i + 1].isspace():
            i += 1
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        fields.append(int(data[i:j]))
        i = j
    width, height, maxval = fields
    if maxval != 255:
        raise ValueError(f"Unsupported PPM maxval {maxval}")
    return np.frombuffer(data, dtype=np.uint8, count=width * height * 3, offset=i + 1).reshape(height, width, 3)


def grab(x, y, width, height):
    """(height, width, 3) uint8 RGB of a layout-coordinate rectangle. `-s 1`
    keeps one image pixel per logical pixel on scaled outputs; the cursor is
    not captured."""
    out = subprocess.run(
        ["grim", "-s", "1", "-t", "ppm", "-g", f"{int(x)},{int(y)} {int(width)}x{int(height)}", "-"],
        capture_output=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"grim failed: {out.stderr.decode(errors='replace').strip()}")
    return _parse_ppm(out.stdout)
