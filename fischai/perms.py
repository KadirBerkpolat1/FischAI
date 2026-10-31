"""One-time permission setup.

The hotkey reads the real keyboards from /dev/input and clicks/keys go out
through a virtual device on /dev/uinput; both need access a fresh user does
not have. `fix()` sets it up with one pkexec run (the desktop's graphical
root prompt): the user joins the input group, a udev rule with the uaccess
tag makes it stick across reboots, and ACLs on the live device nodes make
it work in this session without logging out and back in.
"""

import os
import shutil
import subprocess
import tempfile

RULE_PATH = "/etc/udev/rules.d/99-fischai.rules"
RULES = ('KERNEL=="uinput", GROUP="input", MODE="0660", TAG+="uaccess"\n'
         'SUBSYSTEM=="input", KERNEL=="event*", GROUP="input", MODE="0660", TAG+="uaccess"\n')


def missing():
    """What is not set up yet, in words ([] when everything is in place)."""
    from .inputdev import _keyboard_event_paths
    out = []
    if not os.access("/dev/uinput", os.W_OK):
        out.append("/dev/uinput yazılamıyor (tıklama ve tuşlar için)")
    if not any(os.access(p, os.R_OK) for p in _keyboard_event_paths()):
        out.append("klavye okunamıyor (başlat/durdur kısayolu için)")
    return out


def script(user):
    """The shell script pkexec runs as root."""
    return f"""#!/bin/sh
set -e
U="$1"
usermod -aG input "$U"
cat > {RULE_PATH} <<'EOF'
{RULES}EOF
udevadm control --reload-rules
udevadm trigger
setfacl -m "u:$U:rw" /dev/uinput 2>/dev/null || true
for f in /dev/input/event*; do setfacl -m "u:$U:r" "$f" 2>/dev/null || true; done
"""


def fix(user):
    """Set the permissions up as root. Returns (ok, message)."""
    if not shutil.which("pkexec"):
        return False, "pkexec bulunamadı; doctor çıktısındaki adımları elle uygula."
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write(script(user))
            path = f.name
        os.chmod(path, 0o755)
        run = subprocess.run(["pkexec", "/bin/sh", path, user], capture_output=True, text=True, check=False)
    finally:
        if path:
            os.unlink(path)
    if run.returncode == 0:
        return True, ""
    detail = (run.stderr or run.stdout).strip()
    return False, f"Kurulum tamamlanamadı (kod {run.returncode}){': ' + detail if detail else ''}"
