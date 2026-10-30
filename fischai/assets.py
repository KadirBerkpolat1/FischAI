"""
One-time setup: pull the reel model and the vision templates out of an
ExoPanda Fisch Windows module (fisch-<version>.zip or the ExoPanda_Fischer.exe
inside it) into the data directory, and convert the model for ncnn/Vulkan.

The exe embeds every template as a raw PNG preceded by its relative path
(e.g. "shake\\shake_word_h34.png") and the model as a raw ONNX protobuf.
These files are ExoPanda's; they stay in the user's data directory and are
never part of this repository.
"""

import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
ONNX_HEADER = re.compile(rb"\x08[\x03-\x0a]\x12\x07pytorch")
TEMPLATE_NAME = re.compile(rb"([a-z0-9_\-]+(?:\\[a-z0-9_\-]+)*\.png)")
NAME_TO_PNG_MAX_GAP = 64       # a template's path string sits right before its PNG bytes


def _read_exe(source):
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            exe = next((n for n in zf.namelist() if n.lower().endswith("_fischer.exe")), None)
            if exe is None:
                raise SystemExit(f"{source} has no *_Fischer.exe (the macOS module does not embed the templates)")
            return zf.read(exe)
    return source.read_bytes()


def _read_varint(data, pos):
    value = shift = 0
    while True:
        byte = data[pos]
        value |= (byte & 0x7F) << shift
        pos += 1
        shift += 7
        if byte < 0x80:
            return value, pos


def _onnx_end(data, start):
    """Walk top-level ModelProto fields until the bytes stop parsing as one."""
    pos = start
    while pos < len(data):
        key, after_key = _read_varint(data, pos)
        field, wire = key >> 3, key & 7
        if not 1 <= field <= 25 or wire not in (0, 2):
            break
        if wire == 0:
            _, nxt = _read_varint(data, after_key)
        else:
            length, after_len = _read_varint(data, after_key)
            nxt = after_len + length
        if nxt > len(data):
            break
        pos = nxt
    return pos


def _png_end(data, start):
    pos = start + len(PNG_MAGIC)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        chunk = data[pos + 4:pos + 8]
        pos += 12 + length
        if chunk == b"IEND":
            return pos
    raise ValueError(f"unterminated PNG at {start}")


def _write_model_meta(model_dir):
    import onnxruntime as ort
    session = ort.InferenceSession(str(model_dir / "best.onnx"), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    names = {int(k): v for k, v in _parse_names(meta.get("names", "")).items()}
    shape = session.get_inputs()[0].shape
    if shape[2:] != [640, 640] or sorted(names.values()) != ["catch-bar", "fish-line"]:
        raise SystemExit(f"unexpected model: input {shape}, classes {names}")
    (model_dir / "model.json").write_text(json.dumps({"names": names, "date": meta.get("date")}, indent=1))
    return names


def _parse_names(raw):
    # Ultralytics stores names as a Python dict literal: {0: 'fish-line', 1: 'catch-bar'}
    import ast
    return ast.literal_eval(raw) if raw else {}


def _convert_ncnn(model_dir, log):
    pnnx = shutil.which("pnnx") or str(Path(sys.executable).parent / "pnnx")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "best.onnx"
        shutil.copy(model_dir / "best.onnx", work)
        out = subprocess.run([pnnx, str(work), "inputshape=[1,3,640,640]"], cwd=tmp, capture_output=True, text=True)
        param, weights = Path(tmp) / "best.ncnn.param", Path(tmp) / "best.ncnn.bin"
        if out.returncode != 0 or not param.is_file():
            log(f"ncnn conversion failed, the CPU backend will be used:\n{out.stderr[-2000:]}")
            return False
        shutil.move(param, model_dir / "best.ncnn.param")
        shutil.move(weights, model_dir / "best.ncnn.bin")
    return True


def install(source, data_dir, log=print):
    data = _read_exe(Path(source))
    model_dir = data_dir / "model"
    template_dir = data_dir / "templates"
    model_dir.mkdir(parents=True, exist_ok=True)

    match = ONNX_HEADER.search(data)
    if match is None:
        raise SystemExit("no ONNX model found in the module")
    (model_dir / "best.onnx").write_bytes(data[match.start():_onnx_end(data, match.start())])
    names = _write_model_meta(model_dir)
    log(f"model: {model_dir / 'best.onnx'} classes {names}")

    if template_dir.exists():
        shutil.rmtree(template_dir)
    count = 0
    for png in re.finditer(re.escape(PNG_MAGIC), data):
        window = data[max(0, png.start() - NAME_TO_PNG_MAX_GAP):png.start()]
        found = TEMPLATE_NAME.findall(window)
        if not found:
            continue
        rel = found[-1].decode().replace("\\", "/")
        if "/" not in rel:
            continue  # app icons, not vision templates
        out = template_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data[png.start():_png_end(data, png.start())])
        count += 1
    if count == 0:
        raise SystemExit("no vision templates found in the module")
    log(f"templates: {count} files in {template_dir}")

    if _convert_ncnn(model_dir, log):
        log(f"ncnn model: {model_dir / 'best.ncnn.param'}")
