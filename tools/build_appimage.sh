#!/usr/bin/env bash
# Build FischAI-x86_64.AppImage in ~/appimage-build (output: ~/FischAI-x86_64.AppImage).
# Bundles: standalone CPython 3.14, runtime deps (no torch/pnnx — only the
# model converter needs those), the converted reel model + templates from
# ~/.local/share/fischai, and grim + tesseract with their libraries.
# The friend still needs: Hyprland, Sober, and /dev/uinput + input-group rights.
set -euo pipefail

PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.14.10%2B20261026-x86_64-unknown-linux-gnu-install_only.tar.gz"
TOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
BUILD="$HOME/appimage-build"
OUT="$HOME/FischAI-x86_64.AppImage"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$BUILD/dl"
cd "$BUILD"
[ -f dl/python.tar.gz ] || curl -sL -o dl/python.tar.gz "$PY_URL"
[ -f dl/appimagetool.AppImage ] || curl -sL -o dl/appimagetool.AppImage "$TOOL_URL"
chmod +x dl/appimagetool.AppImage

rm -rf AppDir
mkdir -p AppDir/usr
tar xzf dl/python.tar.gz -C AppDir/usr
PY=AppDir/usr/python/bin/python3
$PY -m pip install --no-cache-dir numpy "opencv-python-headless>=4.10" "onnxruntime>=1.20" \
    "ncnn>=1.0.20250503" "pyside6-essentials>=6.8"
$PY -m pip install --no-cache-dir --no-deps "$REPO"

mkdir -p AppDir/usr/share/fischai-seed AppDir/usr/bin AppDir/usr/lib AppDir/usr/share/tessdata
cp -r "$HOME/.local/share/fischai/model" "$HOME/.local/share/fischai/templates" AppDir/usr/share/fischai-seed/
cp /usr/bin/grim /usr/bin/tesseract AppDir/usr/bin/
cp /usr/share/tessdata/eng.traineddata AppDir/usr/share/tessdata/
for bin in /usr/bin/grim /usr/bin/tesseract; do ldd "$bin" | awk '{print $3}' | grep '^/'; done \
  | sort -u | while read -r lib; do
    case "$lib" in
      */libc.so.6|*/libm.so.6|*/libpthread.so.0|*/libdl.so.2|*/librt.so.1|*/libresolv.so.2|*/libgcc_s.so.1|*/libstdc++.so.6|*/ld-linux*) ;;
      *) cp -L "$lib" AppDir/usr/lib/ ;;
    esac
  done

cp "$REPO/tools/AppRun" AppDir/AppRun
cp "$REPO/tools/fischai.desktop" AppDir/fischai.desktop
cp "$REPO/tools/fischai.png" AppDir/fischai.png
chmod +x AppDir/AppRun

ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 ./dl/appimagetool.AppImage AppDir "$OUT"
echo "built: $OUT"
