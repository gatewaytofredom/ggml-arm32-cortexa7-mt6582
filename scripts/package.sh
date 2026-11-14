#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${BIN_DIR:-$ROOT_DIR/build/cmake-build/bin}"
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist}"
PKG_NAME="ggml-arm32-cortexa7-mt6582-android-armv7"
STAGE_DIR="$DIST_DIR/$PKG_NAME"

NDK_HOME="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-/home/gateway/Android/Sdk/ndk/28.2.13676358}}"
STRIP_BIN="$(command -v llvm-strip || echo "$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip")"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/bin"

cp "$BIN_DIR/llama-server" "$BIN_DIR/llama-completion" "$BIN_DIR/llama-bench" "$STAGE_DIR/bin/"
"$STRIP_BIN" -s "$STAGE_DIR/bin/llama-server" "$STAGE_DIR/bin/llama-completion" "$STAGE_DIR/bin/llama-bench"

cp "$ROOT_DIR/scripts/run.sh" "$ROOT_DIR/scripts/verify.sh" "$ROOT_DIR/README.md" "$ROOT_DIR/OPTIMIZATIONS.md" "$ROOT_DIR/LICENSE" "$STAGE_DIR/"
chmod +x "$STAGE_DIR/run.sh" "$STAGE_DIR/verify.sh"

tar -C "$DIST_DIR" -czf "$DIST_DIR/${PKG_NAME}.tar.gz" "$PKG_NAME"
echo "==> Packaged $DIST_DIR/${PKG_NAME}.tar.gz"
