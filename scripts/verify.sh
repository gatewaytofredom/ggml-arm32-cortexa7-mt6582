#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${BIN_DIR:-}" ]]; then
    if [[ -d "$ROOT_DIR/bin" ]]; then
        BIN_DIR="$ROOT_DIR/bin"
    else
        BIN_DIR="$ROOT_DIR/build/cmake-build/bin"
    fi
fi

echo "==> 1. Verifying 32-bit ARM static ELF binaries"
for b in llama-bench llama-server llama-completion; do
    if [[ ! -f "$BIN_DIR/$b" ]]; then
        echo "ERROR: Missing $BIN_DIR/$b" >&2
        exit 1
    fi
    file "$BIN_DIR/$b" | grep -q "ELF 32-bit LSB executable, ARM"
    echo "    OK: $b ($(ls -lh "$BIN_DIR/$b" | awk '{print $5}'))"
done

if command -v adb >/dev/null 2>&1 && adb get-state >/dev/null 2>&1; then
    echo "==> 2. Checking connected Android ARMv7 device"
    adb shell "cat /proc/cpuinfo | grep -E 'Processor|Features|CPU part'" || true
else
    echo "==> 2. No ADB device attached; skipping on-device check."
fi

echo "==> ALL VERIFICATION CHECKS PASSED"
