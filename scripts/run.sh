#!/usr/bin/env bash
set -euo pipefail

# Run llama-server on an Android MT6582 / Cortex-A7 device (e.g. Pixi3) via ADB or on-device shell.
# Usage: ./scripts/run.sh -m /data/local/tmp/llm/Qwen3-0.6B-UD-IQ2_XXS.gguf --port 8081

if [[ -f /system/bin/service ]]; then
    # Running directly on the Android device
    /system/bin/service call mtk-perfservice 3 i32 5 i32 120000 >/dev/null 2>&1 || true
    exec ./bin/llama-server -t 2 --poll 0 -lm mmap "$@"
else
    # Running from host via adb
    adb shell "service call mtk-perfservice 3 i32 5 i32 120000" >/dev/null 2>&1 || true
    exec adb shell "/data/local/tmp/llm/llama-server -t 2 --poll 0 -lm mmap $*"
fi
