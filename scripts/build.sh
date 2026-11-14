#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/build}"
UPSTREAM_SRC="${UPSTREAM_SRC:-$WORK_DIR/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$WORK_DIR/cmake-build}"
UPSTREAM_COMMIT="3ba1ce0e28d991800e1585b7d3533f77213939d6"
JOBS="${JOBS:-$(nproc)}"

NDK_HOME="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-/home/gateway/Android/Sdk/ndk/28.2.13676358}}"
TOOLCHAIN_FILE="$NDK_HOME/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN_FILE" ]]; then
    echo "ERROR: Android NDK toolchain not found at $TOOLCHAIN_FILE (set ANDROID_NDK_HOME)." >&2
    exit 1
fi

ARCH_FLAGS="-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp -mcpu=cortex-a7 -marm -fomit-frame-pointer -fno-stack-protector -include $ROOT_DIR/kernels/arm/android_shim.h"

mkdir -p "$WORK_DIR"

if [[ ! -d "$UPSTREAM_SRC/.git" ]]; then
    echo "==> Cloning upstream llama.cpp @ $UPSTREAM_COMMIT into $UPSTREAM_SRC"
    git clone --no-checkout https://github.com/ggml-org/llama.cpp.git "$UPSTREAM_SRC"
    git -C "$UPSTREAM_SRC" fetch --depth 1 origin "$UPSTREAM_COMMIT"
    git -C "$UPSTREAM_SRC" checkout "$UPSTREAM_COMMIT"
else
    echo "==> Resetting existing upstream tree $UPSTREAM_SRC to clean state"
    git -C "$UPSTREAM_SRC" reset --hard HEAD
    git -C "$UPSTREAM_SRC" clean -fd
fi

echo "==> Installing Cortex-A7 ARMv7 NEON assembly kernel"
cp "$ROOT_DIR/kernels/arm/ggml_vec_dot_iq2_xxs_q8_K_a7.S" \
   "$UPSTREAM_SRC/ggml/src/ggml-cpu/arch/arm/"

echo "==> Applying patch $ROOT_DIR/patches/llama.cpp-3ba1ce0.patch"
git -C "$UPSTREAM_SRC" apply "$ROOT_DIR/patches/llama.cpp-3ba1ce0.patch"

echo "==> Configuring CMake (armeabi-v7a, Cortex-A7 NEON-VFPv4, static ELF)"
cmake -S "$UPSTREAM_SRC" -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DANDROID_ABI=armeabi-v7a \
    -DANDROID_PLATFORM=android-21 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_FLAGS="$ARCH_FLAGS" \
    -DCMAKE_CXX_FLAGS="$ARCH_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="-static" \
    -DLLAMA_CURL=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DGGML_BUILD_EXAMPLES=OFF \
    -DGGML_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_COMMON=ON

echo "==> Building llama-bench, llama-server, llama-completion (-j$JOBS)"
cmake --build "$BUILD_DIR" --target llama-bench llama-server llama-completion -j "$JOBS"

echo "==> Verifying 32-bit ARM EABI5 static ELF"
file "$BUILD_DIR/bin/llama-bench" | grep -q "ELF 32-bit LSB executable, ARM"
echo "==> Build complete: $BUILD_DIR/bin"
