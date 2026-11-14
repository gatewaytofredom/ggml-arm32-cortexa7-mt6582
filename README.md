# ggml-arm32-cortexa7-mt6582

[![Discord](https://img.shields.io/badge/Discord-Gateway's%20Experiments-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/pD8wR8YVhZ)

Hand-tuned 32-bit **ARMv7-A NEON + VFPv4** `ggml-cpu` and `llama.cpp` inference kernels for the **MediaTek MT6582** (2x active ARM Cortex-A7 @ 1.30 GHz, 512 MB RAM, Android 4.4.2+).

> Primarily an AI-completed project. Contributors: `gateway` (`gateway@laindex.net`), Gemini 4 Hydrogen.
> Hardware requests, `llama-bench` traces & kernel talk: [Gateway's Experiments Discord](https://discord.gg/pD8wR8YVhZ).

Up to **+23.5% prompt processing** (`pp16`) and **+21.0% decode throughput** (`tg8`) over upstream `llama.cpp` (`3ba1ce0`) on a 2014 dual-core Alcatel Pixi3 while matching reference dot-product output.

## Benchmarks (Alcatel Pixi3 MT6582, `1.300 GHz`, `-t 2`)

| Model | Build | `pp16` (t/s) | `tg8` (t/s) |
| :--- | :--- | :---: | :---: |
| Qwen3-0.6B (`UD-IQ2_XXS`, 217.55 MiB) | Upstream | 1.79 ± 0.02 | 1.24 ± 0.03 |
| Qwen3-0.6B (`UD-IQ2_XXS`, 217.55 MiB) | Cortex-A7 | **2.21 ± 0.01** | **1.50 ± 0.00** |

* `Upstream`: out-of-the-box `llama.cpp@3ba1ce0` compiled for `armeabi-v7a`. `Cortex-A7`: this repo with `mtk-perfservice` 1.300 GHz boost and `-lm mmap+mlock` (or `-lm mmap` for `llama-server`).
* Compute-bound on Cortex-A7: at 1.50 t/s, `217.55 MiB` weights stream at ~341 MB/s, only 10.1% of the measured **3.39 GB/s** dual-core DRAM read bandwidth.

## Usage

### Build and verify (Linux x86-64 host with Android NDK)
```bash
git clone https://github.com/gatewaytofredom/ggml-arm32-cortexa7-mt6582.git
cd ggml-arm32-cortexa7-mt6582
./scripts/build.sh
./scripts/verify.sh
```

### Deploy and run on device (ADB)
```bash
./scripts/package.sh
adb push dist/ggml-arm32-cortexa7-mt6582-android-armv7/bin/llama-server /data/local/tmp/llm/
./scripts/run.sh -m /data/local/tmp/llm/Qwen3-0.6B-UD-IQ2_XXS.gguf --port 8081
```

## Changes Included
See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for the full optimization inventory and hardware reasoning behind each design choice.

* `kernels/arm/ggml_vec_dot_iq2_xxs_q8_K_a7.S`: Hand-scheduled Cortex-A7 ARMv7 NEON `IQ2_XXS x Q8_K` assembly kernel.
* `kernels/arm/android_shim.h`: `posix_madvise` -> `madvise` compatibility shim for Android 4.4.2 (API 19).
* `patches/llama.cpp-3ba1ce0.patch`: Dual-accumulator ping-pong `VMLAL` (`IQ2_XXS`/`IQ3_XXS`), `Q5_K` `VSLI` unpacking, `1x4` `MUL_MAT` batching (`Q4_0`/`IQ2_XXS`), NEON `quantize_row_q8_K`, NEON `RMSNorm`, and VFPv4 hardware `vcvtb` FP16 conversion.

## Target Assumptions & Compatibility
* Binaries are statically linked `armeabi-v7a` ELFs (`android-21` NDK with `kernels/arm/android_shim.h`) and require **NEON + VFPv4** (`-mfpu=neon-vfpv4`); non-NEON or non-VFPv4 ARMv7 CPUs will trap with `SIGILL`.
* `scripts/run.sh` invokes MediaTek's `service call mtk-perfservice 3 i32 5 i32 120000` to hold `1.300 GHz` boost; non-MediaTek devices ignore or skip that call.

## License
MIT License (preserving upstream `ggml` copyright).
