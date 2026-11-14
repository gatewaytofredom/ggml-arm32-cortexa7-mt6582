# Optimizations: MediaTek MT6582 (ARM Cortex-A7)

Inventory of every code, compiler, and runtime optimization in this repository, paired with the physical Cortex-A7 / MT6582 hardware constraint behind it.

## Target Silicon Profile
* **Core:** 2x active ARM Cortex-A7 r0p3 (ARMv7-A, 8-stage in-order dual-issue, `1.300 GHz` boost / `598 MHz` idle).
* **SIMD / FP:** 64-bit NEON pipeline (128-bit ops split over 2 cycles) + VFPv4 (`vcvtb.f16.f32`).
* **Memory (512 MB RAM):** Measured `1.95 GB/s` single-core DRAM read, `3.39 GB/s` dual-core aggregate with `pld`, `9.38 GB/s` L1, `4.12 GB/s` L2.
* **Bottleneck:** At 1.50 t/s (`Qwen3-0.6B IQ2_XXS`), DRAM traffic is ~341 MB/s (10.1% of 3.39 GB/s), proving generation is **NEON compute/latency-bound**, not memory-bound.

---

## Optimization Inventory

| # | Optimization | Hardware Constraint | Measured Impact | Location |
| :-: | :--- | :--- | :--- | :--- |
| 1 | Dual-accumulator ping-pong `VMLAL` (`IQ2_XXS`) | 4-cycle NEON accumulator bypass & 8-cycle non-acc latency | Eliminates 180 insns/superblock & halves stack spills | [ggml_vec_dot_iq2_xxs_q8_K_a7.S](kernels/arm/ggml_vec_dot_iq2_xxs_q8_K_a7.S), [quants.c](patches/llama.cpp-3ba1ce0.patch) |
| 2 | Integer scale accumulation + deferred FP | Expensive per-subblock VFP conversions | Moves subblock scales to single-cycle integer `mla` | [quants.c](patches/llama.cpp-3ba1ce0.patch) |
| 3 | `Q5_K` `VSLI` unpacking + 2-stage `VMLAL` | Extra mask registers causing `vpush {d8,d9}` spills | Eliminates `m4b`/`mtwo` masks & callee-saved spills | [quants.c](patches/llama.cpp-3ba1ce0.patch) |
| 4 | `1x4` `MUL_MAT` batching (`Q4_0`, `IQ2_XXS`) | Redundant weight loads across prompt columns | Reuses loaded weight blocks across 4 prompt rows | [ggml-cpu.c](patches/llama.cpp-3ba1ce0.patch) |
| 5 | NEON `RMSNorm` (`vmlaq_f32` dual-acc) | Scalar VFP double-precision `vcvt.f64` / `vadd.f64` | **3.53x** faster isolated RMSNorm | [ops.cpp](patches/llama.cpp-3ba1ce0.patch) |
| 6 | VFPv4 `vcvtb` FP16 conversion + NEON `q8_K` quant | Software FP16 table/bitops & scalar `q8_K` quant | Hardware 1-insn FP16<->FP32 + vectorized `q8_K` | [simd-mappings.h](patches/llama.cpp-3ba1ce0.patch), [quants.c](patches/llama.cpp-3ba1ce0.patch) |
| 7 | `-fomit-frame-pointer -marm -fno-stack-protector` | 32-bit ARM GPR scarcity (`r11` wasted on frame pointer) | Frees `r11` for inner-loop pointers & bitfields | [scripts/build.sh](scripts/build.sh) |
| 8 | `--poll 0` + `-lm mmap` + `mtk-perfservice` | `hotplug` governor 598 MHz idle vs 512 MB `kswapd0` thrash | Boosts to 1.300 GHz on demand; zero `kswapd0` IOW | [scripts/run.sh](scripts/run.sh) |

---

## Detailed Reasoning

### 1. Dual-Accumulator Ping-Pong `VMLAL` (`IQ2_XXS`)
* **What changed:** Replaced `ggml_vdotq_s32` (22 NEON instructions + 4 GPR transfers + 4 stack spills per 64 weights) with direct 64-bit `D`-register loads, single-cycle `vmul_s8`, and two interleaved 16-bit `vmlal_s8` accumulators (`acc0`, `acc1`).
* **Hardware reason:** Cortex-A7 has a 4-cycle accumulator bypass latency and 8-cycle read latency before `vpaddlq_s16`. Alternating `acc0` and `acc1` hides both latencies with zero stall bubbles.
* **Measured effect:** Cuts 45 instructions per 64 weights (180 per superblock), raising `Qwen3-0.6B IQ2_XXS` from `1.79 -> 2.21 t/s` (`pp16`) and `1.24 -> 1.50 t/s` (`tg8`).

### 2. Integer Scale Accumulation
* **What changed:** Moved subblock scale multiplication into 32-bit integer `mla` lanes (`isum += ... * (1 + 2 * sc)`), converting to `float` once per 256-weight superblock.
* **Hardware reason:** Avoids repeated VFP int-to-float conversions inside the tight subblock loop while matching reference integer accumulation.

### 3. `Q5_K` `VSLI` Unpacking
* **What changed:** Replaced mask-and-shift bit unpacking with single-cycle NEON `vsliq_n_u8` and 2-stage `vmlal_s8` accumulation.
* **Hardware reason:** Frees vector registers so GCC/Clang no longer spills `d8`/`d9` on entry/exit.

### 4. `1x4` `MUL_MAT` Batching (`Q4_0`, `IQ2_XXS`)
* **What changed:** Dispatched 4-column prompt batches to `ggml_vec_dot_q4_0_q8_0_1x4` and `ggml_vec_dot_iq2_xxs_q8_K_1x4` in `ggml-cpu.c`.
* **Hardware reason:** Amortizes weight unpacking and L1/L2 weight loads across 4 activations during prompt processing.

### 5. NEON `RMSNorm` Vectorization
* **What changed:** Replaced scalar `vcvt.f64` / `vadd.f64` in `ggml_compute_forward_rms_norm_f32` with dual-accumulator 128-bit NEON `vmlaq_f32` and `vmulq_f32`.
* **Hardware reason:** Double-precision VFP throughput is heavily serialized on Cortex-A7.
* **Measured effect:** **3.53x** faster isolated RMSNorm.

### 6. VFPv4 `vcvtb` FP16 & NEON `quantize_row_q8_K`
* **What changed:** Wired `vcvtb.f32.f16` / `vcvtb.f16.f32` in `simd-mappings.h` and vectorized `quantize_row_q8_K` with `vminq_f32`/`vmaxq_f32`/`vcvtnq_s32_f32`.
* **Hardware reason:** Uses native Cortex-A7 VFPv4 half-precision instructions and NEON `vpaddlq` horizontal sums.

### 7. `-fomit-frame-pointer -marm` & Android 4.4.2 Shim
* **What changed:** Built static `armeabi-v7a` ELF binaries with `-marm -mcpu=cortex-a7 -mfpu=neon-vfpv4 -fomit-frame-pointer -fno-stack-protector` and `kernels/arm/android_shim.h` (`posix_madvise` -> `madvise`).
* **Hardware reason:** Releases `r11` on register-starved 32-bit ARM and provides `posix_madvise` missing from Android 4.4.2 (API 19).

### 8. Governor & 512 MB RAM Hygiene (`--poll 0`, `-lm mmap`)
* **What changed:** `scripts/run.sh` triggers `mtk-perfservice` scenario 5 (`1.300 GHz`) and runs `llama-server` with `--poll 0 -lm mmap`.
* **Hardware reason:** Default `--poll 50` keeps cores busy-spinning so the `hotplug` governor never drops to `598 MHz`, while `mlock` on 512 MB RAM starves page cache and triggers 49% `kswapd0` IO wait.
