Prebuilt static 32-bit ARMv7 (`armeabi-v7a`, Android API 19+ compatible) binaries (`llama-server`, `llama-completion`, `llama-bench`) and helper scripts (`run.sh`, `verify.sh`) tuned for the MediaTek MT6582 (2x ARM Cortex-A7 @ 1.300 GHz).

### Highlights
- **Qwen3-0.6B (UD-IQ2_XXS, 217.55 MiB):** `2.21 t/s` prompt (`+23.5%`), `1.50 t/s` decode (`+21.0%`) at 1.300 GHz
- **NEON RMSNorm:** `3.53x` faster isolated RMSNorm (`vmlaq_f32`)
- **Zero dynamic dependencies:** Static ELF binaries with `android_shim.h` (`posix_madvise` -> `madvise`) for Android 4.4.2+

See `OPTIMIZATIONS.md` for the full microarchitectural breakdown.
