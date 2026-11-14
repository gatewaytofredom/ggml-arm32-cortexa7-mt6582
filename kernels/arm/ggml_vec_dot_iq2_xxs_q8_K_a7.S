    .syntax unified
    .arch armv7-a
    .fpu neon-vfpv4
    .arm
    .text

    .global ggml_vec_dot_iq2_xxs_q8_K_a7
    .type ggml_vec_dot_iq2_xxs_q8_K_a7, %function
    .align 4
@ void ggml_vec_dot_iq2_xxs_q8_K_a7(int nb, float * s, const void * x, const void * y,
@                                     const uint64_t * grid, const uint64_t * signs64)
@ r0 = nb, r1 = s, r2 = x, r3 = y
@ [sp, #0] = grid, [sp, #4] = signs64
ggml_vec_dot_iq2_xxs_q8_K_a7:
    push    {r4, r5, r6, r7, r8, r9, r10, r11, lr}
    @ 9 regs pushed = 36 bytes. Stack args at [sp, #36] and [sp, #40].
    vpush   {d8, d9}                @ Save callee-saved d8-d9 (s16-s19) for outer loop state
    @ 2 D-regs pushed = 16 bytes. Total offset = 52 bytes.
    ldr     r4, [sp, #52]           @ grid
    ldr     r5, [sp, #56]           @ signs64
    sub     sp, sp, #16
    str     r0, [sp, #0]            @ nb
    str     r1, [sp, #4]            @ s
    str     r4, [sp, #8]            @ grid
    str     r5, [sp, #12]           @ signs64

    cmp     r0, #0
    vsub.f32 s16, s16, s16          @ sumf = 0.0f
    ble     .Ldone

    @ Use r0 = x, r1 = y, r2 = grid, r3 = signs64 across inner loop
    mov     r0, r2                  @ x
    mov     r1, r3                  @ y
    mov     r2, r4                  @ grid
    mov     r3, r5                  @ signs64
    vmov.f32 s18, #5.000000e-01     @ 0.5f

.Louter_nb:
    ldrh    r10, [r0], #2           @ x[i].d (fp16), x += 2 -> x[i].qs
    vldr    s17, [r1]               @ y[i].d (f32)
    add     r1, r1, #4              @ y += 4 -> y[i].qs
    vmov    s19, r10
    vcvtb.f32.f16 s19, s19          @ fp16_to_fp32(x[i].d)
    vmul.f32 s17, s19, s17          @ d = fp16_to_fp32(x[i].d) * y[i].d

    mov     r8, #0                  @ isum = 0
    mov     r9, #4                  @ 4 iterations of 64 weights

    ldr     r4, [r0]                @ w0
    ldr     r5, [r0, #4]            @ w1
    ldr     r6, [r0, #8]            @ w2
    ldr     r7, [r0, #12]           @ w3
    add     r0, r0, #16
    vld1.8  {d4, d5, d6, d7}, [r1]! @ q8[0..31]
    vld1.8  {d24, d25, d26, d27}, [r1]! @ q8[32..63]

.Lloop64:
    @ --- Chunk 0 ---
    uxtb    r10, r4
    ubfx    r11, r5, #0, #7
    add     r10, r2, r10, lsl #3
    add     r11, r3, r11, lsl #3
    vldr    d16, [r10]
    vldr    d17, [r11]

    uxtb    r10, r6
    ubfx    r11, r7, #0, #7
    add     r10, r2, r10, lsl #3
    add     r11, r3, r11, lsl #3
    vldr    d18, [r10]
    vldr    d19, [r11]

    vmul.i8 d16, d16, d17

    @ --- Chunk 1 ---
    ubfx    r10, r4, #8, #8
    ubfx    r11, r5, #7, #7
    add     r10, r2, r10, lsl #3
    add     r11, r3, r11, lsl #3
    vldr    d20, [r10]
    vldr    d21, [r11]

    vmul.i8 d18, d18, d19
    vmull.s8 q0, d16, d4

    ubfx    r10, r6, #8, #8
    ubfx    r11, r7, #7, #7
    add     r10, r2, r10, lsl #3
    add     r11, r3, r11, lsl #3
    vldr    d22, [r10]
    vldr    d23, [r11]

    vmull.s8 q1, d18, d24
    vmul.i8 d20, d20, d21

    @ --- Chunk 2 ---
    ubfx    r10, r4, #16, #8
    ubfx    r11, r5, #14, #7
    add     r10, r2, r10, lsl #3
    add     r11, r3, r11, lsl #3
    vldr    d16, [r10]
    vldr    d17, [r11]

    vmul.i8 d22, d22, d23
    vmlal.s8 q0, d20, d5

    ubfx    r10, r6, #16, #8
    ubfx    r11, r7, #14, #7
    add     r10, r2, r10, lsl #3
    add     r11, r3, r11, lsl #3
    vldr    d18, [r10]
    vldr    d19, [r11]

    vmlal.s8 q1, d22, d25
    vmul.i8 d16, d16, d17

    @ --- Chunk 3 ---
    ubfx    r10, r4, #24, #8
    ubfx    r11, r5, #21, #7
    add     r10, r2, r10, lsl #3
    add     r11, r3, r11, lsl #3
    vldr    d20, [r10]
    vldr    d21, [r11]

    vmul.i8 d18, d18, d19
    vmlal.s8 q0, d16, d6

    ubfx    r10, r6, #24, #8
    ubfx    r11, r7, #21, #7
    add     r10, r2, r10, lsl #3
    add     r11, r3, r11, lsl #3
    vldr    d22, [r10]
    vldr    d23, [r11]

    vmlal.s8 q1, d18, d26
    vmul.i8 d20, d20, d21

    lsr     r4, r5, #28
    lsr     r6, r7, #28
    mov     r12, #1
    add     r4, r12, r4, lsl #1
    add     r6, r12, r6, lsl #1

    vmul.i8 d22, d22, d23
    vmlal.s8 q0, d20, d7
    vmlal.s8 q1, d22, d27

    vpaddl.s16 q0, q0
    vpaddl.s16 q1, q1
    vpadd.s32 d0, d0, d1
    vpadd.s32 d2, d2, d3
    vpadd.s32 d0, d0, d2

    subs    r9, r9, #1
    beq     .Llast_iter

    ldr     r10, [r0]
    ldr     r11, [r0, #4]
    ldr     r12, [r0, #8]
    ldr     lr,  [r0, #12]
    add     r0, r0, #16
    vld1.8  {d4, d5, d6, d7}, [r1]!
    vld1.8  {d24, d25, d26, d27}, [r1]!

    vmov    r5, r7, d0
    mla     r8, r5, r4, r8
    mla     r8, r7, r6, r8

    mov     r4, r10
    mov     r5, r11
    mov     r6, r12
    mov     r7, lr
    b       .Lloop64

.Llast_iter:
    vmov    r5, r7, d0
    mla     r8, r5, r4, r8
    mla     r8, r7, r6, r8

    @ End of superblock: sumf += d * ((float)isum * 0.5f)
    vmov    s19, r8
    vcvt.f32.s32 s19, s19
    add     r1, r1, #32             @ skip y[i].bsums[16] -> y[i+1]
    vmul.f32 s19, s19, s18          @ (float)isum * 0.5f
    ldr     r9, [sp, #0]            @ nb
    vmul.f32 s17, s17, s19          @ d * ((float)isum * 0.5f)
    subs    r9, r9, #1
    vadd.f32 s16, s16, s17          @ sumf += ...
    str     r9, [sp, #0]
    bne     .Louter_nb

.Ldone:
    vmov.f32 s18, #2.500000e-01     @ 0.25f
    ldr     r1, [sp, #4]            @ s
    vmul.f32 s16, s16, s18          @ 0.25f * sumf
    vstr    s16, [r1]

    add     sp, sp, #16
    vpop    {d8, d9}
    pop     {r4, r5, r6, r7, r8, r9, r10, r11, pc}
    .size ggml_vec_dot_iq2_xxs_q8_K_a7, .-ggml_vec_dot_iq2_xxs_q8_K_a7
