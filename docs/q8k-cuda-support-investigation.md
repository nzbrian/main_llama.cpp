# Q8_K CUDA support — investigation and port plan

Status: **investigation done, not yet implemented.**
Date: 2026-07-19. Fork: `nzbrian/main_llama.cpp` @ `9b61137df` (master, synced upstream).

## TL;DR

**Yes — Q8_K can be added to CUDA the same way Q6_K was added, and it is
simpler than Q6_K.** `block_q8_K` is a *plain* int8 type (no bit-packing,
single f32 scale per 256-block), so its CUDA math is essentially Q8_0's.
Phase 1 (the mmvq token-generation path + convert) is ~150 lines across 5
files, all mechanical pattern copies. A single patch in `ggml/src/ggml-cuda/`
also fixes the known **HIP SIGABRT** (the HIP build compiles the same `.cu`
sources through `ggml-cuda/vendors/hip.h`).

## What Q8_K is in this fork

`GGML_TYPE_Q8_K` is the classical ggml intermediate type, promoted to a
first-class storage type by the fork:

```c
// ggml/src/ggml-common.h:370  (static_assert at :376)
typedef struct {
    float   d;               // delta (f32 — unlike every other quant's f16)
    int8_t  qs[QK_K];        // 256 plain int8 quants — NO bit-packing
    int16_t bsums[QK_K/16];  // 16 group sums (16 quants each)
} block_q8_K;                // 4 + 256 + 32 = 292 B / 256 = 9.125 bpw
```

Dequant is a single multiply: `y[i] = d * qs[i]`. CPU quantizer/dequantizer
exist (`quantize_row_q8_K_ref` / `dequantize_row_q8_K` in `ggml-quants.c`,
SIMD-dispatched), and `ggml.c` registers the type (`is_quantized = true`,
`quantize_q8_K` in the quantize dispatch, `from_string` support).

**Where Q8_K tensors actually come from:**
- `tools/quantize/quantize.cpp:73-77` — the `Q8_K_M/L/XL` recipes (and
  `UD-Q8_K_XL` / `UD_Q8_K_XL` aliases) are all
  `LLAMA_FTYPE_MOSTLY_Q8_0` — **Q8_0-based recipes with source-type
  overrides** for small/norm/embd/GDN roles. They do *not* emit
  `GGML_TYPE_Q8_K` tensors in the bulk.
- **loq per-tensor overrides**: `llama-quantize --tensor-type-file` can
  assign Q8_K to individual tensors, and loq's `dkl-sweep` menu includes
  Q8_K cells (2B sweep state has them). The `loq budget` DP can therefore
  produce mixed-precision configs whose Q8_K tensors hit this gap.

So the type is real, measurably used in loq pipelines, and currently
unusable on-GPU.

## Current state (verified)

| Surface | Location | Q8_K today |
|---|---|---|
| Op offload gate (`MUL_MAT`/`MUL_MAT_ID` type list) | `ggml-cuda.cu:5078` `ggml_backend_cuda_device_supports_op`, `case GGML_TYPE_Q8_K` at :5181 | **present** (fork-added) → ops *are* scheduled on CUDA |
| mmvq vec_dot (`get_vec_dot_q_cuda`) | `mmvq.cu:40` | **absent** → `default: return nullptr` |
| mmvq type dispatch (`mul_mat_vec_q_switch_type`) | `mmvq.cu:1268` | **absent** → `default: GGML_ABORT("fatal error")` |
| mmvq VDR / mmid-max-batch tables | `mmvq.cu:69,151,177,203,213,240,259,273` | absent (defaults: VDR 1, max batch = MMVQ_MAX_BATCH_SIZE) |
| mmq tiled path (load-tiles + vec_dot + `DECL_MMQ_CASE`) | `mmq.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh` | **absent** |
| convert (`dequantize_row_*_cuda`) | `convert.cu:514,574,631` | **absent** → null `convert_func` → `GGML_ASSERT` in the cuBLAS fallback (`ggml-cuda.cu:1410` `ggml_cuda_mul_mat_cublas_impl`) |
| `ggml_cuda_type_traits` (qk/qr/qi/bs) | `common.cuh:981ff` | **absent** |
| get_rows | `getrows.cu` | absent — but K-quants are absent there too (Q6_K has no get_rows case; the GET_ROWS offload list at `ggml-cuda.cu:5216` excludes K-quants) → **not needed** |
| fattn K/V | `fattn.cu:427ff` | not needed — Q8_K never stores attention K/V |

**Dispatch chain for a Q8_K matmul on CUDA today** (`ggml-cuda.cu:1823
ggml_cuda_mul_mat`): `should_use_mmvf` (no) → `should_use_mmf` (no) →
`should_use_mmvq` (**yes** for batch ≤ `MMVQ_MAX_BATCH_SIZE=32` —
`ggml_is_quantized` is table-true) → `ggml_cuda_mul_mat_vec_q` →
`mul_mat_vec_q_switch_type` → **`GGML_ABORT`**. For batch > 32:
`should_use_mmq` (no, Q8_K not in `mmq.cuh` list) → cuBLAS fallback →
`traits::convert(Q8_K)` null → **assert/abort**.

> **Resolved (2026-09-24, empirical + 27B-machine evidence):** the 27B
> machine's build (`/home/brian/build/main_llama.cpp`) pre-dates the Q8_K
> type entirely — it logs `llama_model_loader: unknown type q8_K` when
> loading a Q8_K shard, which is the 27B SIGABRT's true origin (an old
> build, not a kernel gap). That machine needs a fork sync before any Q8_K
> study there. On the current fork, the Vulkan backend's
> `supports_op → default: return false` path keeps Q8_K ops on the CPU
> backend (verified: 2B all-Q8_K KLD pass on the 5060 Ti via Vulkan runs
> at CPU-fallback speed, no abort) — that is the before-patch baseline the
> CUDA port is measured against. Pre-patch CUDA behavior is deliberately
> NOT exercised (static analysis: it hits the `GGML_ABORT` path above);
> the after-patch CUDA run is the first CUDA contact with Q8_K.

## The Q6_K template (what "adding it like Q6_K" means)

Q6_K's full CUDA surface, verified in this tree:

| # | File:line | What |
|---|---|---|
| 1 | `dequantize.cuh:246` | `dequantize_q6_K<dst_t>` device fn (bit-unpack) |
| 2 | `convert.cu:170,312,514/574/631` | `dequantize_block_q6_K` kernel + host wrapper + 3 dispatch cases |
| 3 | `vecdotq.cuh:623-624` | `VDR_Q6_K_Q8_1_MMVQ` / `_MMQ` constants |
| 4 | `vecdotq.cuh:627,650,1019` | `vec_dot_q6_K_q8_1_impl_mmvq` / `_mmq` + entry |
| 5 | `mmvq.cu:23,55,84` | prefetch list, `get_vec_dot_q_cuda`, `get_vdr_mmvq` |
| 6 | `mmvq.cu:171,207,235,253,278,342…567` | per-arch `get_mmvq_mmid_max_batch_*` + should_use tables |
| 7 | `mmvq.cu:1355` | `mul_mat_vec_q_switch_type` case |
| 8 | `mmq.cuh:86,128,143,414,616,780,1634` | sram layout, dp4a tile sizes, dispatch (dp4a + mma), `DECL_MMQ_CASE` |
| 9 | `mmq-load-tiles.cuh:946` | `ggml_cuda_mmq_load_tiles_q6_K` |
| 10 | `mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot_q6_K_q8_1_dp4a/_mma` |

Plus the `ggml_cuda_type_traits<GGML_TYPE_Q6_K>` entry (`common.cuh:1096`)
and the supports_op entry.

## Port plan

### Phase 1 — mmvq + convert (covers token generation, batch ≤ 32; ~150 LOC)

Q8_K's math is Q8_0's (int8 quants, one scale per block — the scale is f32
and the extra `bsums` field is irrelevant to the dot). Per file:

1. **`common.cuh`** — traits entry:
   `{ qk = QK_K /*256*/, qr = 32, qi = 1, bs = sizeof(block_q8_K) }`
   (mirrors Q8_0 at :1040 with `bs` changed; Q8_K has `qi = 1` like Q8_0).
2. **`dequantize.cuh`** — `dequantize_q8_K<dst_t>`:
   `y[i] = ggml_cuda_cast<dst_t>(x[ib].d * x[ib].qs[i])` — ~8 lines, no
   bit ops. (Needed for convert + any dequant-based path.)
3. **`convert.cu`** — `dequantize_block_q8_K` kernel +
   `dequantize_row_q8_K_cuda` wrapper + the 3 dispatch cases (copy of the
   Q6_K block at :170/:312, dequant call trivial).
4. **`vecdotq.cuh`** — `VDR_Q8_K_Q8_1_MMVQ` (start at 1; tune later) +
   `vec_dot_q8_K_q8_1`: per 256-block, `Σ d8[g] * dp4a-sum(qs[32g:32g+32],
   act[32g:32g+32])` × `bq8_K->d` — a direct copy of `vec_dot_q8_0_q8_1`
   (vecdotq.cuh:850ff) with the block pointer arithmetic changed (d is
   f32 at the block head; no `m` field). The 32-element q8_1 activation
   groups align with dp4a exactly as in Q8_0.
5. **`mmvq.cu`** — cases in: prefetch list (:23), `get_vec_dot_q_cuda`
   (:55), `get_vdr_mmvq` (:84), `mul_mat_vec_q_switch_type` (:1355). Leave
   the per-arch `mmid_max_batch` tables at their defaults initially
   (`MMVQ_MAX_BATCH_SIZE`) — Q8_0 appears in several of those tables; Q8_K
   can follow after benchmarking.

### Phase 2 — mmq tiled path (large-batch GEMM performance; optional, ~200 LOC)

Mirror the Q8_0 mmq entries (mmq.cuh:585/749, `MMQ_DP4A_TXS_Q8_0` tile
table at :390) with a dedicated `ggml_cuda_mmq_load_tiles_q8_K` (block
layout differs from Q8_0: f32 d + 256 B qs + 32 B bsums, 292 B/block —
cannot reuse Q8_0's loader verbatim) and `ggml_cuda_mmq_vec_dot_q8_K_q8_1_
dp4a/_mma` (the dp4a body is identical to Q8_0's int8 dot; the mma body
dequants to f16 in-kernel like Q6_K's). `DECL_MMQ_CASE(GGML_TYPE_Q8_K)` +
template instances file entry.

Without Phase 2, large-batch Q8_K matmuls take the cuBLAS fallback (convert
to F16 on-GPU — Phase 1 makes that safe), which is correct but slower.
**Phase 1 is sufficient for correctness everywhere and for
token-generation speed** (the regime that matters for KLD measurement and
serving).

### HIP

No separate work: `ggml/src/ggml-hip/` is a CMake wrapper — the HIP build
compiles the same `ggml-cuda/*.cu` sources through `ggml-cuda/vendors/hip.h`.
The same Phase-1 patch fixes the 27B-study SIGABRT (shared `supports_op`
entry + missing kernel = abort in `mul_mat_vec_q_switch_type`).

## Validation plan

1. **Before** (characterize current behavior): build the fork as-is; run
   `llama-perplexity --kl-divergence` on a small model with a Q8_K tensor
   (quantize one FFN tensor to Q8_K via `--tensor-type-file` on the 2B)
   with `-ngl 99` on the 5060 Ti and on a CPU-only run; record
   abort-vs-fallback + reference KLD. (Also `scripts/diagnose_ppl_gpu.sh`
   to confirm device split.)
2. **After Phase 1**: same run must (a) not abort, (b) show the Q8_K tensor
   executing on-GPU, (c) KLD identical to the CPU reference to ~1e-6
   (KLD is device-independent to ~1e-7 in our measurements).
3. **Unit**: `tests/`-style C test or `llama-bench` mmvq parity — compare
   Q8_K mmvq output vs CPU `dequantize_row_q8_K` matmul on random data
   (the fork's existing ggml CUDA test infra if present, else a small
   standalone).
4. **Perf**: `llama-bench` t/s on the 2B with a Q8_K FFN tensor,
   Phase-1-only vs CPU — expect Q8_K mmvq to land near Q8_0 (same int8 dot,
   +13% block size).
5. **HIP regression**: re-run the 27B Q8_K cell that SIGABRT'd (or a small
   Q8_K model on the HIP build) — must run clean.
6. **End-to-end**: re-measure one Q8_K dkl-sweep cell on GPU post-patch and
   diff against the stored CPU-measured cell (expect agreement ~1e-5).

## Effort estimate

- Phase 1: **half a day** (mechanical copies; the only thinking is the
  vec_dot pointer arithmetic and the traits entry).
- Phase 2: **1–2 days** (SRAM layout + two tile kernels + tuning).
- Validation: half a day (builds dominate).

## Decision

Proceed with **Phase 1 + validation** as one PR to `nzbrian/main_llama.cpp`
(`feat(ggml-cuda): add Q8_K mmvq and convert kernels`); Phase 2 as a follow-
up if llama-bench shows large-batch Q8_K workloads are worth it (unlikely
for the loq pipeline, which is batch=2048 KLD passes — those *do* take the
mmq/cuBLAS path, so Phase 2 may in fact be wanted for KLD-pass speed; decide
after the Phase-1 llama-bench numbers).

## Validation results (2026-09-24, Phase 1 implemented + built)

Model: all-Q8_K Qwen3.5-2B (187 Q8_K tensors + 133 F32, assembled from the
existing 2B dkl-sweep shard set), full calibration (112,640 tokens),
ctx 512, shared base-logits cache (BF16 base, generated on Vulkan).

| Run | Mean KLD | PPL(Q) | speed |
|---|---|---|---|
| CUDA batch=2048, **after patch** (convert→F16 + cuBLAS GEMM) | **0.001026** | 8.304482 | ~1,800 t/s (~1.05 s/chunk) |
| pure CPU batch=2048 (reference, `dot_q8_K_q8_K`) | 0.004351 | 8.326899 | ~117 t/s |
| Vulkan hybrid batch=2048 (pre-patch baseline; Q8_K ops on CPU backend) | 0.004370 | 8.327829 | ~23–80 t/s (contended by concurrent build) |

**Key finding — backend matmul precision dominates the measured KLD for
high-precision types.** The CPU path (`ggml_vec_dot_q8_K_q8_K`) quantizes the
F32 activations to q8_K before the integer dot; the CUDA convert path keeps
F16 activations. Direct kernel test (real 2B tensor, random activations,
CPU ggml matmul vs fp64 exact dequant-matmul): rms error 4.8e-3, maxabs
4.7e-2 — exactly the activation-quantization loss, confirming the CPU path
is lossy, not my CUDA kernel. The dequant formula itself is bit-exact vs
`dequantize_row_q8_K` (12.6M elements, 0 diff).

Consequences:

1. The CUDA number (0.001026) is the closest available estimate of the true
   all-Q8_K weight-quantization KLD; the CPU-measured 0.00435 overstates it
   ~4× via activation quantization.
2. **loq measurement bias**: dkl-sweep KLD cells measured on the CPU path
   (or Vulkan-hybrid) carry this extra activation-quantization cost for
   every tensor. The bias is largest for the *highest-precision* qtypes
   (Q8_K, Q8_0) where true quantization error is smallest — i.e. exactly
   the types where the measurement is most distorted. Mixed-precision
   allocations that pick Q8_K for sensitive tensors are penalized more in
   CPU-measured KLD than in a GPU deployment.
3. The "hidden CPU fallback" (Vulkan) and the 27B SIGABRT are now both
   explained: the 27B machine's build pre-dates the Q8_K type (`unknown
   type q8_K` at load); Vulkan keeps Q8_K ops on the CPU backend (no
   abort), CUDA pre-patch hits the `GGML_ABORT` in the mmvq type switch.

**llama-bench (5060 Ti, post-patch, all-Q8_K 2B):** pp512 = 4,753 t/s
(convert→F16 + cuBLAS path), tg128 = 60.9 t/s (mmvq path).

## Bug found and fixed during validation: mmvq activation-block wrap

The first mmvq implementation copied `vec_dot_q8_0_q8_1` verbatim. That is
**wrong for q8_K**: q8_0's block is 32 elements — exactly one q8_1
activation block — but q8_K's block is 256 elements, spanning **8** q8_1
blocks (each with its own scale). The copied activation read
`get_int_b4(bq8_1->qs, iqs + i)` with `iqs ∈ {0,2,…,62}` indexed far past
the 32-byte `qs` array (out-of-bounds into neighboring blocks) and applied
the first activation block's scale to all 256 elements. Symptom: batch≤8
token generation produced garbage (KLD 13.0, PPL ~1e6), while the
batch>8 convert path was correct — easy to miss if only large-batch KLD
passes are tested.

Isolation chain (isolated CUDA unit test: real 2B tensor, batch-1
mul_mat vs fp64 exact; Q4_K/Q8_0 controls pass):

1. vec_dot → constant 1.0: C[i] = exactly 256 for all i → loop coverage
   and reduction are exact.
2. vec_dot → raw int8 byte sum: bit-exact vs Python column sums → weight
   `qs` reads are exact.
3. vec_dot → `bq8_K->d` only: exact vs Python → d-scale read is exact.
4. ⇒ the activation read was the bug. Fixed by computing the thread's
   256-group element index `j0 = 4*iqs` (always a multiple of 8, so the
   thread's 8 quants land in one activation block), indexing
   `b8 = bq8_1 + j0/32`, `u = get_int_b4(b8->qs, (j0%32)/4 + i)`, and
   scaling with **that block's** `b8->ds`.

Post-fix: batch-1 mmvq matches the fp64 reference to the q8_1
activation-quantization noise floor (maxrel 6e-5 with constant activations;
~1.4% with random — identical behavior class to Q4_K). This is the same
reason Q4_K/Q6_K's vec_dot has the more elaborate `bq8_offset` activation
indexing that q8_0 doesn't need.
