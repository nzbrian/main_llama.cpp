#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// MMQ profiler — opt-in via GGML_MQ_PROFILE=1
//
// Breaks down where time is spent in and around the mixed-quantization (MMQ)
// matmul kernels, i.e. in the code before / during / after the Blackwell
// MXFP8 / MXFP4 tensor-core fast path. Everything is a no-op (single cached
// branch) when GGML_MQ_PROFILE is not set, so it is safe to leave compiled in.
//
// Host side (CUDA event ring, real GPU wall-time in ms, valid under CUDA graph
// capture because the events become graph nodes that are re-timed each replay):
//
//   BQUANT   B-side activation quantization prepass kernel(s)   ("before")
//   MMQ      the mul_mat_q kernel itself                        ("during")
//   FIXUP    the stream-k fixup kernel launched after MMQ       ("after")
//   OUTSIDE  everything else in the graph (attention/SDPA, GDN/SSM,
//            layernorm, elementwise, ...).  = graph_ms - (BQUANT+MMQ+FIXUP)
//
// Device side (clock64() phase counters inside mul_mat_q_process_tile, erased
// at compile time for every type except MXFP8 via `if constexpr`):
//
//   LOAD   A/B tile loads + __syncthreads barriers
//   MMA    the vec_dot (tensor-core) calls
//   EPI    write_back
//   BOOK   everything else in the kernel (loop bookkeeping, prologue, ...)
//
// Results are bucketed by ncols_y (== ne11), i.e. the number of tokens:
//
//   bucket 0:  ne11 == 1     (decode)
//   bucket 1:  2   .. 64
//   bucket 2:  65  .. 256
//   bucket 3:  257 .. 1024
//   bucket 4:  > 1024        (prefill)
//
// A cumulative summary is printed to stderr every PRINT_EVERY_GRAPHS graphs
// and once more at process exit.
// ---------------------------------------------------------------------------

namespace mmq_profile {

constexpr int NUM_BUCKETS = 5;
constexpr int NUM_KINDS   = 3; // BQUANT, MMQ, FIXUP
constexpr int NUM_PHASES  = 4; // LOAD, MMA, EPI, BOOK
constexpr int RING_SLOTS  = 4096;
constexpr int PRINT_EVERY_GRAPHS = 1024;

enum kind : int {
    KIND_BQUANT = 0,
    KIND_MMQ    = 1,
    KIND_FIXUP  = 2,
};

enum phase : int {
    PHASE_LOAD = 0,
    PHASE_MMA  = 1,
    PHASE_EPI  = 2,
    PHASE_BOOK = 3,
};

// Bucket index for a token count (ncols_y == ne11). Used on host and device.
__host__ __device__ inline int bucket_of(int ncols_y) {
    if (ncols_y <= 1)    { return 0; }
    if (ncols_y <= 64)   { return 1; }
    if (ncols_y <= 256)  { return 2; }
    if (ncols_y <= 1024) { return 3; }
    return 4;
}

// ---------------------------------------------------------------------------
// Host-side API (defined in mmq.cu). All of these are cheap no-ops when
// GGML_MQ_PROFILE is not set.
// ---------------------------------------------------------------------------
bool  enabled();
void  on_kernel_begin(int kind_idx, int bucket, cudaStream_t stream);
void  on_kernel_end(cudaStream_t stream);
void  graph_begin(cudaStream_t stream);
void  graph_end(cudaStream_t stream);
// Flush the last in-flight graph (CUDA calls) and print the final summary.
// Safe to call from ggml_backend_cuda_free, where the CUDA context is alive.
void  shutdown();

// ---------------------------------------------------------------------------
// Device side.
//
// The in-kernel phase counters live in a plain cudaMalloc'd buffer (NOT a
// __device__ global, which is unreliable across translation units in
// whole-program CUDA linking). The buffer pointer is passed to the mul_mat_q
// kernel as an argument and is nullptr when profiling is disabled.
// ---------------------------------------------------------------------------
struct device_counters {
    unsigned long long cycles[NUM_PHASES][NUM_BUCKETS]; // [phase][bucket]
    unsigned long long tiles[NUM_BUCKETS];
};

// Host: returns the device counter buffer (or nullptr when disabled).
device_counters * prof_buffer();

} // namespace mmq_profile
