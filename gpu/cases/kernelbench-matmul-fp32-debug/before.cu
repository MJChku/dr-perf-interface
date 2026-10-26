
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// -----------------------------------------------------------------------------
// Skew padding: 8 half elements (16 B) to avoid shared-memory bank conflicts
// -----------------------------------------------------------------------------
#define SKEW 8                       // 8 halves = 16 B padding

// One block  =  8 warps = 256 threads
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

// Macro-tile sizes that one thread-block computes
#define BLOCK_ROW_TILES 8                 // 8 × 16  = 128 rows
#define BLOCK_COL_TILES 8                 // 8 × 16  = 128 cols
#define BLOCK_M  (WMMA_M * BLOCK_ROW_TILES)
#define BLOCK_N  (WMMA_N * BLOCK_COL_TILES)

// -----------------------------------------------------------------------------
// 256-thread WMMA kernel: one block computes one 128 × 128 tile of C
// -----------------------------------------------------------------------------
__global__ void matmul_kernel_wmma(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {

    const int lane_id = threadIdx.x & 31;        // 0 … 31
    const int warp_id = threadIdx.x >> 5;        // 0 … 7

    // Top-left corner of the macro-tile this block is responsible for
    const int block_row = BLOCK_M * blockIdx.y;
    const int block_col = BLOCK_N * blockIdx.x;

    // -------------------------------------------------------------------------
    // Shared memory with skew padding to remove bank conflicts
    // -------------------------------------------------------------------------
    // Each K-slice (16) that the block works on is stored once in shared memory
    __shared__ half Asub_hi[BLOCK_M][WMMA_K + SKEW];          // 128 × 24
    __shared__ half Asub_lo[BLOCK_M][WMMA_K + SKEW];
    // For B we adopt the transposed storage so that the first index is column,
    // giving the column-major layout required by WMMA
    __shared__ half Bsub_hi[BLOCK_N][WMMA_K + SKEW];          // 128 × 24  (col-major)
    __shared__ half Bsub_lo[BLOCK_N][WMMA_K + SKEW];

    // -------------------------------------------------------------------------
    // Accumulators: each warp keeps eight 16×16 fragments (one per column tile)
    // -------------------------------------------------------------------------
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[BLOCK_COL_TILES];
    #pragma unroll
    for (int n = 0; n < BLOCK_COL_TILES; ++n)
        wmma::fill_fragment(c_frag[n], 0.0f);

    // -------------------------------------------------------------------------
    // Loop over the K dimension in steps of 16 (WMMA_K)
    // -------------------------------------------------------------------------
    for (int k_tile = 0; k_tile < N; k_tile += WMMA_K) {

        // ---------------------------------------------------------------------
        // (1) Cooperative LOAD of the current 128×16 slice of A and 16×128 slice
        //     of B into shared memory, converting to (hi, lo) fp16 components.
        // ---------------------------------------------------------------------
        // --- A ----------------------------------------------------------------
        for (int idx = threadIdx.x; idx < BLOCK_M * WMMA_K; idx += THREADS_PER_BLOCK) {
            int r = idx / WMMA_K;       // 0 … 127
            int c = idx % WMMA_K;       // 0 … 15

            int global_r = block_row + r;
            int global_c = k_tile   + c;

            float a_val = (global_r < N && global_c < N)
                          ? A[global_r * N + global_c]
                          : 0.0f;

            half a_hi = __float2half(a_val);
            half a_lo = __float2half(a_val - __half2float(a_hi));

            Asub_hi[r][c] = a_hi;
            Asub_lo[r][c] = a_lo;
        }

        // --- B ----------------------------------------------------------------
        for (int idx = threadIdx.x; idx < BLOCK_N * WMMA_K; idx += THREADS_PER_BLOCK) {
            int c = idx / WMMA_K;       // 0 … 127   (matrix column)
            int r = idx % WMMA_K;       // 0 … 15    (matrix row / K)

            int global_r = k_tile   + r;
            int global_c = block_col + c;

            float b_val = (global_r < N && global_c < N)
                          ? B[global_r * N + global_c]
                          : 0.0f;

            half b_hi = __float2half(b_val);
            half b_lo = __float2half(b_val - __half2float(b_hi));

            // Transpose during store so first index = column (for col-major)
            Bsub_hi[c][r] = b_hi;
            Bsub_lo[c][r] = b_lo;
        }

        __syncthreads();

        // ---------------------------------------------------------------------
        // (2) Each warp processes one 16-row stripe of the macro-tile and all
        //     eight 16-column tiles within that stripe.
        // ---------------------------------------------------------------------
        const int a_row   = warp_id * WMMA_M;          // 0,16,…,112

        // Load A fragments (hi & lo) for this warp’s 16×16 tile row
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> a_hi_frag, a_lo_frag;

        wmma::load_matrix_sync(a_hi_frag, &Asub_hi[a_row][0], WMMA_K + SKEW);
        wmma::load_matrix_sync(a_lo_frag, &Asub_lo[a_row][0], WMMA_K + SKEW);

        // For every column tile in the macro-tile
        #pragma unroll
        for (int n = 0; n < BLOCK_COL_TILES; ++n) {

            // Column offset inside the macro-tile
            const int b_col = n * WMMA_N;              // 0,16,…,112

            // Load the B fragments (hi & lo) for this column tile
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::col_major> b_hi_frag, b_lo_frag;

            wmma::load_matrix_sync(b_hi_frag, &Bsub_hi[b_col][0], WMMA_K + SKEW);
            wmma::load_matrix_sync(b_lo_frag, &Bsub_lo[b_col][0], WMMA_K + SKEW);

            // 1) hi * hi
            wmma::mma_sync(c_frag[n], a_hi_frag, b_hi_frag, c_frag[n]);
            // 2) hi * lo
            wmma::mma_sync(c_frag[n], a_hi_frag, b_lo_frag, c_frag[n]);
            // 3) lo * hi
            wmma::mma_sync(c_frag[n], a_lo_frag, b_hi_frag, c_frag[n]);
            // 4) lo * lo
            wmma::mma_sync(c_frag[n], a_lo_frag, b_lo_frag, c_frag[n]);
        }

        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // (3) Store accumulators back to global memory
    // -------------------------------------------------------------------------
    const int c_row = block_row + warp_id * WMMA_M;

    #pragma unroll
    for (int n = 0; n < BLOCK_COL_TILES; ++n) {
        const int c_col = block_col + n * WMMA_N;
        float* C_tile = C + c_row * N + c_col;

        wmma::store_matrix_sync(C_tile, c_frag[n], N, wmma::mem_row_major);
    }
}

// -----------------------------------------------------------------------------
// Python-exposed helper
// -----------------------------------------------------------------------------
