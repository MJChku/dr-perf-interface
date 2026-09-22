/* Manual GX integration probe: two ranks, real payload validation, then marked
 * host work. Requires CUDA/NCCL development libraries and plain GX emulation. */
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include "../perfmark/perfmark.h"

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)
#define NCCL(call) do { ncclResult_t e = (call); if (e != ncclSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, ncclGetErrorString(e)); exit(1); } } while (0)

int main(void) {
    ncclComm_t comm[2];
    cudaStream_t stream[2];
    float *buffer[2], host[1024];
    int devices[] = {0, 1};
    for (int rank = 0; rank < 2; ++rank) {
        CUDA(cudaSetDevice(rank));
        CUDA(cudaStreamCreate(&stream[rank]));
        CUDA(cudaMalloc((void **)&buffer[rank], sizeof(host)));
    }
    NCCL(ncclCommInitAll(comm, 2, devices));
    puts("NCCL initialized"); fflush(stdout);
    for (int n = 4; n <= 1024; n *= 4) {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA(cudaSetDevice(rank));
            for (int i = 0; i < n; ++i) host[i] = (float)(rank + 1 + i);
            CUDA(cudaMemcpy(buffer[rank], host, n * sizeof(float), cudaMemcpyHostToDevice));
        }
        perfmark_begin("nccl", "n", n);
        NCCL(ncclGroupStart());
        for (int rank = 0; rank < 2; ++rank) {
            CUDA(cudaSetDevice(rank));
            NCCL(ncclAllReduce(buffer[rank], buffer[rank], n, ncclFloat, ncclSum,
                               comm[rank], stream[rank]));
        }
        NCCL(ncclGroupEnd());
        for (int rank = 0; rank < 2; ++rank) {
            CUDA(cudaSetDevice(rank));
            CUDA(cudaStreamSynchronize(stream[rank]));
        }
        perfmark_end("nccl");
        for (int rank = 0; rank < 2; ++rank) {
            CUDA(cudaSetDevice(rank));
            CUDA(cudaMemcpy(host, buffer[rank], n * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) {
                if (host[i] != (float)(3 + 2 * i)) {
                    fprintf(stderr, "wrong rank=%d n=%d i=%d got=%g\n", rank, n, i, host[i]);
                    return 2;
                }
            }
        }
        volatile unsigned long sum = 0;
        perfmark_begin("after_nccl", "n", n);
        for (int i = 0; i < n; ++i) sum += i;
        perfmark_end("after_nccl");
        printf("NCCL n=%d values OK host_sum=%lu\n", n, sum); fflush(stdout);
    }
    for (int rank = 0; rank < 2; ++rank) {
        CUDA(cudaSetDevice(rank));
        NCCL(ncclCommDestroy(comm[rank]));
        CUDA(cudaFree(buffer[rank]));
        CUDA(cudaStreamDestroy(stream[rank]));
    }
    puts("GX_NCCL_PASS");
    return 0;
}
