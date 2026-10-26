// drperf audit (decode_step lane): integration-18 (4f5de3ae) custom decode kernels,
// unmodified sources from src4f5/ (git archive 4f5de3ae), compiled to PTX for
// drperf --gpu. PDL macros compile to no-ops (no GLM53_ENABLE_DECODE_PDL_ALL).
#define GLM53_ENABLE_DECODE_FUSE_PREPK
#define GLM53_ENABLE_FLASHMLA_SPARSE
#include "../../src/decode_ar_norm.cuh"
#include "../../src/decode_fuse_v4.cuh"
#include "../../src/decode_core_glue.cuh"
#include "../../src/attention_flashmla_sparse.cuh"
#include "../../src/bf16_gate_up.cuh"
#include "../../src/lib_indexer/lib_indexer_glue.cu"
