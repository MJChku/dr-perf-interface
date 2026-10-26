#pragma once
// CUDA-compatible descriptor layout and enum values for the standalone GX
// tensor-map encoder. Namespaced to coexist with the actual CUDA headers.
#include <cstdint>
namespace drperf_gpu_tensor_abi {
using cuuint32_t=uint32_t; using cuuint64_t=uint64_t;
using CUresult=int;
constexpr int CUDA_SUCCESS=0, CUDA_ERROR_INVALID_VALUE=1, CUDA_ERROR_NOT_SUPPORTED=801;
struct alignas(128) CUtensorMap { uint64_t opaque[16]; };
typedef enum CUtensorMapDataType_enum {
    CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
    CU_TENSOR_MAP_DATA_TYPE_UINT16,
    CU_TENSOR_MAP_DATA_TYPE_UINT32,
    CU_TENSOR_MAP_DATA_TYPE_INT32,
    CU_TENSOR_MAP_DATA_TYPE_UINT64,
    CU_TENSOR_MAP_DATA_TYPE_INT64,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ,
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
    CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B
} CUtensorMapDataType;
typedef enum CUtensorMapInterleave_enum {
    CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
    CU_TENSOR_MAP_INTERLEAVE_16B,
    CU_TENSOR_MAP_INTERLEAVE_32B
} CUtensorMapInterleave;
typedef enum CUtensorMapSwizzle_enum {
    CU_TENSOR_MAP_SWIZZLE_NONE = 0,
    CU_TENSOR_MAP_SWIZZLE_32B,
    CU_TENSOR_MAP_SWIZZLE_64B,
    CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B
} CUtensorMapSwizzle;
typedef enum CUtensorMapL2promotion_enum {
    CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
    CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B
} CUtensorMapL2promotion;
typedef enum CUtensorMapFloatOOBfill_enum {
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
} CUtensorMapFloatOOBfill;
}
extern "C" int gx_tensor_map_encode_tiled(
    drperf_gpu_tensor_abi::CUtensorMap*, drperf_gpu_tensor_abi::CUtensorMapDataType,
    uint32_t, void*, const uint64_t*, const uint64_t*, const uint32_t*, const uint32_t*,
    drperf_gpu_tensor_abi::CUtensorMapInterleave,drperf_gpu_tensor_abi::CUtensorMapSwizzle,
    drperf_gpu_tensor_abi::CUtensorMapL2promotion,drperf_gpu_tensor_abi::CUtensorMapFloatOOBfill);
extern "C" int gx_tensor_map_replace_address(drperf_gpu_tensor_abi::CUtensorMap*,void*);
