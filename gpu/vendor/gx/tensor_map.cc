// SPDX-License-Identifier: MIT
// Tensor Memory Accelerator (TMA) emulation for translated PTX.
//
// cuTensorMapEncodeTiled stores a GX-defined descriptor in
// the opaque 128-byte CUtensorMap. Kernels receive it by value (usually inside
// a __grid_constant__ parameter), may copy it or edit it with
// tensormap.replace, and pass its generic address to cp.async.bulk.tensor,
// which GX executes synchronously at issue: the box is gathered with
// out-of-bounds fill, written to every destination CTA with the tensor map's
// shared-memory swizzle, and the transaction bytes are then completed on the
// destination mbarriers. Correct programs wait on those mbarriers (or bulk
// groups) before consuming the data, so synchronous completion is
// indistinguishable from hardware's asynchronous completion.
//
// Access modes (PTX ISA "Tensors"): tiled and tile::gather4 / tile::scatter4.
// im2col (convolution) modes and interleaved layouts are not modelled: their
// encoders report CUDA_ERROR_NOT_SUPPORTED and the PTX forms trap.
#include "reduce_ops.h"

#include "../../include/drperf_gpu_tensor_map.h"
using namespace drperf_gpu_tensor_abi;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C" void drperf_gpu_memory(uint64_t,uint64_t,uint64_t,unsigned);
extern "C" void drperf_gpu_resource(const char*,uint64_t);
extern "C" uint64_t gx_cluster_shared_ptr(uint32_t address);
extern "C" void gx_mbarrier_complete_tx(uint64_t address, uint32_t bytes);

namespace {

constexpr uint32_t kMagic = 0x4d545847u;  // "GXTM"
constexpr uint32_t kOffsetMask = (1u << 24) - 1;

enum MapMode : uint8_t { kTiledMap = 0 };
// PTX tensormap swizzle_mode / swizzle_atomicity encodings.
enum : uint8_t { kSwNone = 0, kSw32B = 1, kSw64B = 2, kSw128B = 3, kSw96B = 4 };
enum : uint8_t { kAtom16B = 0, kAtom32B = 1, kAtom32BFlip8B = 2, kAtom64B = 3 };

// Host code may edit hardware descriptor bits in place: CuTe clears bit 21 of
// 64-bit word 1 when the global-memory prefix is not contiguous. Word 1 is
// therefore reserved and holds no GX field.
struct GxTensorMap {
    uint32_t magic;
    uint8_t type, rank, sw_mode, oob_fill;
    uint64_t reserved_word1;
    uint64_t address;
    uint64_t strides[4];         // bytes between consecutive indices of dims 1..4
    uint32_t dims_m1[5];         // dimension sizes minus one (sizes reach 2^32)
    uint16_t box[5];             // tiled box sizes (elements)
    uint8_t element_strides[5];  // traversal strides
    uint8_t interleave, sw_atom, map_mode;
};
static_assert(sizeof(GxTensorMap) <= sizeof(CUtensorMap), "GX tensor map exceeds CUtensorMap");
static_assert(offsetof(GxTensorMap, reserved_word1) == 8, "word 1 of the descriptor must stay reserved");

uint64_t dim(const GxTensorMap& m, unsigned i) { return uint64_t(m.dims_m1[i]) + 1; }
uint64_t stride(const GxTensorMap& m, unsigned i) { return i ? m.strides[i - 1] : 0; }

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("TMA: " + what); }

// Element width in bits.
int element_bits(unsigned type) {
    switch (type) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8: return 8;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16: case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: return 16;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32: case CU_TENSOR_MAP_DATA_TYPE_INT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32: case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32: case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ: return 32;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64: case CU_TENSOR_MAP_DATA_TYPE_INT64:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64: return 64;
    case CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B: case CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B: return 6;
    default: return 0;
    }
}

bool is_float(unsigned type) {
    switch (type) {
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16: case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64: case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ: case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ: return true;
    default: return false;
    }
}

// CUtensorMapSwizzle <-> (PTX swizzle_mode, swizzle_atomicity).
bool split_swizzle(unsigned cu, uint8_t& mode, uint8_t& atom) {
    atom = kAtom16B;
    switch (cu) {
    case CU_TENSOR_MAP_SWIZZLE_NONE: mode = kSwNone; return true;
    case CU_TENSOR_MAP_SWIZZLE_32B: mode = kSw32B; return true;
    case CU_TENSOR_MAP_SWIZZLE_64B: mode = kSw64B; return true;
    case CU_TENSOR_MAP_SWIZZLE_128B: mode = kSw128B; return true;
    case CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B: mode = kSw128B; atom = kAtom32B; return true;
    case CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B: mode = kSw128B; atom = kAtom32BFlip8B; return true;
    case CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B: mode = kSw128B; atom = kAtom64B; return true;
    default: return false;
    }
}

// Shared-memory swizzle of an absolute (CTA-relative) shared byte address:
//   32B: A[4] ^= A[7]        64B: A[5:4] ^= A[8:7]     128B: A[6:4] ^= A[9:7]
//   128B atom 32B: A[6:5] ^= A[8:7]   (+ A[3] ^= A[7] with the 8-byte flip)
//   128B atom 64B: A[6] ^= A[7]       96B: A[4] ^= A[7]
uint32_t swizzle(uint32_t a, const GxTensorMap& m) {
    switch (m.sw_mode) {
    case kSwNone: return a;
    case kSw32B: return a ^ ((a >> 3) & 0x10);
    case kSw64B: return a ^ ((a >> 3) & 0x30);
    case kSw96B: return a ^ ((a >> 3) & 0x10);
    case kSw128B:
        switch (m.sw_atom) {
        case kAtom16B: return a ^ ((a >> 3) & 0x70);
        case kAtom32B: return a ^ ((a >> 2) & 0x60);
        case kAtom32BFlip8B: return a ^ ((a >> 2) & 0x60) ^ ((a >> 4) & 0x08);
        case kAtom64B: return a ^ ((a >> 1) & 0x40);
        }
    }
    fail("invalid swizzle mode/atomicity combination");
}

const GxTensorMap& descriptor(uint64_t address) {
    const auto* map = reinterpret_cast<const GxTensorMap*>(static_cast<uintptr_t>(address));
    if (!map || map->magic != kMagic) fail("operand is not a GX-encoded tensor map");
    if (map->interleave) fail("interleaved tensor maps are not supported");
    // Encoding accepts a null placeholder (as the driver does); a copy through
    // it would fault on hardware.
    if (!map->address) fail("tensor map global address is null (placeholder never replaced)");
    return *map;
}

// Shared-memory bit offset of the `index`-th element of a box image.
// ALIGN16B sub-byte types pad each group of 16 values to 16 bytes
// (16U4: 8 data bytes + 8, 16U6: 12 data bytes + 4).
uint64_t smem_bit(const GxTensorMap& m, uint64_t index) {
    if (m.type == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B || m.type == CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B)
        return (index / 16) * 128 + (index % 16) * element_bits(m.type);
    return index * element_bits(m.type);
}

uint64_t read_bits(uint64_t bit, unsigned bits) {
    const auto* p = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(bit / 8));
    if (bits >= 8 && bit % 8 == 0) { uint64_t v = 0; std::memcpy(&v, p, bits / 8); return v; }
    uint32_t w = p[0];
    if (bit % 8 + bits > 8) w |= uint32_t(p[1]) << 8;
    return (w >> (bit % 8)) & ((1u << bits) - 1);
}

void write_bits(uint8_t* base, uint64_t bit, unsigned bits, uint64_t value) {
    if (bits >= 8 && bit % 8 == 0) { std::memcpy(base + bit / 8, &value, bits / 8); return; }
    for (unsigned i = 0; i < bits; ) {            // sub-byte fields (FP6 may span two bytes)
        const uint64_t at = bit + i;
        const unsigned n = std::min<unsigned>(bits - i, 8 - unsigned(at % 8));
        const uint8_t mask = uint8_t(((1u << n) - 1) << (at % 8));
        uint8_t& byte = base[at / 8];
        byte = uint8_t((byte & ~mask) | (((value >> i) << (at % 8)) & mask));
        i += n;
    }
}

uint64_t oob_value(const GxTensorMap& m) {
    if (m.oob_fill != CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA || !is_float(m.type)) return 0;
    const int bits = element_bits(m.type);
    return bits == 16 ? 0x7fffull : bits == 32 ? 0x7fffffffull : 0x7fffffffffffffffull;
}

// One box element: whether it is inside the tensor, its global bit address,
// and its shared-memory bit offset relative to the box start.
struct Access {
    enum Mode : unsigned { tile = 0, gather4 = 4 };
};

// Tiled box (also each gather4/scatter4 row): dims 0..rank-1 starting at coords.
template <class F>
void visit_tiled(const GxTensorMap& m, const int32_t* coords, uint64_t smem_base_index, F&& f,
                 const uint16_t* box_override = nullptr) {
    uint64_t counts[5], total = 1;
    for (unsigned i = 0; i < m.rank; ++i) {
        const unsigned box = box_override ? box_override[i] : m.box[i];
        const unsigned st = i == 0 ? 1 : m.element_strides[i];
        counts[i] = (box + st - 1) / st;
        total *= counts[i];
    }
    const unsigned bits = element_bits(m.type);
    for (uint64_t index = 0; index < total; ++index) {
        uint64_t rest = index, bit = 0;
        bool inside = true;
        for (unsigned i = 0; i < m.rank; ++i) {
            const uint64_t k = rest % counts[i];
            rest /= counts[i];
            const int64_t g = int64_t(coords[i]) + int64_t(k) * (i == 0 ? 1 : m.element_strides[i]);
            if (g < 0 || uint64_t(g) >= dim(m, i)) { inside = false; break; }
            bit += i == 0 ? uint64_t(g) * bits : uint64_t(g) * stride(m, i) * 8;
        }
        f(inside, m.address * 8 + bit, smem_bit(m, smem_base_index + index), bits);
    }
}

// Visit every element of a request in its access mode.
template <class F>
void visit(const GxTensorMap& m, const int32_t* coords, uint32_t ncoords, unsigned mode, const uint16_t* offs, F&& f) {
    if (mode == Access::gather4) {
        if (m.map_mode != kTiledMap || m.rank != 2 || ncoords != 5 || m.box[1] != 1)
            fail("tile::gather4/scatter4 needs a 2D tiled map with box[1] = 1 and coordinates {col, r0, r1, r2, r3}");
        const uint64_t per_row = m.box[0];
        for (unsigned r = 0; r < 4; ++r) {
            const int32_t c[2] = {coords[0], coords[1 + r]};
            visit_tiled(m, c, r * per_row, f);
        }
        return;
    }
    if (ncoords != m.rank) fail("coordinate count differs from the tensor map rank");
    if (mode != Access::tile) fail("im2col TMA modes are not supported");
    visit_tiled(m, coords, 0, f);
}

uint64_t box_bytes(const GxTensorMap& m, unsigned mode, const uint16_t* offs) {
    uint64_t elements;
    switch (mode) {
    case Access::tile: {
        elements = 1;
        for (unsigned i = 0; i < m.rank; ++i) {
            const unsigned st = i == 0 ? 1 : m.element_strides[i];
            elements *= (m.box[i] + st - 1) / st;
        }
        break;
    }
    case Access::gather4: elements = 4ull * m.box[0]; break;
    default: fail("unknown TMA access mode");
    }
    // Transaction bytes count the data copied: for the ALIGN16B sub-byte
    // types the shared-memory padding is not written and not counted
    // (CUTLASS's 16U4_ALIGN16B mainloops expect elements * 4 / 8).
    return (elements * element_bits(m.type) + 7) / 8;
}

gx::RedType reduce_type(unsigned type) {
    switch (type) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT32: return gx::RedType::u32;
    case CU_TENSOR_MAP_DATA_TYPE_INT32: return gx::RedType::s32;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64: return gx::RedType::u64;
    case CU_TENSOR_MAP_DATA_TYPE_INT64: return gx::RedType::s64;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32: return gx::RedType::f32;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ: return gx::RedType::f32_ftz;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16: return gx::RedType::f16;
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: return gx::RedType::bf16;
    default: fail("cp.reduce.async.bulk.tensor element type is not reducible");
    }
}

CUresult check_common(CUtensorMapDataType type, cuuint32_t rank, void* address, const cuuint64_t* dims,
                      const cuuint64_t* strides, const cuuint32_t* element_strides, CUtensorMapInterleave interleave,
                      CUtensorMapSwizzle swizzle_mode, unsigned min_rank) {
    uint8_t mode, atom;
    // As the driver: the global address must be 16-byte aligned, and a null
    // placeholder is legal (CUTLASS grouped GEMM replaces it before use).
    if (uintptr_t(address) % 16 || rank < min_rank || rank > 5 || !dims || (rank > 1 && !strides) || !element_strides ||
        !element_bits(type) || interleave != CU_TENSOR_MAP_INTERLEAVE_NONE || !split_swizzle(swizzle_mode, mode, atom))
        return CUDA_ERROR_INVALID_VALUE;
    for (unsigned i = 0; i < rank; ++i)
        if (!dims[i] || dims[i] > (1ull << 32) || !element_strides[i] || element_strides[i] > 8) return CUDA_ERROR_INVALID_VALUE;
    for (unsigned i = 0; i + 1 < rank; ++i)
        if (strides[i] % 16 || strides[i] >= (1ull << 40)) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}

void fill_common(GxTensorMap& m, CUtensorMapDataType type, cuuint32_t rank, void* address, const cuuint64_t* dims,
                 const cuuint64_t* strides, const cuuint32_t* element_strides, CUtensorMapSwizzle swizzle_mode,
                 CUtensorMapFloatOOBfill oob_fill) {
    m.magic = kMagic;
    m.type = uint8_t(type);
    m.rank = uint8_t(rank);
    split_swizzle(swizzle_mode, m.sw_mode, m.sw_atom);
    m.oob_fill = uint8_t(oob_fill);
    m.address = uint64_t(uintptr_t(address));
    for (unsigned i = 0; i < rank; ++i) {
        m.dims_m1[i] = uint32_t(dims[i] - 1);
        m.element_strides[i] = uint8_t(element_strides[i]);
        if (i > 0) m.strides[i - 1] = strides[i - 1];
    }
}

void store_map(CUtensorMap* map, const GxTensorMap& m) {
    std::memset(map, 0, sizeof(*map));
    std::memcpy(map, &m, sizeof(m));
}

template <class F>
void each_destination(uint32_t multicast, uint32_t cta_mask, uint32_t own_rank, F&& deliver) {
    if (!multicast) { deliver(own_rank); return; }
    if (!cta_mask) fail("multicast with an empty CTA mask");
    for (uint32_t rank = 0; rank < 16; ++rank)
        if (cta_mask & (1u << rank)) deliver(rank);
}

}  // namespace

extern "C" CUresult gx_tensor_map_encode_tiled(CUtensorMap* map, CUtensorMapDataType type,
        cuuint32_t rank, void* address, const cuuint64_t* dims, const cuuint64_t* strides,
        const cuuint32_t* box, const cuuint32_t* element_strides, CUtensorMapInterleave interleave,
        CUtensorMapSwizzle swizzle_mode, CUtensorMapL2promotion, CUtensorMapFloatOOBfill oob_fill) {
    if (!map || !box) return CUDA_ERROR_INVALID_VALUE;
    if (const CUresult r = check_common(type, rank, address, dims, strides, element_strides, interleave, swizzle_mode, 1))
        return r;
    for (unsigned i = 0; i < rank; ++i)
        if (!box[i] || box[i] > 256) return CUDA_ERROR_INVALID_VALUE;
    const bool align16 = type == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B || type == CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B;
    if (align16 && (box[0] != 128 || dims[0] % 128)) return CUDA_ERROR_INVALID_VALUE;
    const uint64_t inner_bytes = (uint64_t(box[0]) * element_bits(type) + 7) / 8;
    const uint64_t span = swizzle_mode == CU_TENSOR_MAP_SWIZZLE_32B ? 32 :
                          swizzle_mode == CU_TENSOR_MAP_SWIZZLE_64B ? 64 : 128;
    if (inner_bytes % 16 || (!align16 && swizzle_mode != CU_TENSOR_MAP_SWIZZLE_NONE && inner_bytes > span))
        return CUDA_ERROR_INVALID_VALUE;
    GxTensorMap m{};
    fill_common(m, type, rank, address, dims, strides, element_strides, swizzle_mode, oob_fill);
    m.map_mode = kTiledMap;
    for (unsigned i = 0; i < rank; ++i) m.box[i] = uint16_t(box[i]);
    store_map(map, m);
    return CUDA_SUCCESS;
}

extern "C" CUresult gx_tensor_map_replace_address(CUtensorMap* map, void* address) {
    auto* m = reinterpret_cast<GxTensorMap*>(map);
    if (!m || m->magic != kMagic) return CUDA_ERROR_INVALID_VALUE;
    m->address = uint64_t(uintptr_t(address));
    return CUDA_SUCCESS;
}

// tensormap.replace.tile.<field> on a 128-byte tensor map in global or shared
// memory. field: 0 global_address, 1 rank, 2 box_dim, 3 global_dim,
// 4 global_stride, 5 element_stride, 6 elemtype, 7 interleave_layout,
// 8 swizzle_mode, 9 swizzle_atomicity, 10 fill_mode (PTX Table 33 values).
extern "C" void gx_tensormap_replace(uint64_t object, uint32_t field, uint32_t ord, uint64_t value) {
    auto* m = reinterpret_cast<GxTensorMap*>(static_cast<uintptr_t>(object));
    if (!m || m->magic != kMagic) fail("tensormap.replace on an object that is not a GX tensor map");
    if (ord > 4) fail("tensormap.replace ordinal must be 0..4");
    static const CUtensorMapDataType kTypes[16] = {
        CU_TENSOR_MAP_DATA_TYPE_UINT8, CU_TENSOR_MAP_DATA_TYPE_UINT16, CU_TENSOR_MAP_DATA_TYPE_UINT32,
        CU_TENSOR_MAP_DATA_TYPE_INT32, CU_TENSOR_MAP_DATA_TYPE_UINT64, CU_TENSOR_MAP_DATA_TYPE_INT64,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
        CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B};
    switch (field) {
    case 0: m->address = value; break;
    case 1: if (value > 4) fail("tensormap rank field is rank-1 in 0..4"); m->rank = uint8_t(value + 1); break;
    case 2: if (!value || value > 256) fail("box_dim must be 1..256"); m->box[ord] = uint16_t(value); break;
    case 3: if (!value || value > (1ull << 32)) fail("global_dim must be 1..2^32"); m->dims_m1[ord] = uint32_t(value - 1); break;
    case 4: if (ord > 3 || value % 16) fail("global_stride ordinal must be 0..3 and a multiple of 16 bytes"); m->strides[ord] = value; break;
    case 5: if (!value || value > 8) fail("element_stride must be 1..8"); m->element_strides[ord] = uint8_t(value); break;
    case 6: if (value > 15) fail("invalid elemtype"); m->type = uint8_t(kTypes[value]); break;
    case 7: if (value > 2) fail("invalid interleave_layout"); m->interleave = uint8_t(value); break;
    case 8: if (value > 4) fail("invalid swizzle_mode"); m->sw_mode = uint8_t(value); break;
    case 9: if (value > 3) fail("invalid swizzle_atomicity"); m->sw_atom = uint8_t(value); break;
    case 10: if (value > 1) fail("invalid fill_mode"); m->oob_fill = uint8_t(value); break;
    default: fail("unknown tensormap.replace field");
    }
}

// cp.async.bulk.tensor global -> shared::cluster. dst and mbar are 32-bit
// shared::cluster addresses (cluster rank in bits [24,32)). With multicast,
// every CTA in cta_mask receives the box at dst's offset and its mbarrier at
// mbar's offset is signalled; with cta_group::2 the signal goes to the CTA of
// each destination's pair selected by mbar's rank bit 0 (CUTLASS clears it to
// signal the pair's even CTA). mode: Access::tile or Access::gather4.
extern "C" void gx_tma_load_mode(uint32_t dst, uint64_t tensor_map, const int32_t* coords, uint32_t ncoords,
                                 uint32_t mbar, uint32_t multicast, uint32_t cta_mask, uint32_t cta_group,
                                 uint32_t mode, const uint16_t* offs) {
    const GxTensorMap& m = descriptor(tensor_map);
    if (cta_group != 1 && cta_group != 2) fail("invalid cp.async.bulk.tensor cta_group");
    (void)offs;
    const uint32_t offset = dst & kOffsetMask, bytes = uint32_t(box_bytes(m, mode, offs));
    const uint64_t fill = oob_value(m);
    uint64_t source_bits=0,destination_bits=0,fill_bits=0,destinations=0;
    each_destination(multicast, cta_mask, dst >> 24, [&](uint32_t rank) {
        auto* base = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(gx_cluster_shared_ptr((rank << 24) | offset)));
        visit(m, coords, ncoords, mode, offs, [&](bool inside, uint64_t gbit, uint64_t sbit, unsigned bits) {
            if(!destinations && inside) {
                source_bits+=bits;
                drperf_gpu_memory(gbit/8,(gbit%8+bits+7)/8,0,1);
            }
            destination_bits+=bits;if(!inside)fill_bits+=bits;
            const uint64_t value = inside ? read_bits(gbit, bits) : fill;
            const uint64_t at = swizzle(uint32_t(offset + sbit / 8), m) - offset;
            write_bits(base, at * 8 + sbit % 8, bits, value);
        });
        ++destinations;
        uint32_t barrier_rank = multicast ? rank : (mbar >> 24);
        if (multicast && cta_group == 2) barrier_rank = (rank & ~1u) | ((mbar >> 24) & 1u);
        gx_mbarrier_complete_tx(gx_cluster_shared_ptr((barrier_rank << 24) | (mbar & kOffsetMask)), bytes);
    });
    drperf_gpu_resource("tma_load_instructions",1);
    drperf_gpu_resource("tma_destination_ctas",destinations);
    drperf_gpu_resource("tma_transaction_bytes",uint64_t(bytes)*destinations);
    drperf_gpu_resource("global_read_bytes",(source_bits+7)/8);
    drperf_gpu_resource("shared_write_bytes",(destination_bits+7)/8);
    drperf_gpu_resource("tma_oob_fill_bytes",(fill_bits+7)/8);
}

extern "C" void gx_tma_load(uint32_t dst, uint64_t tensor_map, const int32_t* coords, uint32_t dims,
                            uint32_t mbar, uint32_t multicast, uint32_t cta_mask, uint32_t cta_group) {
    gx_tma_load_mode(dst, tensor_map, coords, dims, mbar, multicast, cta_mask, cta_group, Access::tile, nullptr);
}

// cp.async.bulk.tensor shared::cta -> global (bulk_group completion), and
// cp.reduce.async.bulk.tensor (op >= 0: gx::RedOp, element-wise relaxed.gpu
// atomics). Out-of-bounds box elements are not written.
extern "C" void gx_tma_store_mode(uint64_t tensor_map, const int32_t* coords, uint32_t ncoords, uint32_t src,
                                  uint32_t mode, int32_t op) {
    const GxTensorMap& m = descriptor(tensor_map);
    const uint32_t offset = src & kOffsetMask;
    const auto* base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(gx_cluster_shared_ptr(src)));
    gx::RedType rtype{};
    if (op >= 0) {
        rtype = reduce_type(m.type);
        if (!gx::reduce_supported(gx::RedOp(op), rtype, /*tensor=*/true)) fail("unsupported tensor reduction op/type");
    }
    uint64_t transferred_bits=0;
    visit(m, coords, ncoords, mode, nullptr,
          [&](bool inside, uint64_t gbit, uint64_t sbit, unsigned bits) {
        if (!inside) return;
        transferred_bits+=bits;
        drperf_gpu_memory(gbit/8,op>=0?(gbit%8+bits+7)/8:0,(gbit%8+bits+7)/8,1);
        const uint64_t at = (swizzle(uint32_t(offset + sbit / 8), m) - offset) * 8 + sbit % 8;
        const uint64_t value = read_bits(uint64_t(uintptr_t(base)) * 8 + at, bits);
        if (op < 0) {
            write_bits(reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(gbit / 8)), gbit % 8, bits, value);
            return;
        }
        if (gbit % 8) fail("sub-byte tensor reduction");
        gx::reduce_element(reinterpret_cast<void*>(static_cast<uintptr_t>(gbit / 8)), value, rtype, gx::RedOp(op));
    });
    uint64_t bytes=(transferred_bits+7)/8;
    drperf_gpu_resource(op<0?"tma_store_instructions":"tma_reduce_instructions",1);
    drperf_gpu_resource("shared_read_bytes",bytes);
    drperf_gpu_resource("global_write_bytes",bytes);
    if(op>=0){drperf_gpu_resource("global_read_bytes",bytes);drperf_gpu_resource("reduction_ops",transferred_bits/element_bits(m.type));}
}

extern "C" void gx_tma_store(uint64_t tensor_map, const int32_t* coords, uint32_t dims, uint32_t src) {
    gx_tma_store_mode(tensor_map, coords, dims, src, Access::tile, -1);
}
