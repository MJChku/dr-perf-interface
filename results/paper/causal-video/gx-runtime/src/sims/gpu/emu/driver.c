// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The GX Project Authors

#define _GNU_SOURCE
#include <light/cudnn_passthrough.h>

// CUDA driver-API context + module interposer for the GX GPU emulator.
//
// Split out of func-emu.c by pure, behavior-preserving code motion (function
// bodies + the defaultFuncAttrs table copied verbatim). This file holds:
//   * The driver-context family and the per-thread / per-device context state
//     it owns: the thread_ctx (current context) and primary_contexts[] statics,
//     cuCtxGetCurrent/SetCurrent, cuCtxCreate_v2, cuCtxDestroy_v2,
//     cuCtxGetDevice, cuCtxSynchronize, cuCtxPushCurrent_v2/PopCurrent_v2,
//     cuCtxEnablePeerAccess/DisablePeerAccess, and the primary-context calls
//     cuDevicePrimaryCtxGetState/Retain/Release_v2. thread_ctx and
//     primary_contexts are also read by cuPointerGetAttribute's
//     CU_POINTER_ATTRIBUTE_CONTEXT case in emu/device.c, so they keep the
//     hidden-visibility external linkage (FEMU_INTERNAL) they had when they
//     lived in func-emu.c (declared in func-emu-internal.h); the definitions
//     move here.
//   * The module family: cuModuleLoadData(_v2)/LoadDataEx/Unload/
//     GetGlobal_v2/GetFunction and the static defaultFuncAttrs table they use.
//     These call into the PTX translator/JIT (emu/ptx_module.cc) via the
//     gx_ptx_* declarations below, exactly as func-emu.c did.
//   * The driver entry-point resolver cuGetProcAddress_v2 (a self-contained
//     self-dlsym-with-version-suffix lookup falling back to the generic success
//     stub). The generic stub gx_generic_driver_stub stays in func-emu.c (its
//     heavier user is the runtime cudaGetDriverEntryPoint dispatch chain, which
//     stays there) and is reached here via FEMU_INTERNAL.
//   * The private export tables used by CUDA's runtime bootstrap. Unknown UUIDs
//     fail closed; the CUDART primary-context and callback-buffer interfaces
//     have their observed CUDA 13 layouts and always initialize out-params.
//   * Leftover driver bits: cuGetErrorString, cuStreamWriteValue32(_v2),
//     cuStreamGetCtx(_v2), and the runtime cudaDriverGetVersion.
//
// The func_ptr resolver dispatch entries (the strcmp(symbol, ...) chain in
// cudaGetDriverEntryPoint) that map these symbol strings to the functions stay
// in func-emu.c and reference them by their exported names.
//
// Cross-boundary symbols reached via headers exactly as func-emu.c does:
// _device (extern __thread, defined in func-emu.c); createEmuHandle/EmuHandle/
// EmuFuncAttrs/cuda_host_api_lock/ENTRY_PRINT_FEMU from <light/func-emu.h>;
// rm_functions / wrap_func_t / resource_mg_add_sorted from
// <light/resource-mg.h>; physical_device_cnt / gx_generic_driver_stub from
// <light/func-emu-internal.h>; the PTX translator entry points via the
// gx_ptx_* extern declarations below.

#include <driver_types.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <pthread.h>
#include <dlfcn.h>
#include <light/log.h>
#include <light/resource-mg.h>
#include <light/func-emu.h>
#include <config/config.h>
#include <light/func-emu-internal.h>

// _device: thread-local physical current-device index, defined (with external
// linkage) in func-emu.c; declared here exactly as emu/device.c declares it.
extern __thread int _device;

/* emu/ptx_module.cc: translate + JIT PTX so driver-API/nvrtc kernels
 * execute for real on the CPU (falls back to the skip-execution module
 * when the image is not PTX or translation fails). */
extern int gx_ptx_module_load(const void* image, void** module_out);
extern int gx_ptx_is_module(void* module);
extern int gx_ptx_module_get_function(void* module, const char* name, void** func_out);
extern int gx_ptx_module_get_global(void* module, const char* name, void** addr_out, size_t* bytes_out);

// Per-thread CUDA context (CUDA contexts are per-thread in NVIDIA's model)
__thread EmuHandle* thread_ctx = NULL;

// Per-device primary contexts (shared across threads, lazily initialized)
// These represent the "primary context" for each device which is shared,
// but each thread maintains its own current context pointer
EmuHandle* primary_contexts[CONFIG_NUM_LOCAL_GPUS] = {NULL};

/* CUDA private driver interfaces used by libcudart.
 *
 * cuGetExportTable is public, but the UUID-keyed table layouts are private.
 * These two layouts are stable across the CUDA runtimes exercised by GX and
 * match the independently maintained ZLUDA definitions:
 *
 *   6bd5fb6c-5bf4-e74a-8987-d93912fd9df9: CUDART interface (13 slots)
 *   a094798c-2e74-2e74-93f2-0800200c0a66: runtime callback hooks (7 slots)
 *
 * CUDA 13.2's CuPy bootstrap calls CUDART slot 2 once per device and callback
 * slots 2 and 6. It also queries the CUDA 13 requirements interface below and
 * calls its third function in the 15-entry subtable (global slot 25). Unknown
 * UUIDs must return an error: reporting success without writing
 * *ppExportTable makes libcudart jump through an uninitialized pointer.
 */
static const unsigned char kCuda13RequirementsUuid[16] = {
    0xf8, 0xcf, 0xf9, 0x51, 0x21, 0x46, 0x8b, 0x4e,
    0xb9, 0xe2, 0xfb, 0x46, 0x9e, 0x7c, 0x0d, 0xd9,
};
static const unsigned char kCudartInterfaceUuid[16] = {
    0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
    0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9,
};
static const unsigned char kToolsTlsUuid[16] = {
    0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47,
    0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc,
};
static const unsigned char kRuntimeCallbackHooksUuid[16] = {
    0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
    0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66,
};
static const unsigned char kContextLocalStorageUuid[16] = {
    0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
    0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93,
};
static const unsigned char kContextChecksUuid[16] = {
    0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43,
    0x92, 0xf6, 0xbb, 0xd5, 0x00, 0x6d, 0xfa, 0x7e,
};
static const unsigned char kIntegrityCheckUuid[16] = {
    0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b,
    0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2,
};

static unsigned char cudart_callback_buffer1[1024];
static unsigned char cudart_callback_buffer2[14];

typedef void (*CudartContextDtor)(CUcontext, void *, void *);

typedef struct CudartContextLocalValue {
    CUcontext context;
    void *key;
    void *value;
    CudartContextDtor destructor;
    struct CudartContextLocalValue *next;
} CudartContextLocalValue;

static CudartContextLocalValue *cudart_context_local_values;

static CUresult cudart_get_primary_context(CUcontext *pctx, CUdevice dev) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0 || dev >= physical_device_cnt) return CUDA_ERROR_INVALID_DEVICE;

    pthread_mutex_lock(&cuda_host_api_lock);
    if (!primary_contexts[dev]) {
        primary_contexts[dev] = createEmuHandle(CU_CTX);
        primary_contexts[dev]->dev = dev;
    }
    *pctx = (CUcontext)primary_contexts[dev];
    pthread_mutex_unlock(&cuda_host_api_lock);
    return CUDA_SUCCESS;
}

static void cudart_get_callback_buffer1(void **ptr, size_t *size) {
    if (ptr) *ptr = cudart_callback_buffer1;
    if (size) *size = sizeof(cudart_callback_buffer1);
}

static void cudart_get_callback_buffer2(void **ptr, size_t *size) {
    if (ptr) *ptr = cudart_callback_buffer2;
    if (size) *size = sizeof(cudart_callback_buffer2);
}

static CUresult cudart_check_requirements(unsigned int requirement,
                                          const void *state,
                                          const void *requirements) {
    (void)requirement;
    (void)state;
    (void)requirements;
    return CUDA_SUCCESS;
}

static CUcontext cudart_resolve_context(CUcontext context) {
    if (context) return context;
    if (thread_ctx) return (CUcontext)thread_ctx;

    CUcontext primary = NULL;
    if (cudart_get_primary_context(&primary, _device) != CUDA_SUCCESS) {
        return NULL;
    }
    return primary;
}

static CUresult cudart_context_local_put(CUcontext context, void *key,
                                         void *value,
                                         CudartContextDtor destructor) {
    context = cudart_resolve_context(context);
    if (!context || !key) return CUDA_ERROR_INVALID_VALUE;

    pthread_mutex_lock(&cuda_host_api_lock);
    for (CudartContextLocalValue *entry = cudart_context_local_values;
         entry; entry = entry->next) {
        if (entry->context == context && entry->key == key) {
            entry->value = value;
            entry->destructor = destructor;
            pthread_mutex_unlock(&cuda_host_api_lock);
            return CUDA_SUCCESS;
        }
    }

    CudartContextLocalValue *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        pthread_mutex_unlock(&cuda_host_api_lock);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    entry->context = context;
    entry->key = key;
    entry->value = value;
    entry->destructor = destructor;
    entry->next = cudart_context_local_values;
    cudart_context_local_values = entry;
    pthread_mutex_unlock(&cuda_host_api_lock);
    return CUDA_SUCCESS;
}

static CUresult cudart_context_local_delete(CUcontext context, void *key) {
    context = cudart_resolve_context(context);
    if (!context || !key) return CUDA_ERROR_INVALID_VALUE;

    pthread_mutex_lock(&cuda_host_api_lock);
    CudartContextLocalValue **link = &cudart_context_local_values;
    while (*link) {
        CudartContextLocalValue *entry = *link;
        if (entry->context == context && entry->key == key) {
            *link = entry->next;
            free(entry);
            pthread_mutex_unlock(&cuda_host_api_lock);
            return CUDA_SUCCESS;
        }
        link = &entry->next;
    }
    pthread_mutex_unlock(&cuda_host_api_lock);
    return CUDA_SUCCESS;
}

static CUresult cudart_context_local_get(void **value, CUcontext context,
                                         void *key) {
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    *value = NULL;
    context = cudart_resolve_context(context);
    if (!context || !key) return CUDA_ERROR_INVALID_VALUE;

    pthread_mutex_lock(&cuda_host_api_lock);
    for (CudartContextLocalValue *entry = cudart_context_local_values;
         entry; entry = entry->next) {
        if (entry->context == context && entry->key == key) {
            *value = entry->value;
            pthread_mutex_unlock(&cuda_host_api_lock);
            return CUDA_SUCCESS;
        }
    }
    pthread_mutex_unlock(&cuda_host_api_lock);
    return CUDA_ERROR_INVALID_HANDLE;
}

static CUresult cudart_context_check(CUcontext context, unsigned int *flags,
                                     const void **detail) {
    (void)context;
    if (flags) *flags = 0;
    if (detail) *detail = NULL;
    return CUDA_SUCCESS;
}

static unsigned int cudart_context_check_global(void) {
    return 0;
}

static const void *cudart_interface[13];
static const void *integrity_check_interface[3];

/* CUDA runtime integrity hash, ported from ZLUDA's independently
 * reverse-engineered dark_api implementation (MIT/Apache-2.0). */
static const unsigned char kIntegrityMix[256] = {
    0x29,0x2e,0x43,0xc9,0xa2,0xd8,0x7c,0x01,0x3d,0x36,0x54,0xa1,0xec,0xf0,0x06,0x13,
    0x62,0xa7,0x05,0xf3,0xc0,0xc7,0x73,0x8c,0x98,0x93,0x2b,0xd9,0xbc,0x4c,0x82,0xca,
    0x1e,0x9b,0x57,0x3c,0xfd,0xd4,0xe0,0x16,0x67,0x42,0x6f,0x18,0x8a,0x17,0xe5,0x12,
    0xbe,0x4e,0xc4,0xd6,0xda,0x9e,0xde,0x49,0xa0,0xfb,0xf5,0x8e,0xbb,0x2f,0xee,0x7a,
    0xa9,0x68,0x79,0x91,0x15,0xb2,0x07,0x3f,0x94,0xc2,0x10,0x89,0x0b,0x22,0x5f,0x21,
    0x80,0x7f,0x5d,0x9a,0x5a,0x90,0x32,0x27,0x35,0x3e,0xcc,0xe7,0xbf,0xf7,0x97,0x03,
    0xff,0x19,0x30,0xb3,0x48,0xa5,0xb5,0xd1,0xd7,0x5e,0x92,0x2a,0xac,0x56,0xaa,0xc6,
    0x4f,0xb8,0x38,0xd2,0x96,0xa4,0x7d,0xb6,0x76,0xfc,0x6b,0xe2,0x9c,0x74,0x04,0xf1,
    0x45,0x9d,0x70,0x59,0x64,0x71,0x87,0x20,0x86,0x5b,0xcf,0x65,0xe6,0x2d,0xa8,0x02,
    0x1b,0x60,0x25,0xad,0xae,0xb0,0xb9,0xf6,0x1c,0x46,0x61,0x69,0x34,0x40,0x7e,0x0f,
    0x55,0x47,0xa3,0x23,0xdd,0x51,0xaf,0x3a,0xc3,0x5c,0xf9,0xce,0xba,0xc5,0xea,0x26,
    0x2c,0x53,0x0d,0x6e,0x85,0x28,0x84,0x09,0xd3,0xdf,0xcd,0xf4,0x41,0x81,0x4d,0x52,
    0x6a,0xdc,0x37,0xc8,0x6c,0xc1,0xab,0xfa,0x24,0xe1,0x7b,0x08,0x0c,0xbd,0xb1,0x4a,
    0x78,0x88,0x95,0x8b,0xe3,0x63,0xe8,0x6d,0xe9,0xcb,0xd5,0xfe,0x3b,0x00,0x1d,0x39,
    0xf2,0xef,0xb7,0x0e,0x66,0x58,0xd0,0xe4,0xa6,0x77,0x72,0xf8,0xeb,0x75,0x4b,0x0a,
    0x31,0x44,0x50,0xb4,0x8f,0xed,0x1f,0x1a,0xdb,0x99,0x8d,0x33,0x9f,0x11,0x83,0x14,
};

static void integrity_mix_byte(unsigned char state[66], unsigned char byte) {
    unsigned char index = state[64];
    state[(size_t)index + 16] = byte;
    unsigned char next = (unsigned char)((index + 1) & 0x0f);
    state[(size_t)index + 32] = state[index] ^ byte;
    unsigned char mixed = kIntegrityMix[byte ^ state[65]];
    state[(size_t)index + 48] ^= mixed;
    state[65] = state[(size_t)index + 48];
    state[64] = next;
    if (next != 0) return;

    unsigned char value = 0x29;
    for (unsigned char round = 0; round < 0x12; ++round) {
        value ^= state[0];
        state[0] = value;
        for (size_t i = 1; i < 48; ++i) {
            value = state[i] ^ kIntegrityMix[value];
            state[i] = value;
        }
        value = (unsigned char)(value + round);
        if (round != 0x11) value = kIntegrityMix[value];
    }
}

static void integrity_hash_bytes(unsigned char state[66], const void *data,
                                 size_t size, unsigned char xor_mask) {
    const unsigned char *bytes = data;
    for (size_t i = 0; i < size; ++i) {
        integrity_mix_byte(state, bytes[i] ^ xor_mask);
    }
}

static void integrity_finalize(unsigned char state[66], uint64_t result[2]) {
    unsigned char pad = (unsigned char)(16 - state[64]);
    for (unsigned int i = 0; i < pad; ++i) integrity_mix_byte(state, pad);
    for (size_t i = 48; i < 64; ++i) integrity_mix_byte(state, state[i]);
    memcpy(result, state, 2 * sizeof(uint64_t));
}

typedef struct CudartIntegrityInput {
    uint32_t driver_version;
    uint32_t version;
    uint32_t process;
    uint32_t thread;
    const void *cudart_table;
    const void *integrity_table;
    const void *function;
    uint64_t unix_seconds;
} CudartIntegrityInput;

typedef struct CudartIntegrityDevice {
    CUuuid uuid;
    int32_t pci_domain;
    int32_t pci_bus;
    int32_t pci_device;
} CudartIntegrityDevice;

static CUresult cudart_integrity_check(unsigned int version,
                                       uint64_t unix_seconds,
                                       uint64_t result[2]) {
    if (!result) return CUDA_ERROR_INVALID_VALUE;
    LOG(LOG_DEBUG, "cudart integrity check version=%u unix_seconds=%" PRIu64,
        version, unix_seconds);

    switch (version % 10) {
        case 0:
            result[0] = UINT64_C(0x3341181c03cb675c);
            result[1] = UINT64_C(0x8ed383aa1f4cd1e8);
            return CUDA_SUCCESS;
        case 1:
            result[0] = UINT64_C(0x1841181c03cb675c);
            result[1] = UINT64_C(0x8ed383aa1f4cd1e8);
            return CUDA_SUCCESS;
        default:
            break;
    }

    static const unsigned char pass1[16] = {
        0x14,0x6a,0xdd,0xae,0x53,0xa9,0xa7,0x52,
        0xaa,0x08,0x41,0x36,0x0b,0xf5,0x5a,0x9f,
    };
    unsigned char state[66] = {0};
    integrity_hash_bytes(state, pass1, sizeof(pass1), 0x36);

    CudartIntegrityInput input = {
        .driver_version = GX_CUDA_API_VERSION,
        .version = version,
        .process = (uint32_t)getpid(),
        .thread = (uint32_t)(uintptr_t)pthread_self(),
        .cudart_table = cudart_interface,
        .integrity_table = integrity_check_interface,
        .function = (const void *)cudart_integrity_check,
        .unix_seconds = unix_seconds,
    };
    integrity_hash_bytes(state, &input, sizeof(input), 0);

    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count < 0) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    for (int ordinal = 0; ordinal < device_count; ++ordinal) {
        CUdevice dev = 0;
        CudartIntegrityDevice info;
        memset(&info, 0, sizeof(info));
        if (cuDeviceGet(&dev, ordinal) != CUDA_SUCCESS ||
            gx_cuDeviceGetUuid(&info.uuid, dev) != CUDA_SUCCESS) {
            return CUDA_ERROR_INVALID_DEVICE;
        }
        (void)cuDeviceGetAttribute(&info.pci_domain,
                                   CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev);
        (void)cuDeviceGetAttribute(&info.pci_bus,
                                   CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
        (void)cuDeviceGetAttribute(&info.pci_device,
                                   CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
        integrity_hash_bytes(state, &info, sizeof(info), 0);
    }

    uint64_t first[2];
    integrity_finalize(state, first);
    memset(state, 0, 16);
    memset(state + 48, 0, 18);
    integrity_hash_bytes(state, pass1, sizeof(pass1), 0x5c);
    integrity_hash_bytes(state, first, sizeof(first), 0);
    integrity_finalize(state, result);
    return CUDA_SUCCESS;
}

/* cuDNN calls slot 1 to report CUDA launch errors.  Its arguments are a
 * domain, severity, printf-style format, and va_list.  Leaving this callback
 * NULL turns an otherwise reported launch error into a jump to address zero. */
static void cuda13_log_message(const char *domain, int severity,
                               const char *format, va_list args) {
    fprintf(stderr, "GX CUDA13 %s severity=%d: ",
            domain ? domain : "CUDA", severity);
    if (format) vfprintf(stderr, format, args);
    fputc('\n', stderr);
}

/* CUDA 13.2 exposes several adjacent private subtables under this UUID. Keep
 * the observed size headers so libcudart can locate the 15-entry requirements
 * subtable beginning at slot 22. Only its exercised slot 3 (global slot 25)
 * is advertised; unknown functions stay NULL and therefore cannot silently
 * succeed with uninitialized outputs. */
static const void *cuda13_requirements_interface[37] = {
    [0] = (const void *)(uintptr_t)(3 * sizeof(void *)),
    [1] = (const void *)cuda13_log_message,
    [4] = (const void *)(uintptr_t)(2 * sizeof(void *)),
    [6] = (const void *)(uintptr_t)(4 * sizeof(void *)),
    [10] = (const void *)(uintptr_t)(2 * sizeof(void *)),
    [14] = (const void *)(uintptr_t)(7 * sizeof(void *)),
    [22] = (const void *)(uintptr_t)(15 * sizeof(void *)),
    [25] = (const void *)cudart_check_requirements,
};

static const void *cudart_interface[13] = {
    [0] = (const void *)(uintptr_t)(13 * sizeof(void *)),
    [2] = (const void *)cudart_get_primary_context,
};

static const void *runtime_callback_hooks[7] = {
    [0] = (const void *)(uintptr_t)(7 * sizeof(void *)),
    [2] = (const void *)cudart_get_callback_buffer1,
    [6] = (const void *)cudart_get_callback_buffer2,
};

/* Verified against the profiled CUDA driver: TOOLS_TLS slot 2 returns the
 * current CUcontext. The second table's slot 4 returns its unique context ID,
 * identical to cuCtxGetId. cuBLASLt uses both for its kernel registry; treating
 * the first callback as optional tooling silently disables every candidate. */
static CUresult tools_tls_current_context(CUcontext *context) {
    if (!gx_real_cudnn_enabled()) return CUDA_ERROR_NOT_SUPPORTED;
    if (!context) return CUDA_ERROR_INVALID_VALUE;
    return cuCtxGetCurrent(context);
}
static const void *tools_tls_interface[4] = {
    [0] = (const void *)(uintptr_t)(4 * sizeof(void *)),
    [2] = (const void *)tools_tls_current_context,
};
static const unsigned char kContextIdentityUuid[16] = {
    0x21, 0x31, 0x8c, 0x60, 0x97, 0x14, 0x32, 0x48,
    0x8c, 0xa6, 0x41, 0xff, 0x73, 0x24, 0xc8, 0xf2,
};
static CUresult context_identity_unsupported(void) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
/* Advertise only the verified prefix, not the driver's other 88 callbacks. */
static const void *context_identity_interface[5] = {
    [0] = (const void *)(uintptr_t)(5 * sizeof(void *)),
    [1] = (const void *)context_identity_unsupported,
    [2] = (const void *)context_identity_unsupported,
    [3] = (const void *)context_identity_unsupported,
    [4] = (const void *)cuCtxGetId,
};

static const void *context_local_storage_interface[4] = {
    [0] = (const void *)cudart_context_local_put,
    [1] = (const void *)cudart_context_local_delete,
    [2] = (const void *)cudart_context_local_get,
};

static const void *context_checks_interface[4] = {
    [2] = (const void *)cudart_context_check,
    [3] = (const void *)cudart_context_check_global,
};

static const void *integrity_check_interface[3] = {
    [0] = (const void *)(uintptr_t)(3 * sizeof(void *)),
    [1] = (const void *)cudart_integrity_check,
};

CUresult gx_cuGetExportTable(const void **ppExportTable,
                            const CUuuid *pExportTableId) {
    if (!ppExportTable || !pExportTableId) return CUDA_ERROR_INVALID_VALUE;
    *ppExportTable = NULL;

    LOG(LOG_DEBUG,
        "cuGetExportTable UUID "
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-"
        "%02x%02x-%02x%02x%02x%02x%02x%02x",
        (unsigned char)pExportTableId->bytes[0],
        (unsigned char)pExportTableId->bytes[1],
        (unsigned char)pExportTableId->bytes[2],
        (unsigned char)pExportTableId->bytes[3],
        (unsigned char)pExportTableId->bytes[4],
        (unsigned char)pExportTableId->bytes[5],
        (unsigned char)pExportTableId->bytes[6],
        (unsigned char)pExportTableId->bytes[7],
        (unsigned char)pExportTableId->bytes[8],
        (unsigned char)pExportTableId->bytes[9],
        (unsigned char)pExportTableId->bytes[10],
        (unsigned char)pExportTableId->bytes[11],
        (unsigned char)pExportTableId->bytes[12],
        (unsigned char)pExportTableId->bytes[13],
        (unsigned char)pExportTableId->bytes[14],
        (unsigned char)pExportTableId->bytes[15]);

    if (memcmp(pExportTableId->bytes, kCuda13RequirementsUuid,
               sizeof(kCuda13RequirementsUuid)) == 0) {
        *ppExportTable = cuda13_requirements_interface;
        return CUDA_SUCCESS;
    }
    if (memcmp(pExportTableId->bytes, kCudartInterfaceUuid,
               sizeof(kCudartInterfaceUuid)) == 0) {
        *ppExportTable = cudart_interface;
        return CUDA_SUCCESS;
    }
    if (gx_real_cudnn_enabled() &&
        memcmp(pExportTableId->bytes, kContextIdentityUuid, sizeof(kContextIdentityUuid)) == 0) {
        *ppExportTable = context_identity_interface;
        return CUDA_SUCCESS;
    }
    if (memcmp(pExportTableId->bytes, kToolsTlsUuid,
               sizeof(kToolsTlsUuid)) == 0) {
        *ppExportTable = tools_tls_interface;
        return CUDA_SUCCESS;
    }
    if (memcmp(pExportTableId->bytes, kRuntimeCallbackHooksUuid,
               sizeof(kRuntimeCallbackHooksUuid)) == 0) {
        *ppExportTable = runtime_callback_hooks;
        return CUDA_SUCCESS;
    }
    if (memcmp(pExportTableId->bytes, kContextLocalStorageUuid,
               sizeof(kContextLocalStorageUuid)) == 0) {
        *ppExportTable = context_local_storage_interface;
        return CUDA_SUCCESS;
    }
    if (memcmp(pExportTableId->bytes, kContextChecksUuid,
               sizeof(kContextChecksUuid)) == 0) {
        *ppExportTable = context_checks_interface;
        return CUDA_SUCCESS;
    }
    if (memcmp(pExportTableId->bytes, kIntegrityCheckUuid,
               sizeof(kIntegrityCheckUuid)) == 0) {
        *ppExportTable = integrity_check_interface;
        return CUDA_SUCCESS;
    }

    return CUDA_ERROR_INVALID_VALUE;
}

static const int defaultFuncAttrs[] = {
    [CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK]            = 1024,            // max threads/block
    [CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]                = 48 * 1024,       // 48 KB static shared
    [CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES]                 = 64 * 1024,       // 64 KB const mem
    [CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]                 = 32 * 1024,       // 32 KB local spill
    [CU_FUNC_ATTRIBUTE_NUM_REGS]                         = 64,              // ~64 registers/thread
    [CU_FUNC_ATTRIBUTE_PTX_VERSION]                      = 80,              // PTX 7.0 → 70
    [CU_FUNC_ATTRIBUTE_BINARY_VERSION]                   = 90,              // CC 8.0 → 80
    [CU_FUNC_ATTRIBUTE_CACHE_MODE_CA]                    = 0,               // no L1-cache preference
    [CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES]    = 96 * 1024,       // 96 KB dynamic shared
    [CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT] = 0,               // 0% carveout → all to shared
    [CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET]         = 0,               // no cluster size requirement
    [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH]           = 0,               // unused
    [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT]          = 0,               // unused
    [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH]           = 0,               // unused
    [CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED]= 1,               // allow non-portable clusters
    [CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE] = 0,
};

CUresult cuModuleLoadData(CUmodule* module, const void* image)
{
    ENTRY_PRINT_FEMU;
    void* ptx_mod = NULL;
    if (image && gx_ptx_module_load(image, &ptx_mod) == 0) {
        *module = (CUmodule)ptx_mod;
        return 0;
    }
    EmuHandleType type = MODULE;
    EmuHandle* handle = createEmuHandle(type);
    *module = (CUmodule) handle;
    return 0;

}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, CUjit_option *options,
                            void **optionValues)
{
    ENTRY_PRINT_FEMU;
    (void)numOptions;
    (void)options;
    (void)optionValues;
    return cuModuleLoadData(module, image);
}

CUresult cuModuleLoadData_v2(CUmodule *module, const void *image)
{
    return cuModuleLoadData(module, image);
}

CUresult cuModuleUnload(CUmodule module){
    ENTRY_PRINT_FEMU;
    free(module);
    return 0;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name){
    ENTRY_PRINT_FEMU;
    if (gx_ptx_is_module((void*)hmod)) {
        void* p = NULL;
        if (gx_ptx_module_get_global((void*)hmod, name, &p, bytes) == 0) {
            *dptr = (CUdeviceptr)(uintptr_t)p;
            return 0;
        }
        return CUDA_ERROR_NOT_FOUND;
    }
    EmuHandleType type = MODULE;
    EmuHandle* handle = createEmuHandle(type);
    *dptr = (CUdeviceptr) handle;
    return 0;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name){
    ENTRY_PRINT_FEMU;
    if (gx_ptx_is_module((void*)hmod)) {
        void* pf = NULL;
        if (gx_ptx_module_get_function((void*)hmod, name, &pf) == 0) {
            *hfunc = (CUfunction)pf;
            return 0;
        }
        LOG(LOG_ERROR, "gxptx: kernel '%s' not found in translated module", name);
        return CUDA_ERROR_NOT_FOUND;
    }
    EmuHandleType type = FUNC;
    EmuHandle* handle = createEmuHandle(type);
    *hfunc = (CUfunction)handle;
    EmuFuncAttrs* data = calloc(1, sizeof(EmuFuncAttrs));
    handle->data = data;
    memcpy(data->arr, defaultFuncAttrs, sizeof(int)*CU_FUNC_ATTRIBUTE_MAX);

    wrap_func_t *func = (wrap_func_t*)malloc(sizeof(wrap_func_t));
    if (func == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    func->devFunc = (void*)*hfunc;
    func->name = strdup(name ? name : "unknown");
    if (func->name == NULL) {
        free(func);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    func->module = (void*)hmod;
    if (resource_mg_add_sorted(rm_functions, (void*)*hfunc, (void*)func) != 0) {
        free(func->name);
        free(func);
        return CUDA_ERROR_UNKNOWN;
    }
    return 0;
}

CUresult cuCtxGetCurrent(CUcontext *pctx){
    if (gx_real_cudnn_enabled() && !thread_ctx) {
        CUcontext primary = NULL;
        CUresult status = cudart_get_primary_context(&primary, _device);
        if (status != CUDA_SUCCESS) return status;
        thread_ctx = (EmuHandle*)primary;
    }

    ENTRY_PRINT_FEMU;
    if (!thread_ctx) {
        // Lazily create a context on the current device if none exists
        EmuHandleType type = CU_CTX;
        thread_ctx = createEmuHandle(type);
        thread_ctx->dev = _device;
    }
    *pctx = (CUcontext)thread_ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx){
    ENTRY_PRINT_FEMU;
    thread_ctx = (EmuHandle*)ctx;
    // Update the thread's current device to match the context's device
    if (thread_ctx) {
        _device = thread_ctx->dev;
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev){
    ENTRY_PRINT_FEMU;
    EmuHandleType type = CU_CTX;
    EmuHandle* new_ctx = createEmuHandle(type);
    new_ctx->dev = dev;
    *pctx = (CUcontext)new_ctx;
    // Set as current context for this thread
    thread_ctx = new_ctx;
    _device = dev;
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy_v2(CUcontext ctx){
    ENTRY_PRINT_FEMU;
    if (thread_ctx == (EmuHandle*)ctx) {
        thread_ctx = NULL;
    }
    free(ctx);
    return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice *device){
    ENTRY_PRINT_FEMU;
    if (thread_ctx) {
        *device = thread_ctx->dev;
    } else {
        *device = _device; // Use thread's current device
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void){
    ENTRY_PRINT_FEMU;
    // In emulation, synchronization is a no-op
    return 0;
}

typedef struct gx_ctx_stack_entry {
    EmuHandle *previous;
    int device;
    struct gx_ctx_stack_entry *next;
} gx_ctx_stack_entry;
static __thread gx_ctx_stack_entry *gx_context_stack;

CUresult cuCtxPushCurrent_v2(CUcontext ctx){
    if (gx_real_cudnn_enabled()) {
        if (!ctx) return CUDA_ERROR_INVALID_CONTEXT;
        gx_ctx_stack_entry *entry = malloc(sizeof(*entry));
        if (!entry) return CUDA_ERROR_OUT_OF_MEMORY;
        entry->previous = thread_ctx;
        entry->device = _device;
        entry->next = gx_context_stack;
        gx_context_stack = entry;
        thread_ctx = (EmuHandle*)ctx;
        _device = thread_ctx->dev;
        return CUDA_SUCCESS;
    }

    ENTRY_PRINT_FEMU;
    // Simple implementation - just set as current
    thread_ctx = (EmuHandle*)ctx;
    if (thread_ctx) {
        _device = thread_ctx->dev;
    }
    return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent_v2(CUcontext *pctx){
    if (gx_real_cudnn_enabled() && gx_context_stack) {
        if (!pctx) return CUDA_ERROR_INVALID_VALUE;
        *pctx = (CUcontext)thread_ctx;
        gx_ctx_stack_entry *entry = gx_context_stack;
        thread_ctx = entry->previous;
        _device = entry->device;
        gx_context_stack = entry->next;
        free(entry);
        return CUDA_SUCCESS;
    }

    ENTRY_PRINT_FEMU;
    if (pctx) {
        *pctx = (CUcontext)thread_ctx;
    }
    // In simple emulation, we don't maintain a stack, just clear current
    thread_ctx = NULL;
    return CUDA_SUCCESS;
}


CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    ENTRY_PRINT_FEMU;
    (void)peerContext;
    (void)Flags;
    return CUDA_SUCCESS;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    ENTRY_PRINT_FEMU;
    (void)peerContext;
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char** pStr)
{
    ENTRY_PRINT_FEMU;
    if (!pStr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    switch (error) {
        case CUDA_SUCCESS:
            *pStr = "CUDA_SUCCESS";
            break;
        case CUDA_ERROR_INVALID_VALUE:
            *pStr = "CUDA_ERROR_INVALID_VALUE";
            break;
        case CUDA_ERROR_INVALID_DEVICE:
            *pStr = "CUDA_ERROR_INVALID_DEVICE";
            break;
        case CUDA_ERROR_OUT_OF_MEMORY:
            *pStr = "CUDA_ERROR_OUT_OF_MEMORY";
            break;
        default:
            *pStr = "CUDA_ERROR_UNKNOWN";
            break;
    }
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active)
{
    ENTRY_PRINT_FEMU;
    if (dev < 0 || dev >= physical_device_cnt) {
        LOG(LOG_ERROR, "cuDevicePrimaryCtxGetState invalid device %d", dev);
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *active = 1;
    if (flags) {
        *flags = 0; // Default flags
    }
    return CUDA_SUCCESS;
    // Primary context is always active in our emulation
    *active = (primary_contexts[dev] != NULL) ? 1 : 0;
    if (flags) {
        *flags = 0; // Default flags
    }
    LOG(LOG_DEBUG, "cuDevicePrimaryCtxGetState device %d, active %d", dev, *active);
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
{
    ENTRY_PRINT_FEMU;
    if (dev < 0 || dev >= physical_device_cnt) {
        return CUDA_ERROR_INVALID_DEVICE;
    }

    // Lazily create the primary context for this device if it doesn't exist
    if (!primary_contexts[dev]) {
        EmuHandleType type = CU_CTX;
        primary_contexts[dev] = createEmuHandle(type);
        primary_contexts[dev]->dev = dev;
    }

    // Set the primary context as the current context for this thread
    thread_ctx = primary_contexts[dev];
    *pctx = (CUcontext)primary_contexts[dev];
    _device = dev;

    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev)
{
    ENTRY_PRINT_FEMU;
    if (dev < 0 || dev >= physical_device_cnt) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    return CUDA_SUCCESS;
}

/* Driver-API entry-point lookup (used by statically linked cudart and by
 * frameworks probing driver features). Every emulated driver symbol is
 * exported by this library, so resolve against ourselves, preferring the
 * newest _v3/_v2 variant like the real driver does. */
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
    ENTRY_PRINT_FEMU;
    (void)cudaVersion;
    (void)flags;
    if (!symbol || !pfn) return CUDA_ERROR_INVALID_VALUE;

    static void* self_handle = NULL;
    if (self_handle == NULL) {
        Dl_info info;
        if (dladdr((void*)&cuGetProcAddress_v2, &info) && info.dli_fname) {
            self_handle = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
        }
        if (self_handle == NULL) {
            self_handle = RTLD_DEFAULT;
        }
    }

    char buf[256];
    void* fn = NULL;
    const char* suffixes[] = {"_v3", "_v2", ""};
    for (int i = 0; i < 3 && fn == NULL; ++i) {
        snprintf(buf, sizeof(buf), "%s%s", symbol, suffixes[i]);
        fn = dlsym(self_handle, buf);
    }
    if (fn == NULL) {
        // Match the emulator's stub philosophy: unknown driver entry points
        // resolve to a generic success stub so callers that store-and-call
        // unconditionally keep working.
        LOG(LOG_WARNING, "cuGetProcAddress: unknown symbol '%s' (generic stub)", symbol);
        fn = (void*)gx_generic_driver_stub;
    }
    *pfn = fn;
    if (symbolStatus) {
        *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
}

CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr,
                                 cuuint32_t value, unsigned int flags) {
    /* real implementation (or stub) */
    return CUDA_SUCCESS;
}

#ifdef cuStreamWriteValue32
#undef cuStreamWriteValue32
#endif

CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                              cuuint32_t value, unsigned int flags) {
    return cuStreamWriteValue32_v2(stream, addr, value, flags);
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    ENTRY_PRINT_FEMU;
    // Return the current thread's context (streams are associated with the current context)
    if (!thread_ctx) {
        // Lazily create a context if none exists
        EmuHandleType type = CU_CTX;
        thread_ctx = createEmuHandle(type);
        thread_ctx->dev = _device;
    }
    *pctx = (CUcontext)thread_ctx;
    return CUDA_SUCCESS;
}

CUresult cuStreamGetCtx_v2(CUstream hStream, CUcontext *pCtx, CUgreenCtx *pGreenCtx) {
    ENTRY_PRINT_FEMU;
    // Return the current thread's context
    if (!thread_ctx) {
        // Lazily create a context if none exists
        EmuHandleType type = CU_CTX;
        thread_ctx = createEmuHandle(type);
        thread_ctx->dev = _device;
    }
    *pCtx = (CUcontext)thread_ctx;
    // Green context not supported in emulation
    if (pGreenCtx) {
        *pGreenCtx = NULL;
    }
    return CUDA_SUCCESS;
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    ENTRY_PRINT_FEMU;
    if (!driverVersion) return cudaErrorInvalidValue;
    *driverVersion = GX_CUDA_API_VERSION;
    return cudaSuccess;
}

/* Was a generated FEMU no-op returning success WITHOUT setting the output --
 * callers (e.g. torch feature gating) then branched on a garbage version.
 * Same silent-no-op class as the cudaEventRecordWithFlags ordering bug. */
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    ENTRY_PRINT_FEMU;
    if (!runtimeVersion) return cudaErrorInvalidValue;
    *runtimeVersion = GX_CUDA_API_VERSION;
    return cudaSuccess;
}

/* GX surfaces errors synchronously from each interposed call and keeps no
 * sticky per-thread error state, so peek always reports success. Hand-written
 * (not the generated stub) to make that a documented semantic, not an accident. */
cudaError_t cudaPeekAtLastError(void) {
    ENTRY_PRINT_FEMU;
    return cudaSuccess;
}
