#!/usr/bin/env python3
"""Workspace-only candidate for the adapter's proven all-scales=1 contract."""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from run2 import *
s=(H/'deps/flashmla/csrc/sm100/decode/head64/kernel.cuh').read_text()
a=s.index('                            float4 cur_scale_fp32 = __ldg((float4*)(cur_k_scales_ptr + offset));')
b=s.index('                            if (!is_token_valid)',a)
s=s[:a]+'''                            // Candidate: decode_gather_v32 emits exactly 1.0f scales.
                            e8m0 res[4];
                            *(uint32_t*)res = 0x7f7f7f7fu;
'''+s[b:]
s=s.replace('data_bf16[i] = fp8x2_to_bf16x2_with_scale(data_fp8[i], *(ku::nvbf16*)(&scale));','data_bf16[i] = fp8x2_to_bf16x2(data_fp8[i]);',1)
(H/'drivers2/flash_unit_scale_kernel.cuh').write_text('// WORKSPACE CANDIDATE: V32 unit-scale specialization; pinned original is in deps/flashmla.\n'+s)
wrapper=(H/'drivers2/flash_core.cu').read_text().replace('#include "sm100/decode/head64/kernel.cuh"','#include "flash_unit_scale_kernel.cuh"')
(H/'drivers2/flash_unit_scale.cu').write_text(wrapper)
rc=call('nvcc_flash_unit_scale',[NV,'-std=c++17','-O2','--expt-relaxed-constexpr','-arch=compute_100f',*incs,'-I'+str(H/'deps/flashmla/csrc/sm100/decode/head64'),'-ptx',H/'drivers2/flash_unit_scale.cu','-o',H/'build2/flash_unit_scale.ptx'])
raise SystemExit(rc)
