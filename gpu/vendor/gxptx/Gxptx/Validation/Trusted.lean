/-
  Machine-readable inventory of the PTX-to-C translator's trusted boundary.

  These entries are intentionally data rather than claims hidden in prose:
  `Gxptx.Check` reports them and checks that every handled opcode is routed
  through a proved template, cites one of these keys, cites an explicit known
  gap, or traps.  A trusted boundary records an unproved environmental/model
  assumption; known incorrect or incomplete ISA behavior belongs in
  `knownGapInventory`, not here.
-/

namespace Gxptx

/- PTX definition: PTX does not define this trusted-boundary inventory; host C, runtime, parsing, and machine integration are external. -/
/- Lean modeling: this checked key/value inventory names every such unproved boundary separately from theorem-backed models and known incorrect behavior. -/
def trustedInventory : List (String × String) := [
  ("template-print",
   "manual link from each theorem-backed Lean C-value model to its emitted ISO-C string; integer selp casts both alternatives to the instruction type before the host conditional, with bit-size destinations written by memcpy; reviewed one shape at a time and exercised by semantics_test at -O0/-O2, plain/RN sub.f16x2 composes the exact add helper with tested lane-sign negation without a separate subtraction theorem; forms named in knownGapInventory are explicitly outside the claim"),
  ("float-compare",
   "single-destination setp f32/f64 uses tested IEEE encoding classification/order, with ordered/unordered NaN rules and optional f32 FTZ; helper, selector, and register-bitcast correspondence is tested at -O0/-O2, not yet a Lean floating-comparison theorem"),
  ("float-arith",
   "the narrow plain/RN f32/f64 arithmetic allowlist plus div.full.f32, ex2.approx.ftz.f32, fma.rn.ftz.f32, sqrt.approx.ftz.f32 and rsqrt.approx.ftz.f32 rely on host IEEE, libm, RN mode and compiler behavior outside the BitVec model; explicit bit-FTZ helpers preserve input/result zero signs and approximation bounds are numerically tested, not hardware-bit-identical guarantees; plain f32 min/max, min.xorsign.abs.f32 and max.NaN.f32 use a tested encoding helper for signed-zero ordering/NaN rules while f64 min/max retains host fmin/fmax; source/destination reinterpretation uses C storage/memcpy, not name conventions; helper representation/little-endian and host rounding remain trusted, not floating-value proofs; unlisted formats/modifiers/third-source forms fail closed"),
  ("packed-bfloat-arith",
   "plain add.bf16 uses the low lane of gx_bf16x2_add with zero upper lanes; plain add.bf16x2 and mul.bf16x2 route through gx_bf16x2_add/gx_bf16x2_mul; their memcpy lane unpack/pack and native __bf16 arithmetic rely on the host compiler's bfloat representation and rounding behavior outside the BitVec model, while mul has a theorem-backed packed RN-even value model; every modifier and other packed bfloat arithmetic route still fails closed"),
  ("cvt-float",
   "cvt{.rn,.rz,.rm,.rp}{.ftz}{.sat}.f32.f64 uses tested binary64 significand decoding, integer rounding and binary32 encoding in gx_cvt_f64_f32 without changing the host FP mode; no Lean narrowing theorem; float→int rounding (nearbyint/trunc/floor/ceil + C conversion, UB-on-overflow inherited from the reference translator) and int→float / float→float value conversions — IEEE, trusted; same-type cvt{.rni,.rzi,.rmi,.rpi}{.ftz}{.sat}.f32.f32 and {.sat}.f64.f64 (gx_cvt_irnd_f32/f64) use tested binary32 significand masking for rni/rzi/rmi and trusted ceilf/libm roundeven/trunc/floor/ceil otherwise, .ftz flushing signed subnormal inputs and .sat clamping to [+0,1] with NaN to +0, not a Lean floating-value theorem, NaN payloads unspecified; the int-source inner cast IS the proven CCvt reading"),
  ("cvt-f16",
   "gx_cvt_tf32_f32 implements tested binary32-to-TF32 RN-even/RZ/RNA rounding, satfinite and relu with integer encodings, not a Lean floating-value theorem; gx_h2f/gx_f2h/gx_bf2f/gx_f2bf: bit-reinterpret then native _Float16/__bf16 value conversion — hardware half formats trusted; cvt.rn.bf16x2.f32 uses a tested integer RN-even conversion and high-a/low-b packing in gx_cvt_rn_bf16x2_f32, not yet formally proved"),
  ("cvt-fp8",
   "exact cvt.rn.f16x2.e4m3x2 and cvt.rn.satfinite.e4m3x2.f32 use tested integer encoding helpers; finite E4M3 widening magnitude/sign is kernel-proved for the Lean model, not the C implementation or printer; binary32 encoding, tie rounding, saturation, chosen canonical NaNs, and emitted lane/register correspondence remain tested boundaries"),
  ("mem-deref",
   "host *(T*) dereference and gx_gld/gx_gst byte behavior are assumed; Lean proves internal agreement for address arithmetic, type/stride pairing, and GX's fake-base policy, not that fakeBase is PTX ISA semantics"),
  ("memory-fence",
   "printMemoryFence emits a compiler memory barrier for exact CTA scope and __sync_synchronize for wider, restricted, and proxy forms; correctness assumes GCC/x86 fence behavior over GX host/shared mappings, while PTX synchronizes-with, proxy, cluster, and scope equivalence are not formally proved"),
  ("shared-window",
   "shared-space addressing rebases into the per-block window (_gx_shm + offset) and cvta mirrors it — runtime layout contract, not value semantics"),
  ("popc-clz",
   "gx_popc32/64, gx_clz32/64 wrap __builtin_popcount/__builtin_clz — builtin = population count / leading zeros; cross-checked by semantics_test vectors"),
  ("brev",
   "gx_brev32/64 mask-swap + bswap bodies compute bit reversal — cross-checked by semantics_test vectors"),
  ("bfind",
   "gx_bfind32/64 via __builtin_clz — MSB position per ISA incl. .shiftamt; cross-checked by semantics_test vectors"),
  ("bmsk",
   "gx_bmsk32 generates clamp/wrap b32 intervals with full-width clamp controls, five-bit wrap controls and word-end truncation; independent per-bit regression covers edges and random controls, not a Lean bit-value theorem"),
  ("sreg-ctrl",
   "special registers, labels/goto, @p wrapping (the guard TRUTH VALUE is proven: Core emit*_pred / CSem pred lemmas), bar/vote/barrier.red runtime calls, call/param marshalling, ret — structural/runtime plumbing, not value semantics"),
  ("warp-data",
   "warp rendezvous and cross-lane storage, including integer redux add/min/max/and/or/xor, are tested runtime assumptions; CComm equalities cover the corrected idx/bfly source-selection and permitted elect value policies, while unsupported shuffle forms are listed in knownGapInventory"),
  ("ldmatrix",
   "ldmatrix/stmatrix.sync.aligned.m8n8.x1/x2/x4[.trans].shared.b16 (also count-before-shape order) use full-warp pointer shuffles, row/column fragment distribution, real shared-memory loads and completion rendezvous; lane mapping and collective runtime are tested boundaries, not Lean proofs of warp memory execution"),
  ("wmma",
   "m16n16k16 f16 and m16n16k16/m8n32k16/m32n8k16 bf16 A/B with f32 C/D use consistent emulator-specific fragment mappings, real full-warp publication/reuse barriers, aligned uniform row/column shared/global addresses, and ordered host f32 fma accumulation; numerical and routing tests cover this trusted implementation, not a Lean warp-execution proof or bit-identical tensor-core rounding"),
  ("mma-sync",
   "integer mma.sync.aligned m16n8k16/k32 row.col.s32.{s8,u8}.{s8,u8}.s32 with optional satfinite uses tested PTX int8 fragments, exact int64 dot-plus-prior then s32 saturation or modular wrapping in gx_mma_sync_i8; no Lean collective proof; floating mma.sync.aligned m16n8k8 (f16/bf16/tf32), m16n8k16 (f16/bf16) and m16n8k32 (e4m3/e5m2) .row.col with f32 or f16 C/D use the PTX fragment layouts over a full-warp publication barrier; each D element is C plus the double-precision sum of exact products, rounded once; tested against host references, not a Lean proof or bit-identical tensor-core accumulation"),
  ("wgmma",
   "dense wgmma.mma_async m64nNkK for f16/bf16/tf32/e4m3/e5m2/s8/u8 (all legal N, descriptor or register A, all four descriptor layouts, MN-major transposes, imm-scale negation, scale-d, satfinite) plus fence/commit/wait_group N use eager completion and a real 128-lane warpgroup rendezvous (emu/ptx/wgmma.cc); descriptor canonical layouts, swizzles and register-fragment lane maps follow PTX 9.7.16 and CUTLASS SM90 GMMA and are tested against independent host references, not Lean execution proofs; exact products summed in double and rounded once is the chosen unspecified accumulation policy, not bit-identical H100 precision; operand decode, shared visibility and fiber scheduling remain trusted"),
  ("atom-runtime",
   "host sequentially-consistent __atomic indivisibility/memory ordering for the checked scalar type/operation allowlist; represented old/new add/cas/exch/min/max values are proven by CComm.atomic, with unsupported legal PTX forms listed separately"),
  ("cluster-runtime",
   "thread-block clusters: the runtime schedules every CTA of a cluster co-resident, derives %cluster_* registers from the launch cluster shape, tags shared::cluster addresses as (rank<<24)|offset as hardware does (cvta, symbols, mapa) and decodes every shared::cluster access through one helper, and implements barrier.cluster as a split-phase fiber barrier over all cluster threads; clusterlaunchcontrol.try_cancel cancels the next not-yet-started cluster in launch order (GX response encoding: u32 words canceled, ctaid.x, ctaid.y, ctaid.z) and the launch skips canceled clusters; tested against CPU oracles, not proved"),
  ("tensor-memory",
   "tcgen05 executes at issue in the runtime: per-CTA 128-lane x 512-column tensor memory with an aligned power-of-two column allocator, ld/st data-path shapes, commit as an immediate mbarrier arrival (optionally multicast), and mma decoding SM100 shared-memory descriptors (canonical K/MN-major layouts with 32/64/128B swizzle) and instruction descriptors, multiplying exactly decoded operands with double-precision sums rounded once into the f32/s32 accumulator; accumulator and operand placement follow CUTLASS SM100 atoms; tested end to end, not proved; unsupported shapes, kinds and modifiers fail closed"),
  ("tma-runtime",
   "cp.async.bulk.tensor (tile and tile::gather4/scatter4) loads, stores and cp.reduce.async.bulk.tensor reductions, plain cp.async.bulk (cluster destinations, multicast, .ignore_oob, .cp_mask) and cp.reduce.async.bulk, and tensormap.replace/cp_fenceproxy execute at issue in the runtime from a GX-encoded CUtensorMap: box traversal with element strides, zero or NaN out-of-bounds fill, packed 4/6-bit ALIGN8B/ALIGN16B layouts, all PTX swizzle modes and atomicities on shared addresses, element-wise atomic reductions, multicast delivery and cta_group::2 even/odd mbarrier routing, then complete_tx on each destination mbarrier; tested against CPU oracles, not proved; im2col modes and interleaved layouts fail closed"),
  ("async-copy",
   "for the direction-checked global/shared::cta subset GX collapses async copy to synchronous completion and proves only the post-copy byte function; literal source-size zero fill and per-launch CTA mbarrier arrival/transaction/phase tracking are regression-tested runtime behavior, not Lean proofs; masks, tensor, cluster, and visibility limitations remain explicit known gaps"),
  ("trap",
   "multimem/NVLS, prmt non-generic modes, and every unknown opcode emit gx_ptx_trap_op — loud trap, never a silent wrong value"),
  ("parsing",
   "PTX text parsing, operand/register naming, vector-vs-scalar shape detection (Trans/Module string plumbing)"),
  ("toolchain",
   "the Lean 4.15 kernel/compiler and clang at -O1 -fno-strict-aliasing -fwrapv")
]

end Gxptx
