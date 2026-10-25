/-
  Machine-readable PTX gap inventory.

  Each key names a behavior that the current translator accepts or encounters
  but does not yet model faithfully enough for an ISA-conformance claim.  The
  opcode coverage table cites these keys and `gxcheck` rejects stale, unknown,
  duplicate, or uncited entries.
-/

namespace Gxptx

/- PTX definition: additional instruction forms exist beyond the currently proved scalar value templates. -/
/- Lean modeling: keeping every audited mismatch in one checked inventory prevents a local proof from being mistaken for full opcode conformance. -/
def knownGapInventory : List (String × String) := [
  ("integer-arithmetic-extra-forms",
   "PTX 9.3 §§9.7.1–.2: packed/saturating/ReLU add/sub/neg/min/max, clmad/mul24/mad24/sad, and explicit addc/subc/madc plus CC.CF carry/borrow forms need distinct opcode/value, packed-lane, or CC-state models as applicable; all currently unmodeled families fail closed with gx_ptx_trap_op"),
  ("integer-abs-min",
   "PTX 9.3 §9.7.1.11 states d = |a| but does not separately define the unrepresentable signed minimum; GX models fixed-width modular negation, so the minimum bit pattern is retained and the theorem certifies that explicit modeling choice"),
  ("mad-saturation",
   "PTX 9.3 §9.7.1.4 permits exactly mad.hi.sat.s32 in addition to the 16 non-saturating lo/hi/wide forms; the 16 non-saturating forms are modeled separately while the saturating form now fails closed"),
  ("integer-divrem-policy",
   "PTX 9.3 §§9.7.1.9–.10 leave division by zero machine-specific and allow machine-dependent signed remainder for negative operands; GX chooses all-ones/dividend zero-divisor results and truncation-toward-zero tdiv/tmod, so its theorems certify that allowed project policy rather than a target-independent PTX result"),
  ("float-arithmetic-modes",
   "PTX 9.3 §§9.7.3–.5 give instruction-specific rounding, FTZ, saturation, approximation, half/bfloat, packed, mixed, and fused rules; GX emits plain/RN sub.f16x2 through tested per-lane sign negation and exact half-add (no subtraction theorem); GX proves exact plain add.f16x2 and models plain mul.bf16x2 with bit-level RN-even semantics, retains narrow plain/RN f32/f64 plus div.full.f32, ex2.approx.ftz.f32, fma.rn.ftz.f32, sqrt.approx.ftz.f32 and rsqrt.approx.ftz.f32 under trusted:float-arith, and plain add.bf16 and add/mul.bf16x2 emission under trusted:packed-bfloat-arith; exact FTZ classification and tested approximation bounds are not floating-value theorems, and unlisted modifier/type combinations fail closed"),
  ("setp-forms",
   "PTX 9.3 §§9.7.6.2 and 9.7.7.2 permit half/bfloat comparisons, optional complementary destinations, and BoolOp composition beyond the current routes. GX proves single-destination basic b/u/s16/32/64 comparisons and tests f32/f64 ordered/unordered/num/nan comparisons including f32 FTZ; half/bfloat, dual destinations, BoolOp, and invalid modifier combinations still fail closed"),
  ("float-minmax",
   "PTX 9.3 §§9.7.3.11–.12 and 9.7.4.7–.8 require NaN selection, signed-zero ordering, optional third-source, abs/xorsign, FTZ, and half/bfloat rules; plain f32 min/max, min.xorsign.abs.f32 and max.NaN.f32 use a tested encoding helper, not a Lean floating-value theorem, while plain f64 min/max retains the trusted host fmin/fmax boundary; unlisted modifier combinations, third-source, half, and bfloat forms fail closed"),
  ("cvt-modifiers-range",
   "PTX 9.3 §9.7.9.22 defines rounding, saturation, FTZ, ReLU, packed-format, and out-of-range conversion rules beyond GX's narrow plain/float-to-int-rounding routes; unsupported modifier combinations now fail closed, while an accepted out-of-range float-to-int conversion still inherits C undefined behavior instead of the PTX result policy"),
  ("mov-forms",
   "PTX 9.3 §§9.7.9.3–.4 include b128 values, address/function operands, and other shapes beyond GX's local scalar ABI. GX models/proves the exact b16=2x8, b32=2x16/4x8, b64=2x32/4x16 pack/unpack layouts; braces select direction, helpers validate actual declared C widths, and unpack sinks omit writes. b128 and malformed vector arities fail closed; address/function spelling, full PTX declaration/type legality, and C helper/compiler/representation correspondence remain outside the value theorem"),
  ("shfl-forms",
   "PTX 9.3 §9.7.9.6 has up/down/idx/bfly modes and an optional predicate destination; GX implements idx/bfly/up/down with optional predicates, while deprecated shfl forms fail closed and warp rendezvous remains structural"),
  ("memory-forms",
   "PTX 9.3 §§9.7.9.8–.11 include b128, state-space/type/vector legality, cache/eviction, ordering, and transaction rules beyond GX's scalarized host accesses; exact L2::cache_hint operands are now separated from addresses, explicitly guarded b128/type/vector/arity/remote-cluster cases fail closed, complete modifier ordering and mutual exclusion remain in trusted:parsing, and vector accesses have no indivisible-transaction claim"),
  ("bit-float-register-class",
   "PTX 9.3 §9.4 Table 26 permits same-width .b32/.b64 and .f32/.f64 register crossings; Tables 27–28 define wider-register reads and extension. GX uses declared C local representations, not nvcc register prefixes, for allowed f32/f64 arithmetic and selp, scalar b32/b64/f32/f64 mov and memory accesses, exact 2/4-field vector mov, f32/f64 cvt, and b32/b64/float setp inputs. Bare .reg values and numbered groups are recognized. Remaining affected routes include subword scalar mov, bit logic/shifts, bit-typed selp, popc/clz/brev, bar member masks, vote/shfl/elect masks and values, atom bit cas/exch, shf/bfi/prmt, register shadowing/declaration grammar, and general wider-register chop/extension; whole-program register type legality is not checked"),
  ("cp-async-semantics",
   "PTX 9.3 §9.7.9.26 specifies asynchronous groups, completion, mbarriers, masks, out-of-bounds modes, tensor forms, proxy visibility, and cluster spaces; GX preserves ordinary global/shared::cta direction, implements literal source-size zero fill, completes copies synchronously, and accounts their mbarrier completion in runtime state. Dynamic source-size/predicate masks, tensor and cluster forms remain unsupported; complete modifier ordering/multiplicity remains in trusted:parsing"),
  ("vote-forms",
   "PTX 9.3 §§9.7.14.9–.10 define all/any/uni and ballot over active or member-mask lanes; GX routes exact vote.sync any/all/ballot including one predicate negation, while uni, deprecated non-sync, sink, and malformed forms fail closed and collective runtime behavior is not a Lean value theorem"),
  ("atom-forms",
   "PTX 9.3 §9.7.14.5 has a type/operation legality table including inc/dec, bitwise, floating, packed/vector, b128, cache-policy, space, scope, and ordering forms; GX allows only its checked scalar add/cas/exch/min/max operation/type pairs and fails closed on every other pair, while complete qualifier order/mutual exclusion remains in trusted:parsing and the theorem covers pure old/next values rather than atomic memory execution"),
  ("sync-runtime-limits",
   "PTX 9.3 §§9.7.14.1–.4, 9.7.14.14, and 9.7.14.16: host fences and the supported CTA mbarrier arrival/transaction/token/parity runtime are not formal PTX synchronizes-with proofs. Unsupported mbarrier forms trap; proxy/address-range, cluster, griddepcontrol, bar, and barrier remain unmodeled or partial runtime approximations"),
  ("matrix-forms",
   "WMMA m16n16k16 f16 A/B loads, f32 D stores and f32.f32 MMA support explicit row/column shared/global forms with a legal emulator-specific fragment mapping; other WMMA shapes/types/modifiers and implicit strides trap. Apart from that, only ldmatrix/stmatrix.sync.aligned.m8n8.x1/x2/x4[.trans].shared.b16 (also count-before-shape order) and the dense WGMMA forms (f16/bf16/tf32/e4m3/e5m2/s8/u8, all legal N, descriptor or register A, all descriptor layouts and transposes) with fence/commit/wait_group N are selected; sparse and b1 WGMMA trap, as do descriptors with reserved bits, CTAs without a complete warpgroup and non-uniform warpgroup operands at run time. Eager completion does not model overlapping tensor-core execution; exact double accumulation rounded once is not H100 bit identity. Other ldmatrix/stmatrix forms and unlisted mma forms trap; integer m16n8k16/k32 s8/u8 MMA uses the separately inventoried trusted integer helper. Both selected collectives require converged valid participation and valid shared addresses mandated by PTX; divergent/invalid programs are outside the claim"),
  ("special-register-coverage",
   "PTX 9.3 chapter 10 defines many special registers beyond GX's tid/ntid/ctaid/nctaid/laneid/warpid/clock/globaltimer subset and envreg1/2 fallback policy; warpid now uses the linear CTA thread index divided by 32, while nwarpid, lane masks, SM/cluster/grid identifiers, memory-size registers, and other unavailable names still need explicit runtime mappings"),
  ("destination-sink",
   "PTX 9.3 §9.2 permits the `_` bit bucket for instructions whose destination may be discarded; audited mov/setp/vote/shfl/atom/ld/cvt sink shapes now fail closed instead of emitting an undeclared C `_`, but sink handling is not yet a uniform opcode-wide translation feature"),
  ("subword-template-certificate",
   "PTX 9.3 §9.6.1 makes 16-bit computation semantics machine-specific; GX deliberately chooses a strict fixed-width target policy, and several 8/16-bit forms print through PW.w32 then rely on destination truncation, but no end-to-end theorem yet connects that ISO-C composition to the chosen subword model")
]

end Gxptx
