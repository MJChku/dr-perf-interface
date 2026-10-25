/-
  Reviewed C string templates for PTX instruction families.

  Value semantics live under `Gxptx.Semantics`; this module is the compact
  string-rendering boundary.  Each direct renderer has two adjacent comments:
  the relevant PTX shape and why this exact existing C spelling models it.
-/
import Gxptx.Translation.C.Types
import Gxptx.Semantics.C.Communication

namespace Gxptx

/-! ## Memory-ordering templates -/

/- PTX definition: `membar` and thread `fence` order this thread's earlier and later memory operations at their declared level or scope. -/
/- Lean modeling: plain `.cta` uses a compiler memory barrier on GX's x86 host, while wider, restricted, and proxy forms use a full host fence so payload writes precede shared completion flags. -/
def printMemoryFence (ctaScope : Bool) : String :=
  if ctaScope then "__asm__ __volatile__(\"\" ::: \"memory\");"
  else "__sync_synchronize();"

/-! ## Core arithmetic, logic, comparison, and selection templates -/

/- PTX definition: these constructors cover the modeled scalar arithmetic, predicate, comparison, selection, logic, and shift instruction shapes. -/
/- Lean modeling: every branch preserves the reviewed C spelling associated with a shallow `CTemplate.evalC` value case; the spelling/compiler correspondence itself remains `trusted:template-print`. -/
def CTemplate.print (t : CTemplate) (a b c : String) : String :=
  match t with
  /- PTX definition: `add` returns the low fixed-width sum of two operands. -/
  /- Lean modeling: unsigned casts make wraparound defined and mirror the shallow `.add` value case; the emitted text/compiler link remains `trusted:template-print`. -/
  | .add t  => "((" ++ t.uN ++ ")(" ++ a ++ ") + (" ++ t.uN ++ ")(" ++ b ++ "))"
  /- PTX definition: `add.f16x2` performs two independent default-RN binary16 additions. -/
  /- Lean modeling: the integer-only helper implements the reviewed exact-unit/RN-even algorithm; generated-C execution remains at the common `trusted:template-print` boundary. -/
  | .addF16x2 => "gx_f16x2_add_rn(" ++ a ++ ", " ++ b ++ ")"
  /- PTX definition: `sub` returns the low fixed-width difference of two operands. -/
  /- Lean modeling: unsigned casts make wraparound defined and mirror the shallow `.sub` value case; the emitted text/compiler link remains trusted. -/
  | .sub t  => "((" ++ t.uN ++ ")(" ++ a ++ ") - (" ++ t.uN ++ ")(" ++ b ++ "))"
  /- PTX definition: `mul.lo` retains the low result-width bits of the product. -/
  /- Lean modeling: unsigned result-width multiplication mirrors the shallow `.mul` value case; printing/compiler behavior is trusted and subword certificate coverage is separately gapped. -/
  | .mul t  => "((" ++ t.uN ++ ")(" ++ a ++ ") * (" ++ t.uN ++ ")(" ++ b ++ "))"
  /- PTX definition: `mul.bf16x2` performs two independent default-RN packed bfloat16 products. -/
  /- Lean modeling: the helper performs native `__bf16` lane operations; its compiler correspondence remains `trusted:packed-bfloat-arith` and `trusted:template-print`. -/
  | .mulBF16x2 => "gx_bf16x2_mul(" ++ a ++ ", " ++ b ++ ")"
  /- PTX definition: `mad.lo` adds a third operand to the low fixed-width product. -/
  /- Lean modeling: unsigned casts mirror the shallow `.mad` wraparound value case; exact C parsing/evaluation remains `trusted:template-print`. -/
  | .mad t  => "((" ++ t.uN ++ ")(" ++ a ++ ") * (" ++ t.uN ++ ")(" ++ b ++ ") + (" ++ t.uN ++ ")(" ++ c ++ "))"
  /- PTX definition: integer `neg` returns the two's-complement fixed-width negation. -/
  /- Lean modeling: the selected fixed width mirrors the shallow `.neg` value case; generated C interpretation remains a trusted printer/compiler boundary. -/
  | .neg t  => "(-(" ++ t.uN ++ ")(" ++ a ++ "))"
  /- PTX definition: `not.bN` complements every bit of its source. -/
  /- Lean modeling: the unsigned cast fixes the operand width and mirrors the shallow `.bnot` value case; final C correspondence stays trusted. -/
  | .bnot t => "(~(" ++ t.uN ++ ")(" ++ a ++ "))"
  /- PTX definition: `and.bN` combines corresponding source bits with conjunction. -/
  /- Lean modeling: same-width unsigned casts mirror the shallow `.band` value case; emitted syntax/compiler execution remains trusted. -/
  | .band t => "((" ++ t.uN ++ ")(" ++ a ++ ") & (" ++ t.uN ++ ")(" ++ b ++ "))"
  /- PTX definition: `or.bN` combines corresponding source bits with disjunction. -/
  /- Lean modeling: same-width unsigned casts mirror the shallow `.bor` value case; emitted syntax/compiler execution remains trusted. -/
  | .bor t  => "((" ++ t.uN ++ ")(" ++ a ++ ") | (" ++ t.uN ++ ")(" ++ b ++ "))"
  /- PTX definition: `xor.bN` combines corresponding source bits with exclusive-or. -/
  /- Lean modeling: same-width unsigned casts mirror the shallow `.bxor` value case; emitted syntax/compiler execution remains trusted. -/
  | .bxor t => "((" ++ t.uN ++ ")(" ++ a ++ ") ^ (" ++ t.uN ++ ")(" ++ b ++ "))"
  /- PTX definition: `not.pred` complements the source predicate. -/
  /- Lean modeling: PTX has Boolean truth but no integer encoding; C `!` agrees with GX's chosen zero/nonzero host representation, while printing/compiler behavior remains trusted. -/
  | .lnot   => "(!" ++ a ++ ")"
  /- PTX definition: `and.pred` is true exactly when both source predicates are true. -/
  /- Lean modeling: C `&&` mirrors the shallow zero/nonzero predicate value case; the generated-C link itself remains trusted. -/
  | .land   => "(" ++ a ++ " && " ++ b ++ ")"
  /- PTX definition: `or.pred` is true exactly when either source predicate is true. -/
  /- Lean modeling: C `||` mirrors the shallow zero/nonzero predicate value case; the generated-C link itself remains trusted. -/
  | .lor    => "(" ++ a ++ " || " ++ b ++ ")"
  /- PTX definition: `xor.pred` is true exactly when source predicate truth values differ. -/
  /- Lean modeling: comparing nonzero tests mirrors the shallow predicate-xor value case; parsing/compiler behavior remains trusted. -/
  | .lxor   => "((" ++ a ++ " != 0) != (" ++ b ++ " != 0))"
  -- shift amounts stay UNCAST: gx_* converts the amount via (uint64_t)y,
  -- which equals `b.toNat` for any unsigned amount register (and for signed
  -- ones under sign-extension) — a (uint32_t) cast here would truncate a
  -- 64-bit amount and diverge from evalC's clamp on the full value.
  /- PTX definition: `shl` returns zero when the amount is at least the operand width. -/
  /- Lean modeling: the helper selection mirrors the shallow `.shl` clamp value case without truncating the amount; helper/compiler behavior is trusted. -/
  | .shl t  => "gx_shl((" ++ t.uN ++ ")(" ++ a ++ "), " ++ b ++ ")"
  /- PTX definition: unsigned/bit `shr` zero-fills and returns zero for an out-of-range amount. -/
  /- Lean modeling: the unsigned operand cast and helper mirror the shallow `.lshr` value case; helper/compiler execution remains trusted. -/
  | .lshr t => "gx_shr((" ++ t.uN ++ ")(" ++ a ++ "), " ++ b ++ ")"
  /- PTX definition: signed `shr` fills vacated bits with the source sign bit. -/
  /- Lean modeling: the signed operand cast and helper mirror the shallow `.ashr` value case; helper/compiler execution remains trusted. -/
  | .ashr t => "gx_sar((" ++ t.iN ++ ")(" ++ a ++ "), " ++ b ++ ")"
  /- PTX definition: unsigned `div` uses unsigned integer quotient semantics, while division by zero is unspecified. -/
  /- Lean modeling: unsigned casts/helper mirror GX's shallow `.udiv` value case, including its chosen all-ones zero-divisor policy; that policy and helper/compiler link are trusted rather than uniquely ISA-mandated. -/
  | .udiv t => "gx_udiv((" ++ t.uN ++ ")(" ++ a ++ "), (" ++ t.uN ++ ")(" ++ b ++ "))"
  /- PTX definition: signed `div` computes `d = a / b`, while division by zero yields an unspecified, machine-specific value. -/
  /- Lean modeling: truncation toward zero and the zero/`MIN ÷ -1` guards are GX/C helper policies; those choices and the printer/compiler link remain trusted. -/
  | .sdiv t => "gx_sdiv((" ++ t.iN ++ ")(" ++ a ++ "), (" ++ t.iN ++ ")(" ++ b ++ "))"
  /- PTX definition: unsigned `rem` returns the division remainder, while a zero divisor has unspecified behavior. -/
  /- Lean modeling: unsigned casts/helper mirror GX's chosen shallow `.urem` policy of returning the dividend on zero; that policy and helper/compiler link remain trusted. -/
  | .urem t => "gx_urem((" ++ t.uN ++ ")(" ++ a ++ "), (" ++ t.uN ++ ")(" ++ b ++ "))"
  /- PTX definition: negative signed operands are machine-dependent, while a zero divisor is unspecified and machine-specific. -/
  /- Lean modeling: signed casts/helper mirror GX's chosen `Int.tmod`/edge-case shallow policy; that choice and helper/compiler link remain trusted. -/
  | .srem t => "gx_srem((" ++ t.iN ++ ")(" ++ a ++ "), (" ++ t.iN ++ ")(" ++ b ++ "))"
  /- PTX definition: unsigned `min` returns the lesser unsigned operand. -/
  /- Lean modeling: unsigned comparison mirrors the shallow `.umin` value case and returns an untouched operand; C printing/compiler evaluation remains trusted. -/
  | .umin t => "((" ++ t.uN ++ ")(" ++ a ++ ") < (" ++ t.uN ++ ")(" ++ b ++ ") ? " ++ a ++ " : " ++ b ++ ")"
  /- PTX definition: unsigned `max` returns the greater unsigned operand. -/
  /- Lean modeling: unsigned comparison mirrors the shallow `.umax` value case and returns an untouched operand; C printing/compiler evaluation remains trusted. -/
  | .umax t => "((" ++ t.uN ++ ")(" ++ a ++ ") > (" ++ t.uN ++ ")(" ++ b ++ ") ? " ++ a ++ " : " ++ b ++ ")"
  /- PTX definition: signed `min` returns the lesser signed operand. -/
  /- Lean modeling: signed comparison mirrors the shallow `.smin` value case and returns an untouched operand; C printing/compiler evaluation remains trusted. -/
  | .smin t => "((" ++ t.iN ++ ")(" ++ a ++ ") < (" ++ t.iN ++ ")(" ++ b ++ ") ? " ++ a ++ " : " ++ b ++ ")"
  /- PTX definition: signed `max` returns the greater signed operand. -/
  /- Lean modeling: signed comparison mirrors the shallow `.smax` value case and returns an untouched operand; C printing/compiler evaluation remains trusted. -/
  | .smax t => "((" ++ t.iN ++ ")(" ++ a ++ ") > (" ++ t.iN ++ ")(" ++ b ++ ") ? " ++ a ++ " : " ++ b ++ ")"
  /- PTX definition: `selp` returns its first value for true and its second value for false. -/
  /- Lean modeling: the conditional mirrors GX's chosen host representation of PTX Boolean truth; PTX supplies no integer encoding, and printer/compiler behavior remains trusted. -/
  | .sel    => "(" ++ a ++ " ? " ++ b ++ " : " ++ c ++ ")"
  /- PTX definition: integer `setp` applies eq/ne/lt/le/gt/ge under signed or unsigned interpretation. -/
  /- Lean modeling: cast choice and operator mirror the shallow `.cmp` value case; the emitted comparison and C compiler interpretation remain `trusted:template-print`. -/
  | .cmp sgn cop t =>
      let ct := if sgn then t.iN else t.uN
      "((" ++ ct ++ ")(" ++ a ++ ") " ++ cop.c ++ " (" ++ ct ++ ")(" ++ b ++ "))"

/- PTX definition: single-destination `setp` on `.b16`, `.u16`, or `.s16` compares the low 16-bit values; bit types admit only equality/inequality, while signedness selects integer ordering. -/
/- Lean modeling: explicit `int16_t`/`uint16_t` casts make comparison declaration-independent and implement the width-generic `CTemplate.cmp` relation; printer/compiler execution remains trusted. -/
def printCmp16 (sgn : Bool) (cop : CmpOp) (a b : String) : String :=
  let ct := if sgn then "int16_t" else "uint16_t"
  "((" ++ ct ++ ")(" ++ a ++ ") " ++ cop.c ++ " (" ++ ct ++ ")(" ++ b ++ "))"

/-! ## Signed integer absolute value -/

/- PTX definition: `abs.s16`, `abs.s32`, and `abs.s64` negate a negative source and otherwise copy it, with fixed-width result bits. -/
/- Lean modeling: a signed comparison chooses the branch while unsigned subtraction implements negation modulo the exact width without signed-overflow undefined behavior. -/
def printAbsS (w : IW) (a : String) : String :=
  "((" ++ w.iN ++ ")(" ++ a ++ ") < 0 ? (" ++ w.uN ++ ")(0 - (" ++
    w.uN ++ ")(" ++ a ++ ")) : (" ++ w.uN ++ ")(" ++ a ++ "))"

/-! ## GX fixed-width 16-bit division, remainder, and ordering -/

/- PTX definition: `div.u16` divides operands interpreted as unsigned 16-bit values. -/
/- Lean modeling: explicit `uint16_t` casts implement GX's strict fixed-width policy and prevent host promotion from changing the chosen value. -/
def printDivU16 (a b : String) : String :=
  "((uint16_t)gx_udiv((uint16_t)(" ++ a ++ "), (uint16_t)(" ++ b ++ ")))"

/- PTX definition: `div.s16` divides operands interpreted as signed 16-bit values. -/
/- Lean modeling: explicit `int16_t` casts implement GX's strict fixed-width policy before the guarded helper preserves the chosen result pattern. -/
def printDivS16 (a b : String) : String :=
  "((uint16_t)gx_sdiv((int16_t)(" ++ a ++ "), (int16_t)(" ++ b ++ ")))"

/- PTX definition: `rem.u16` returns the remainder of unsigned 16-bit division. -/
/- Lean modeling: explicit `uint16_t` casts implement GX's strict fixed-width policy and prevent declaration signedness from changing helper inputs. -/
def printRemU16 (a b : String) : String :=
  "((uint16_t)gx_urem((uint16_t)(" ++ a ++ "), (uint16_t)(" ++ b ++ ")))"

/- PTX definition: `rem.s16` returns GX's chosen signed 16-bit remainder for the machine-dependent PTX cases. -/
/- Lean modeling: explicit `int16_t` casts and `gx_srem` implement GX's strict fixed-width, truncation-toward-zero policy. -/
def printRemS16 (a b : String) : String :=
  "((uint16_t)gx_srem((int16_t)(" ++ a ++ "), (int16_t)(" ++ b ++ ")))"

/- PTX definition: `min.u16` returns the lesser operand under unsigned 16-bit ordering. -/
/- Lean modeling: both comparison operands and selected results are cast to `uint16_t`, making the result independent of the C register declaration type. -/
def printMinU16 (a b : String) : String :=
  "((uint16_t)(" ++ a ++ ") < (uint16_t)(" ++ b ++ ") ? (uint16_t)(" ++ a ++ ") : (uint16_t)(" ++ b ++ "))"

/- PTX definition: `min.s16` returns the lesser operand under signed 16-bit ordering. -/
/- Lean modeling: `int16_t` comparison reads the exact two's-complement values and unsigned result casts preserve the selected bit pattern. -/
def printMinS16 (a b : String) : String :=
  "((int16_t)(" ++ a ++ ") < (int16_t)(" ++ b ++ ") ? (uint16_t)(" ++ a ++ ") : (uint16_t)(" ++ b ++ "))"

/- PTX definition: `max.u16` returns the greater operand under unsigned 16-bit ordering. -/
/- Lean modeling: both comparison operands and selected results are cast to `uint16_t`, making the result independent of the C register declaration type. -/
def printMaxU16 (a b : String) : String :=
  "((uint16_t)(" ++ a ++ ") > (uint16_t)(" ++ b ++ ") ? (uint16_t)(" ++ a ++ ") : (uint16_t)(" ++ b ++ "))"

/- PTX definition: `max.s16` returns the greater operand under signed 16-bit ordering. -/
/- Lean modeling: `int16_t` comparison reads the exact two's-complement values and unsigned result casts preserve the selected bit pattern. -/
def printMaxS16 (a b : String) : String :=
  "((int16_t)(" ++ a ++ ") > (int16_t)(" ++ b ++ ") ? (uint16_t)(" ++ a ++ ") : (uint16_t)(" ++ b ++ "))"

/-! ## Multiply-family templates -/

/- PTX definition: `mul.wide.{s,u}{16,32}` multiplies source-width values and returns the double-width product. -/
/- Lean modeling: inner casts truncate/read each source at 16 or 32 bits and outer casts extend it to the 32- or 64-bit product type before multiplication. -/
def printMulWide (sgn src32 : Bool) (a b : String) : String :=
  let (oT, iT) :=
    if src32 then (if sgn then ("int64_t", "int32_t") else ("uint64_t", "uint32_t"))
    else (if sgn then ("int32_t", "int16_t") else ("uint32_t", "uint16_t"))
  "((" ++ oT ++ ")(" ++ iT ++ ")(" ++ a ++ ") * (" ++ oT ++ ")(" ++ iT ++ ")(" ++ b ++ "))"

/- PTX definition: `mul.hi.{s,u}16` returns product bits 31 through 16 in a 16-bit destination. -/
/- Lean modeling: 16-bit source casts form an exact 32-bit product, conversion to its unsigned bit pattern makes the right shift defined, and truncation selects precisely bits 31 through 16. -/
def printMulHi16 (sgn : Bool) (a b : String) : String :=
  let product :=
    if sgn then
      "(uint32_t)((int32_t)(int16_t)(" ++ a ++ ") * (int32_t)(int16_t)(" ++ b ++ "))"
    else
      "(uint32_t)(uint16_t)(" ++ a ++ ") * (uint32_t)(uint16_t)(" ++ b ++ ")"
  "((uint16_t)((" ++ product ++ ") >> 16))"

/- PTX definition: `mul.hi.{s,u}{32,64}` returns the high half of the signed or unsigned double-width product. -/
/- Lean modeling: the exact double-width product is converted to its unsigned bit pattern before a defined logical shift, then truncated to the PTX result width. -/
def printMulHi (sgn : Bool) (t : PW) (a b : String) : String :=
  match t, sgn with
  | .w32, false =>
      "((uint32_t)(((uint64_t)(uint32_t)(" ++ a ++ ") * (uint64_t)(uint32_t)(" ++ b ++ ")) >> 32))"
  | .w32, true =>
      "((uint32_t)(((uint64_t)((int64_t)(int32_t)(" ++ a ++ ") * (int64_t)(int32_t)(" ++ b ++ "))) >> 32))"
  | .w64, false =>
      "((uint64_t)(((unsigned __int128)(uint64_t)(" ++ a ++ ") * (unsigned __int128)(uint64_t)(" ++ b ++ ")) >> 64))"
  | .w64, true =>
      "((uint64_t)(((unsigned __int128)((__int128)(int64_t)(" ++ a ++ ") * (__int128)(int64_t)(" ++ b ++ "))) >> 64))"

/- PTX definition: `mad.wide.{s,u}{16,32}` computes the full double-width product `a*b` and adds double-width operand `c`. -/
/- Lean modeling: `printMulWide` forms the exact product and conversion to the unsigned double-width type makes the final fixed-width addition defined modulo 2^(2n). -/
def printMadWide (sgn src32 : Bool) (a b c : String) : String :=
  let oT := if src32 then "uint64_t" else "uint32_t"
  "((" ++ oT ++ ")(" ++ printMulWide sgn src32 a b ++ ") + (" ++ oT ++ ")(" ++ c ++ "))"

/- PTX definition: `mad.hi.{s,u}16` adds 16-bit `c` to product bits 31 through 16 and writes the low 16 sum bits. -/
/- Lean modeling: `printMulHi16` selects the required product half and the outer unsigned 16-bit cast implements result-width wraparound without signed-overflow undefined behavior. -/
def printMadHi16 (sgn : Bool) (a b c : String) : String :=
  "((uint16_t)(" ++ printMulHi16 sgn a b ++ " + (uint16_t)(" ++ c ++ ")))"

/- PTX definition: `mad.hi.{s,u}` adds the third operand to the selected high half of the product with result-width wraparound. -/
/- Lean modeling: `printMulHi` supplies the modeled high half and the unsigned result-width casts make the final addition wrap modulo the destination width. -/
def printMadHi (sgn : Bool) (t : PW) (a b c : String) : String :=
  "((" ++ t.uN ++ ")" ++ printMulHi sgn t a b ++ " + (" ++ t.uN ++ ")(" ++ c ++ "))"

/-! ## Conversion templates -/

/- PTX definition: integer `cvt` reads the source with its declared signedness/width and converts to the declared destination signedness/width. -/
/- Lean modeling: the inner source cast chooses sign- or zero-extension and the outer destination cast fixes the resulting width exactly as in the existing `CCvt.evalC` model. -/
def printCvtII (ssgn : Bool) (sw : IW) (dsgn : Bool) (dw : IW) (a : String) : String :=
  let st := if ssgn then sw.iN else sw.uN
  let dt := if dsgn then dw.iN else dw.uN
  "((" ++ dt ++ ")((" ++ st ++ ")(" ++ a ++ ")))"

/- PTX definition: integer-to-float `cvt` first interprets the integer source according to its signedness and width. -/
/- Lean modeling: the inner fixed-width C cast is the proved source interpretation; the outer float conversion preserves the existing explicitly trusted IEEE boundary. -/
def printCvtIF (ssgn : Bool) (sw : IW) (dstCT : String) (a : String) : String :=
  let st := if ssgn then sw.iN else sw.uN
  "((" ++ dstCT ++ ")((" ++ st ++ ")(" ++ a ++ ")))"

/-! ## Move-family templates -/

/- PTX definition: scalar `mov` may copy a bit pattern without changing it. -/
/- Lean modeling: the generated helper call is the unchanged C rendering of the proved `CMov.movB` bit move. -/
def printMovB (d a : String) : String :=
  "gx_movb(" ++ d ++ ", " ++ a ++ ");"

/- PTX definition: vector-to-scalar `mov` packs two source components into the low and high halves of the destination. -/
/- Lean modeling: the helper arguments preserve destination, low half, and high half order from the proved `CMov.pack` model. -/
def printMovPack (d lo hi : String) : String :=
  "gx_movpack(" ++ d ++ ", " ++ lo ++ ", " ++ hi ++ ");"

/- PTX definition: scalar-to-vector `mov` extracts the low and high halves into two destination components. -/
/- Lean modeling: the helper arguments preserve low destination, high destination, and source order from the proved `CMov.unpackLo`/`unpackHi` models. -/
def printMovUnpack (lo hi s : String) : String :=
  "gx_movunpack(" ++ lo ++ ", " ++ hi ++ ", " ++ s ++ ");"

/-! ## Bit-manipulation templates -/

/- PTX definition: `shf.{l,r}.{wrap,clamp}.b32` concatenates two words and uses a wrapped or clamped shift amount. -/
/- Lean modeling: direction selects the matching helper and the amount expression implements the unchanged `& 31` wrap or `min(n,32)` clamp used by `CBits`. -/
def printShf (right wrap : Bool) (a b c : String) : String :=
  let dir := if right then "r" else "l"
  let n := if wrap then "((" ++ c ++ ") & 31u)"
           else "((" ++ c ++ ") > 32u ? 32u : (" ++ c ++ "))"
  "gx_shf_" ++ dir ++ "(" ++ a ++ ", " ++ b ++ ", " ++ n ++ ")"

/- PTX definition: `bfi.b{32,64}` inserts `len` low bits at `pos`, with position and length read from low control bits. -/
/- Lean modeling: the width selects the matching helper and explicit `uint32_t` control casts preserve the existing `CBits.bfi` printer/model interface. -/
def printBfi (w64 : Bool) (ins base pos len : String) : String :=
  (if w64 then "gx_bfi64(" else "gx_bfi32(")
    ++ ins ++ ", " ++ base ++ ", (uint32_t)(" ++ pos ++ "), (uint32_t)(" ++ len ++ "))"

/- PTX definition: `bfe.{u,s}{32,64}` extracts a positioned field with zero- or sign-extension. -/
/- Lean modeling: signedness and width choose the helper spelling; `bfeU_spec` and `bfeS_spec` prove normalized-control extraction and padding, while template printing remains trusted. -/
def printBfe (sgn w64 : Bool) (a pos len : String) : String :=
  "gx_bfe_" ++ (if sgn then "s" else "u") ++ (if w64 then "64" else "32")
    ++ "(" ++ a ++ ", " ++ pos ++ ", " ++ len ++ ")"

/- PTX definition: generic `prmt.b32` selects output bytes from two inputs under a control operand. -/
/- Lean modeling: the helper receives operands in unchanged PTX order and its value behavior is the proved `CBits.prmt` model; non-generic modes trap elsewhere. -/
def printPrmt (a b c : String) : String :=
  "gx_prmt(" ++ a ++ ", " ++ b ++ ", " ++ c ++ ")"

/- PTX definition: `popc.b32/.b64` returns the number of set bits in the source word. -/
/- Lean modeling: width chooses the existing 32- or 64-bit runtime helper, whose builtin behavior is an explicitly inventoried trusted boundary. -/
def printPopc (w64 : Bool) (a : String) : String :=
  (if w64 then "gx_popc64(" else "gx_popc32(") ++ a ++ ")"

/- PTX definition: `clz.b32/.b64` returns the number of leading zero bits in the source word. -/
/- Lean modeling: width chooses the existing 32- or 64-bit runtime helper, whose builtin behavior is an explicitly inventoried trusted boundary. -/
def printClz (w64 : Bool) (a : String) : String :=
  (if w64 then "gx_clz64(" else "gx_clz32(") ++ a ++ ")"

/- PTX definition: `brev.b32/.b64` reverses the bit order of the source word. -/
/- Lean modeling: width chooses the unchanged 32- or 64-bit helper spelling, with helper correctness explicitly recorded in the trusted inventory and tests. -/
def printBrev (w64 : Bool) (a : String) : String :=
  (if w64 then "gx_brev64(" else "gx_brev32(") ++ a ++ ")"

/- PTX definition: `bfind` returns a bit position, optionally transformed by `.shiftamt`, according to the source signedness. -/
/- Lean modeling: the selected helper receives the source-width cast plus unchanged shift-amount and signedness flags used by the existing runtime implementation. -/
def printBfind (w64 : Bool) (a shiftamt sgn : String) : String :=
  (if w64 then "gx_bfind64((uint64_t)(" else "gx_bfind32((uint32_t)(")
    ++ a ++ "), " ++ shiftamt ++ ", " ++ sgn ++ ")"

/-! ## Communication templates -/

/- PTX definition: these cases render atomic, election, restricted shuffle, and bulk-copy shapes. -/
/- Lean modeling: each branch preserves helper, operand order, predicate assignment, and temporary spelling associated with its shallow `CommTemplate` value model; printing/runtime/compiler links remain trusted. -/
def CommTemplate.print (t : CommTemplate) (out pred a b c d ct : String) : String :=
  match t with
  /- PTX definition: `atom.add` atomically adds a source value and returns the prior memory value. -/
  /- Lean modeling: operand/result order mirrors the shallow atomic value model; atomicity, helper execution, and printing remain trusted runtime boundaries. -/
  | .atomAdd =>
      out ++ " = gx_atom_add<" ++ ct ++ ">(" ++ a ++ ", (" ++ ct ++ ")(" ++ b ++ "));"
  /- PTX definition: `atom.cas` conditionally stores the replacement when memory equals the comparison value and returns the old value. -/
  /- Lean modeling: address/comparison/replacement order mirrors the shallow value model; atomic runtime and printer/compiler correspondence remain trusted. -/
  | .atomCas =>
      out ++ " = gx_atom_cas<" ++ ct ++ ">(" ++ a ++ ", (" ++ ct ++ ")(" ++ b ++ "), (" ++ ct ++ ")(" ++ c ++ "));"
  /- PTX definition: `atom.exch` atomically replaces memory with the source and returns the prior value. -/
  /- Lean modeling: address/value/result order mirrors the shallow exchange model; atomic runtime and printer/compiler behavior remain trusted. -/
  | .atomExch =>
      out ++ " = gx_atom_exch<" ++ ct ++ ">(" ++ a ++ ", (" ++ ct ++ ")(" ++ b ++ "));"
  /- PTX definition: `atom.min.uN` stores the lesser unsigned value atomically and returns the old memory value. -/
  /- Lean modeling: the unsigned `ct` and operand order mirror the shallow `.atomMinU` value; atomic/helper/printer correspondence remains trusted. -/
  | .atomMinU =>
      out ++ " = gx_atom_min<" ++ ct ++ ">(" ++ a ++ ", (" ++ ct ++ ")(" ++ b ++ "));"
  /- PTX definition: `atom.min.sN` stores the lesser signed value atomically and returns the old memory value. -/
  /- Lean modeling: the signed `ct` and operand order mirror the shallow `.atomMinS` value; atomic/helper/printer correspondence remains trusted. -/
  | .atomMinS =>
      out ++ " = gx_atom_min<" ++ ct ++ ">(" ++ a ++ ", (" ++ ct ++ ")(" ++ b ++ "));"
  /- PTX definition: `atom.max.uN` stores the greater unsigned value atomically and returns the old memory value. -/
  /- Lean modeling: the unsigned `ct` and operand order mirror the shallow `.atomMaxU` value; atomic/helper/printer correspondence remains trusted. -/
  | .atomMaxU =>
      out ++ " = gx_atom_max<" ++ ct ++ ">(" ++ a ++ ", (" ++ ct ++ ")(" ++ b ++ "));"
  /- PTX definition: `atom.max.sN` stores the greater signed value atomically and returns the old memory value. -/
  /- Lean modeling: the signed `ct` and operand order mirror the shallow `.atomMaxS` value; atomic/helper/printer correspondence remains trusted. -/
  | .atomMaxS =>
      out ++ " = gx_atom_max<" ++ ct ++ ">(" ++ a ++ ", (" ++ ct ++ ")(" ++ b ++ "));"
  /- PTX definition: `elect.sync` selects one participating lane and reports whether the current lane was selected. -/
  /- Lean modeling: the two helper results mirror shallow election values and preserve destination order; warp runtime and printing remain trusted. -/
  | .elect =>
      out ++ " = gx_elect_lane((uint32_t)" ++ a ++ "); " ++ pred ++
        " = gx_elect_pred((uint32_t)" ++ a ++ ");"
  /- PTX definition: `shfl.sync.idx.b32` selects an indexed source lane and reports participation validity. -/
  /- Lean modeling: mode zero and operand order mirror the shallow `.shflIdx` value; rendezvous/data exchange, temporary text, and compiler behavior remain trusted. -/
  | .shflIdx =>
      let butterfly := "0"
      "{\n      int _gx_shfl_p = 0;\n      " ++ out ++
        " = gx_shfl_sync((uint32_t)" ++ a ++ ", (uint32_t)" ++ b ++
        ", (uint32_t)" ++ c ++ ", (uint32_t)" ++ d ++ ", " ++ butterfly ++
        ", &_gx_shfl_p);\n      " ++ pred ++ " = _gx_shfl_p;\n    }"
  /- PTX definition: `shfl.sync.bfly.b32` selects a butterfly source lane and reports participation validity. -/
  /- Lean modeling: mode one and operand order mirror the shallow `.shflBfly` value; rendezvous/data exchange, temporary text, and compiler behavior remain trusted. -/
  | .shflBfly =>
      let butterfly := "1"
      "{\n      int _gx_shfl_p = 0;\n      " ++ out ++
        " = gx_shfl_sync((uint32_t)" ++ a ++ ", (uint32_t)" ++ b ++
        ", (uint32_t)" ++ c ++ ", (uint32_t)" ++ d ++ ", " ++ butterfly ++
        ", &_gx_shfl_p);\n      " ++ pred ++ " = _gx_shfl_p;\n    }"
  /- PTX definition: downward shuffle uses a non-wrapping lane-plus-offset and upper clamp. -/
  /- Lean modeling: helper mode three mirrors shflDown; the helper owns data exchange. -/
  | .shflDown =>
      let butterfly := "3"
      "{\n      int _gx_shfl_p = 0;\n      " ++ out ++
        " = gx_shfl_sync((uint32_t)" ++ a ++ ", (uint32_t)" ++ b ++
        ", (uint32_t)" ++ c ++ ", (uint32_t)" ++ d ++ ", " ++ butterfly ++
        ", &_gx_shfl_p);\n      " ++ pred ++ " = _gx_shfl_p;\n    }"
  /- PTX definition: `shfl.sync.up.b32` reads an earlier lane or returns its own value below the clamp. -/
  /- Lean modeling: runtime mode two preserves the offset, clamp, membership mask, and optional result predicate. -/
  | .shflUp =>
      "{\n      int _gx_shfl_p = 0;\n      " ++ out ++
        " = gx_shfl_sync((uint32_t)" ++ a ++ ", (uint32_t)" ++ b ++
        ", (uint32_t)" ++ c ++ ", (uint32_t)" ++ d ++ ", 2" ++
        ", &_gx_shfl_p);\n      " ++ pred ++ " = _gx_shfl_p;\n    }"
  /- PTX definition: a valid, unmasked `cp.async.bulk` transfers its requested bytes after completion. -/
  /- Lean modeling: this route assumes legal size/alignment/ranges and excludes masks/out-of-bounds modes; effects, asynchrony, and printer/compiler execution remain trusted/gapped. -/
  | .copyBytes =>
      "gx_ptx_memcpy(" ++ a ++ ", " ++ b ++ ", (size_t)" ++ c ++ ", " ++ d ++ ");"

end Gxptx
