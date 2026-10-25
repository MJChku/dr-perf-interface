/-
  Shallow, width-generic PTX integer arithmetic semantics, including the
  complete non-saturating integer multiply and multiply-add families.

  Every operation is a direct Lean function over `BitVec w`; there is no PTX
  syntax tree or interpreter layer. Related scalar arithmetic families are
  consolidated here so their reviewed definitions can be audited together.
-/
import Gxptx.Semantics.Types

namespace Gxptx.Ptx

/-! ## Scalar addition and subtraction -/

/- PTX definition: unsaturated scalar `add` computes `d = a + b`. -/
/- Lean modeling: `BitVec w` addition returns the low `w` bits, matching fixed-width wrapping addition. -/
def add  (a b : BitVec w) : BitVec w := a + b

/- PTX definition: `add.u16` stores the low 16 bits of the unsigned sum. -/
/- Lean modeling: `BitVec 16` implements GX's documented strict fixed-width target policy for this machine-specific subword case. -/
def addU16 (a b : BitVec 16) : BitVec 16 := add a b

/- PTX definition: `add.s16` stores the 16-bit two's-complement sum pattern. -/
/- Lean modeling: low signed/unsigned sum bits coincide under GX's documented strict fixed-width 16-bit policy. -/
def addS16 (a b : BitVec 16) : BitVec 16 := add a b

/- PTX definition: unsaturated `add.u32` computes the low 32 bits of `a + b`. -/
/- Lean modeling: fixing the generic `BitVec` addition to width 32 gives exactly the specified modulo-`2^32` unsigned result. -/
def addU32 (a b : BitVec 32) : BitVec 32 := add a b

/- PTX definition: unsaturated `add.s32` computes the same 32-bit sum pattern under signed interpretation. -/
/- Lean modeling: two's-complement signed and unsigned addition have identical low 32 bits, so the same width-fixed shallow operation is correct. -/
def addS32 (a b : BitVec 32) : BitVec 32 := add a b

/- PTX definition: unsaturated `add.u64` computes the low 64 bits of `a + b`. -/
/- Lean modeling: fixing the generic `BitVec` addition to width 64 gives exactly the specified modulo-`2^64` unsigned result. -/
def addU64 (a b : BitVec 64) : BitVec 64 := add a b

/- PTX definition: unsaturated `add.s64` computes the same 64-bit sum pattern under signed interpretation. -/
/- Lean modeling: two's-complement signed and unsigned addition have identical low 64 bits, so the same width-fixed shallow operation is correct. -/
def addS64 (a b : BitVec 64) : BitVec 64 := add a b

/-! ## Packed binary16 addition -/

/- PTX definition: binary16 has one sign bit, five exponent bits, and ten fraction bits. -/
/- Lean modeling: these projections interpret a lane directly from its 16-bit register representation. -/
def f16Sign (x : BitVec 16) : Bool := x.toNat / 32768 = 1
def f16Exp (x : BitVec 16) : Nat := (x.toNat / 1024) % 32
def f16Frac (x : BitVec 16) : Nat := x.toNat % 1024

/- PTX definition: half-precision arithmetic accepts finite values, infinities, and NaNs. -/
/- Lean modeling: predicates classify the exact exponent/fraction encodings; NaN payload choice is handled by `addF16RN`. -/
def f16IsNaN (x : BitVec 16) : Bool := f16Exp x = 31 && f16Frac x != 0
def f16IsInf (x : BitVec 16) : Bool := f16Exp x = 31 && f16Frac x == 0
def f16IsZero (x : BitVec 16) : Bool := f16Exp x = 0 && f16Frac x == 0

/- PTX definition: finite binary16 values are sign × significand × a power of two, including subnormals. -/
/- Lean modeling: every finite value is represented exactly as an integer number of `2^-24` units, avoiding host floating-point semantics. -/
def f16FiniteUnits (x : BitVec 16) : Int :=
  let e := f16Exp x
  let f := f16Frac x
  let magnitude := if e = 0 then f else (1024 + f) * 2 ^ (e - 1)
  if f16Sign x then -(Int.ofNat magnitude) else Int.ofNat magnitude

/- PTX definition: plain `add.f16` rounds to nearest with ties to an even significand, preserves subnormals, and overflows to infinity. -/
/- Lean modeling: exact `2^-24` units are rounded to an 11-bit significand; canonical quiet NaN is GX's permitted choice for PTX's unspecified NaN payload. -/
def f16EncodeFiniteRN (negative : Bool) (magnitude : Nat) : BitVec 16 :=
  let signBits := if negative then 32768 else 0
  if magnitude = 0 then
    BitVec.ofNat 16 signBits
  else if magnitude < 1024 then
    BitVec.ofNat 16 (signBits + magnitude)
  else
    let top := Nat.log2 magnitude
    let shift := top - 10
    let scale := 2 ^ shift
    let q := magnitude / scale
    let remainder := magnitude % scale
    let halfway := scale / 2
    let roundUp := shift ≠ 0 && (halfway < remainder || (remainder = halfway && q % 2 = 1))
    let rounded := if roundUp then q + 1 else q
    let carry := rounded = 2048
    let significand := if carry then 1024 else rounded
    let roundedTop := if carry then top + 1 else top
    let exponent := roundedTop - 9
    if 31 ≤ exponent then
      BitVec.ofNat 16 (signBits + 31744)
    else
      BitVec.ofNat 16 (signBits + exponent * 1024 + significand - 1024)

/- PTX definition: `add.f16` adds two binary16 values with default `.rn`; NaN results have unspecified payload. -/
/- Lean modeling: special values are handled explicitly, finite addition is exact before one RN-even rounding, and all NaN cases select canonical `0x7e00`. -/
def addF16RN (a b : BitVec 16) : BitVec 16 :=
  if f16IsNaN a || f16IsNaN b then
    BitVec.ofNat 16 0x7e00
  else if f16IsInf a then
    if f16IsInf b && f16Sign a != f16Sign b then BitVec.ofNat 16 0x7e00
    else BitVec.ofNat 16 (if f16Sign a then 0xfc00 else 0x7c00)
  else if f16IsInf b then
    BitVec.ofNat 16 (if f16Sign b then 0xfc00 else 0x7c00)
  else
    let total := f16FiniteUnits a + f16FiniteUnits b
    if total = 0 then
      let negativeZero := f16IsZero a && f16IsZero b && f16Sign a && f16Sign b
      BitVec.ofNat 16 (if negativeZero then 0x8000 else 0)
    else
      f16EncodeFiniteRN (total < 0) total.natAbs

/- PTX definition: `add.f16x2` independently adds bits 0..15 and 16..31 as binary16 lanes. -/
/- Lean modeling: lane extraction, `addF16RN`, and repacking are width-generic only so `CTemplate.evalC` can remain total; the named PTX instruction fixes width 32. -/
def addF16x2At {w : Nat} (a b : BitVec w) : BitVec w :=
  let alo := BitVec.ofNat 16 (a.toNat % 65536)
  let ahi := BitVec.ofNat 16 ((a.toNat / 65536) % 65536)
  let blo := BitVec.ofNat 16 (b.toNat % 65536)
  let bhi := BitVec.ofNat 16 ((b.toNat / 65536) % 65536)
  let lo := addF16RN alo blo
  let hi := addF16RN ahi bhi
  BitVec.ofNat w (lo.toNat + hi.toNat * 65536)

def addF16x2 (a b : BitVec 32) : BitVec 32 := addF16x2At a b

/-! ## Packed bfloat16 multiplication -/

/- PTX definition: bfloat16 has one sign bit, eight exponent bits, and seven fraction bits. -/
/- Lean modeling: these projections classify a lane directly from its 16-bit register representation. -/
def bf16Sign (x : BitVec 16) : Bool := x.toNat / 32768 = 1
def bf16Exp (x : BitVec 16) : Nat := (x.toNat / 128) % 256
def bf16Frac (x : BitVec 16) : Nat := x.toNat % 128

/- PTX definition: bfloat16 arithmetic accepts finite values, infinities, and NaNs. -/
/- Lean modeling: exact exponent/fraction predicates support explicit special-value handling and a canonical quiet NaN result. -/
def bf16IsNaN (x : BitVec 16) : Bool := bf16Exp x = 255 && bf16Frac x != 0
def bf16IsInf (x : BitVec 16) : Bool := bf16Exp x = 255 && bf16Frac x == 0
def bf16IsZero (x : BitVec 16) : Bool := bf16Exp x = 0 && bf16Frac x == 0

/- PTX definition: every finite bfloat16 value is sign times an integer significand times a power of two. -/
/- Lean modeling: the unsigned magnitude is represented exactly in units of `2^-133`; multiplication therefore produces exact `2^-266` units before rounding. -/
def bf16FiniteMagnitude (x : BitVec 16) : Nat :=
  let e := bf16Exp x
  let f := bf16Frac x
  if e = 0 then f else (128 + f) * 2 ^ (e - 1)

/- PTX definition: plain bfloat16 multiplication rounds to nearest with ties to even, preserves subnormals, and overflows to infinity. -/
/- Lean modeling: an exact product in `2^-266` units is rounded either to the subnormal `2^-133` grid or to an eight-bit normal significand. -/
def bf16EncodeProductRN (negative : Bool) (magnitude : Nat) : BitVec 16 :=
  let signBits := if negative then 32768 else 0
  if magnitude = 0 then
    BitVec.ofNat 16 signBits
  else if magnitude < 2 ^ 140 then
    let scale := 2 ^ 133
    let q := magnitude / scale
    let remainder := magnitude % scale
    let halfway := scale / 2
    let roundUp := halfway < remainder || (remainder = halfway && q % 2 = 1)
    let rounded := if roundUp then q + 1 else q
    if 128 ≤ rounded then
      BitVec.ofNat 16 (signBits + 128)
    else
      BitVec.ofNat 16 (signBits + rounded)
  else
    let top := Nat.log2 magnitude
    let shift := top - 7
    let scale := 2 ^ shift
    let q := magnitude / scale
    let remainder := magnitude % scale
    let halfway := scale / 2
    let roundUp := halfway < remainder || (remainder = halfway && q % 2 = 1)
    let rounded := if roundUp then q + 1 else q
    let carry := rounded = 256
    let significand := if carry then 128 else rounded
    let roundedTop := if carry then top + 1 else top
    let exponent := roundedTop - 139
    if 255 ≤ exponent then
      BitVec.ofNat 16 (signBits + 32640)
    else
      BitVec.ofNat 16 (signBits + exponent * 128 + significand - 128)

/- PTX definition: `mul.bf16` multiplies two lanes with default RN behavior; NaN payload selection is unspecified. -/
/- Lean modeling: special values follow IEEE multiplication, finite values multiply exactly before one RN-even rounding, and NaN results use canonical `0x7fc0`. -/
def mulBF16RN (a b : BitVec 16) : BitVec 16 :=
  let negative := bf16Sign a != bf16Sign b
  if bf16IsNaN a || bf16IsNaN b then
    BitVec.ofNat 16 0x7fc0
  else if (bf16IsInf a && bf16IsZero b) || (bf16IsZero a && bf16IsInf b) then
    BitVec.ofNat 16 0x7fc0
  else if bf16IsInf a || bf16IsInf b then
    BitVec.ofNat 16 (if negative then 0xff80 else 0x7f80)
  else
    bf16EncodeProductRN negative (bf16FiniteMagnitude a * bf16FiniteMagnitude b)

/- PTX definition: `mul.bf16x2` independently multiplies bits 0..15 and 16..31 as bfloat16 lanes. -/
/- Lean modeling: lane extraction, exact scalar multiplication, and repacking are width-generic only for total template evaluation; the instruction fixes width 32. -/
def mulBF16x2At {w : Nat} (a b : BitVec w) : BitVec w :=
  let alo := BitVec.ofNat 16 (a.toNat % 65536)
  let ahi := BitVec.ofNat 16 ((a.toNat / 65536) % 65536)
  let blo := BitVec.ofNat 16 (b.toNat % 65536)
  let bhi := BitVec.ofNat 16 ((b.toNat / 65536) % 65536)
  let lo := mulBF16RN alo blo
  let hi := mulBF16RN ahi bhi
  BitVec.ofNat w (lo.toNat + hi.toNat * 65536)

def mulBF16x2 (a b : BitVec 32) : BitVec 32 := mulBF16x2At a b

/- PTX definition: unsaturated scalar `sub` computes `d = a - b`. -/
/- Lean modeling: `BitVec w` subtraction performs the same fixed-width modulo-`2^w` subtraction. -/
def sub  (a b : BitVec w) : BitVec w := a - b

/- PTX definition: `sub.u16` stores the low 16 bits of the unsigned difference. -/
/- Lean modeling: modulo-`2^16` subtraction implements GX's documented strict fixed-width target policy. -/
def subU16 (a b : BitVec 16) : BitVec 16 := sub a b

/- PTX definition: `sub.s16` stores the 16-bit two's-complement difference pattern. -/
/- Lean modeling: low signed/unsigned difference bits coincide under GX's documented strict fixed-width 16-bit policy. -/
def subS16 (a b : BitVec 16) : BitVec 16 := sub a b

/- PTX definition: `sub.u32` stores the low 32 bits of the unsigned difference. -/
/- Lean modeling: width-32 `BitVec` subtraction computes the required result modulo `2^32`. -/
def subU32 (a b : BitVec 32) : BitVec 32 := sub a b

/- PTX definition: `sub.s32` stores the 32-bit two's-complement difference pattern. -/
/- Lean modeling: signed and unsigned subtraction share the same low 32 bits, so width-32 `sub` is exact. -/
def subS32 (a b : BitVec 32) : BitVec 32 := sub a b

/- PTX definition: `sub.u64` stores the low 64 bits of the unsigned difference. -/
/- Lean modeling: width-64 `BitVec` subtraction computes the required result modulo `2^64`. -/
def subU64 (a b : BitVec 64) : BitVec 64 := sub a b

/- PTX definition: `sub.s64` stores the 64-bit two's-complement difference pattern. -/
/- Lean modeling: signed and unsigned subtraction share the same low 64 bits, so width-64 `sub` is exact. -/
def subS64 (a b : BitVec 64) : BitVec 64 := sub a b

/-! ## Signed absolute value and negation -/

/- PTX definition: signed integer `abs` returns `-a` when `a < 0` and returns `a` otherwise. -/
/- Lean modeling: `toInt` tests the sign and modular `BitVec` negation defines GX's fixed-width minimum-pattern policy, recorded separately because the PDF does not resolve that edge. -/
def absS (a : BitVec w) : BitVec w := if a.toInt < 0 then -a else a

/- PTX definition: `abs.s16` applies signed absolute value to a 16-bit operand. -/
/- Lean modeling: the width-16 specialization implements the ordinary absolute value rule and GX's explicit modular policy for the minimum pattern. -/
def absS16 (a : BitVec 16) : BitVec 16 := absS a

/- PTX definition: `abs.s32` applies signed absolute value to a 32-bit operand. -/
/- Lean modeling: the width-32 specialization implements the ordinary absolute value rule and GX's explicit modular policy for the minimum pattern. -/
def absS32 (a : BitVec 32) : BitVec 32 := absS a

/- PTX definition: `abs.s64` applies signed absolute value to a 64-bit operand. -/
/- Lean modeling: the width-64 specialization implements the ordinary absolute value rule and GX's explicit modular policy for the minimum pattern. -/
def absS64 (a : BitVec 64) : BitVec 64 := absS a

/- PTX definition: scalar signed `neg` computes `d = -a`. -/
/- Lean modeling: `BitVec w` negation is two's-complement negation modulo `2^w`, including the minimum signed value. -/
def neg  (a : BitVec w) : BitVec w := -a

/- PTX definition: `neg.s16` stores the 16-bit two's-complement negation of its operand. -/
/- Lean modeling: width-16 `BitVec` negation is the required modular two's-complement operation. -/
def negS16 (a : BitVec 16) : BitVec 16 := neg a

/- PTX definition: `neg.s32` stores the 32-bit two's-complement negation of its operand. -/
/- Lean modeling: width-32 `BitVec` negation is the required modular two's-complement operation. -/
def negS32 (a : BitVec 32) : BitVec 32 := neg a

/- PTX definition: `neg.s64` stores the 64-bit two's-complement negation of its operand. -/
/- Lean modeling: width-64 `BitVec` negation is the required modular two's-complement operation. -/
def negS64 (a : BitVec 64) : BitVec 64 := neg a

/-! ## Integer division and remainder -/

/- PTX definition: unsigned `div` computes `a / b`, while division by zero is unspecified and machine-specific. -/
/- Lean modeling: GX deliberately retains its existing policy `a / 0 = allOnes`; otherwise `BitVec` division is the unsigned quotient. -/
def divU (a b : BitVec w) : BitVec w :=
  if b = 0 then BitVec.allOnes w else a / b

/- PTX definition: `div.u16` computes the unsigned 16-bit quotient when the divisor is nonzero. -/
/- Lean modeling: this specialization uses unsigned `BitVec` division and preserves GX's explicit unspecified-case policy. -/
def divU16 (a b : BitVec 16) : BitVec 16 := divU a b

/- PTX definition: `div.u32` computes the unsigned 32-bit quotient when the divisor is nonzero. -/
/- Lean modeling: this specialization uses unsigned `BitVec` division and preserves GX's explicit unspecified-case policy. -/
def divU32 (a b : BitVec 32) : BitVec 32 := divU a b

/- PTX definition: `div.u64` computes the unsigned 64-bit quotient when the divisor is nonzero. -/
/- Lean modeling: this specialization uses unsigned `BitVec` division and preserves GX's explicit unspecified-case policy. -/
def divU64 (a b : BitVec 64) : BitVec 64 := divU a b

/- PTX definition: signed `div` computes `a / b`, while division by zero is unspecified and machine-specific. -/
/- Lean modeling: GX deliberately retains its all-ones zero-divisor policy and uses `Int.tdiv` for its existing truncation-toward-zero signed policy. -/
def divS (a b : BitVec w) : BitVec w :=
  if b = 0 then BitVec.allOnes w else BitVec.ofInt w (a.toInt.tdiv b.toInt)

/- PTX definition: `div.s16` computes the signed 16-bit quotient when the divisor is nonzero. -/
/- Lean modeling: signed interpretation plus `Int.tdiv` gives truncation toward zero and preserves GX's unspecified-case policy. -/
def divS16 (a b : BitVec 16) : BitVec 16 := divS a b

/- PTX definition: `div.s32` computes the signed 32-bit quotient when the divisor is nonzero. -/
/- Lean modeling: signed interpretation plus `Int.tdiv` gives truncation toward zero and preserves GX's unspecified-case policy. -/
def divS32 (a b : BitVec 32) : BitVec 32 := divS a b

/- PTX definition: `div.s64` computes the signed 64-bit quotient when the divisor is nonzero. -/
/- Lean modeling: signed interpretation plus `Int.tdiv` gives truncation toward zero and preserves GX's unspecified-case policy. -/
def divS64 (a b : BitVec 64) : BitVec 64 := divS a b

/- PTX definition: unsigned `rem` computes `a % b`, while division by zero is unspecified and machine-specific. -/
/- Lean modeling: GX deliberately retains its existing policy `a % 0 = a`; otherwise `BitVec` remainder is the unsigned remainder. -/
def remU (a b : BitVec w) : BitVec w :=
  if b = 0 then a else a % b

/- PTX definition: `rem.u16` computes the unsigned 16-bit division remainder for a nonzero divisor. -/
/- Lean modeling: unsigned `BitVec` remainder is exact and the wrapper preserves GX's explicit unspecified-case policy. -/
def remU16 (a b : BitVec 16) : BitVec 16 := remU a b

/- PTX definition: `rem.u32` computes the unsigned 32-bit division remainder for a nonzero divisor. -/
/- Lean modeling: unsigned `BitVec` remainder is exact and the wrapper preserves GX's explicit unspecified-case policy. -/
def remU32 (a b : BitVec 32) : BitVec 32 := remU a b

/- PTX definition: `rem.u64` computes the unsigned 64-bit division remainder for a nonzero divisor. -/
/- Lean modeling: unsigned `BitVec` remainder is exact and the wrapper preserves GX's explicit unspecified-case policy. -/
def remU64 (a b : BitVec 64) : BitVec 64 := remU a b

/- PTX definition: signed remainder behavior for negatives is machine-dependent and division by zero is unspecified. -/
/- Lean modeling: GX deliberately retains `a % 0 = a` and `Int.tmod`, whose result follows the dividend and preserves the existing C-style policy. -/
def remS (a b : BitVec w) : BitVec w :=
  if b = 0 then a else BitVec.ofInt w (a.toInt.tmod b.toInt)

/- PTX definition: `rem.s16` computes the signed 16-bit division remainder for a nonzero divisor. -/
/- Lean modeling: signed interpretation plus `Int.tmod` matches GX's C-style policy, including its chosen unspecified cases. -/
def remS16 (a b : BitVec 16) : BitVec 16 := remS a b

/- PTX definition: `rem.s32` computes the signed 32-bit division remainder for a nonzero divisor. -/
/- Lean modeling: signed interpretation plus `Int.tmod` matches GX's C-style policy, including its chosen unspecified cases. -/
def remS32 (a b : BitVec 32) : BitVec 32 := remS a b

/- PTX definition: `rem.s64` computes the signed 64-bit division remainder for a nonzero divisor. -/
/- Lean modeling: signed interpretation plus `Int.tmod` matches GX's C-style policy, including its chosen unspecified cases. -/
def remS64 (a b : BitVec 64) : BitVec 64 := remS a b

/-! ## Integer minimum and maximum -/

/- PTX definition: unsigned `min` stores the lesser of `a` and `b`. -/
/- Lean modeling: the unsigned `BitVec` order selects `a` exactly when `a ≤ b`, with equality observationally irrelevant. -/
def minU (a b : BitVec w) : BitVec w := if a ≤ b then a else b

/- PTX definition: `min.u16` selects the lesser unsigned 16-bit operand. -/
/- Lean modeling: unsigned `BitVec 16` order implements GX's documented strict fixed-width target policy. -/
def minU16 (a b : BitVec 16) : BitVec 16 := minU a b

/- PTX definition: `min.u32` selects the lesser unsigned 32-bit operand. -/
/- Lean modeling: specializing `minU` to width 32 applies the exact unsigned `BitVec` order. -/
def minU32 (a b : BitVec 32) : BitVec 32 := minU a b

/- PTX definition: `min.u64` selects the lesser unsigned 64-bit operand. -/
/- Lean modeling: specializing `minU` to width 64 applies the exact unsigned `BitVec` order. -/
def minU64 (a b : BitVec 64) : BitVec 64 := minU a b

/- PTX definition: signed `min` stores the lesser signed integer operand. -/
/- Lean modeling: comparing `BitVec.toInt` values interprets both bit patterns as width-`w` two's-complement integers. -/
def minS (a b : BitVec w) : BitVec w := if a.toInt ≤ b.toInt then a else b

/- PTX definition: `min.s16` selects the lesser signed 16-bit operand. -/
/- Lean modeling: signed `BitVec 16` order implements GX's documented strict fixed-width target policy. -/
def minS16 (a b : BitVec 16) : BitVec 16 := minS a b

/- PTX definition: `min.s32` selects the lesser signed 32-bit operand. -/
/- Lean modeling: specializing `minS` to width 32 compares the exact two's-complement interpretations. -/
def minS32 (a b : BitVec 32) : BitVec 32 := minS a b

/- PTX definition: `min.s64` selects the lesser signed 64-bit operand. -/
/- Lean modeling: specializing `minS` to width 64 compares the exact two's-complement interpretations. -/
def minS64 (a b : BitVec 64) : BitVec 64 := minS a b

/- PTX definition: unsigned `max` stores the greater of `a` and `b`. -/
/- Lean modeling: the unsigned `BitVec` order selects `a` exactly when `b ≤ a`, with equality observationally irrelevant. -/
def maxU (a b : BitVec w) : BitVec w := if b ≤ a then a else b

/- PTX definition: `max.u16` selects the greater unsigned 16-bit operand. -/
/- Lean modeling: unsigned `BitVec 16` order implements GX's documented strict fixed-width target policy. -/
def maxU16 (a b : BitVec 16) : BitVec 16 := maxU a b

/- PTX definition: `max.u32` selects the greater unsigned 32-bit operand. -/
/- Lean modeling: specializing `maxU` to width 32 applies the exact unsigned `BitVec` order. -/
def maxU32 (a b : BitVec 32) : BitVec 32 := maxU a b

/- PTX definition: `max.u64` selects the greater unsigned 64-bit operand. -/
/- Lean modeling: specializing `maxU` to width 64 applies the exact unsigned `BitVec` order. -/
def maxU64 (a b : BitVec 64) : BitVec 64 := maxU a b

/- PTX definition: signed `max` stores the greater signed integer operand. -/
/- Lean modeling: comparing `BitVec.toInt` values interprets both bit patterns as width-`w` two's-complement integers. -/
def maxS (a b : BitVec w) : BitVec w := if b.toInt ≤ a.toInt then a else b

/- PTX definition: `max.s16` selects the greater signed 16-bit operand. -/
/- Lean modeling: signed `BitVec 16` order implements GX's documented strict fixed-width target policy. -/
def maxS16 (a b : BitVec 16) : BitVec 16 := maxS a b

/- PTX definition: `max.s32` selects the greater signed 32-bit operand. -/
/- Lean modeling: specializing `maxS` to width 32 compares the exact two's-complement interpretations. -/
def maxS32 (a b : BitVec 32) : BitVec 32 := maxS a b

/- PTX definition: `max.s64` selects the greater signed 64-bit operand. -/
/- Lean modeling: specializing `maxS` to width 64 compares the exact two's-complement interpretations. -/
def maxS64 (a b : BitVec 64) : BitVec 64 := maxS a b

/-! ## Integer multiply and multiply-add -/

/- PTX definition: `mul.lo` writes bits `n-1..0` for the legal 16-, 32-, and 64-bit integer types. -/
/- Lean modeling: `BitVec w` multiplication retains the low `w` bits for any width; it denotes a PTX instruction only at an ISA-legal specialization. -/
def mulLo (a b : BitVec w) : BitVec w := a * b

/- PTX definition: `mul.lo.u16` writes bits 15 through 0 of the full unsigned product. -/
/- Lean modeling: width-16 multiplication retains the low half under GX's documented strict fixed-width target policy. -/
def mulLoU16 (a b : BitVec 16) : BitVec 16 := mulLo a b

/- PTX definition: `mul.lo.s16` writes bits 15 through 0 of the full signed product. -/
/- Lean modeling: signedness leaves the low product bits unchanged under GX's strict fixed-width 16-bit policy. -/
def mulLoS16 (a b : BitVec 16) : BitVec 16 := mulLo a b

/- PTX definition: `mul.lo.u32` writes bits 31 through 0 of the full unsigned product. -/
/- Lean modeling: instantiating the width-generic `mulLo` at 32 bits retains exactly that low-half bit pattern. -/
def mulLoU32 (a b : BitVec 32) : BitVec 32 := mulLo a b

/- PTX definition: `mul.lo.s32` writes bits 31 through 0 of the full signed product. -/
/- Lean modeling: signedness does not change the low 32 product bits, so the 32-bit `mulLo` instance is exact. -/
def mulLoS32 (a b : BitVec 32) : BitVec 32 := mulLo a b

/- PTX definition: `mul.lo.u64` writes bits 63 through 0 of the full unsigned product. -/
/- Lean modeling: instantiating the width-generic `mulLo` at 64 bits retains exactly that low-half bit pattern. -/
def mulLoU64 (a b : BitVec 64) : BitVec 64 := mulLo a b

/- PTX definition: `mul.lo.s64` writes bits 63 through 0 of the full signed product. -/
/- Lean modeling: signedness does not change the low 64 product bits, so the 64-bit `mulLo` instance is exact. -/
def mulLoS64 (a b : BitVec 64) : BitVec 64 := mulLo a b

/- PTX definition: `mul.wide.uN` writes the full 2N-bit product only for the legal N = 16 or 32 forms. -/
/- Lean modeling: `BitVec.ofNat (2 * w)` is useful generically, but it denotes PTX instruction semantics only when specialized by the legal U16/U32 wrappers. -/
def mulWideU {w : Nat} (a b : BitVec w) : BitVec (2 * w) :=
  BitVec.ofNat (2 * w) (a.toNat * b.toNat)

/- PTX definition: `mul.wide.sN` writes the full 2N-bit product only for the legal N = 16 or 32 forms. -/
/- Lean modeling: `BitVec.ofInt (2 * w)` is useful generically, but it denotes PTX instruction semantics only when specialized by the legal S16/S32 wrappers. -/
def mulWideS {w : Nat} (a b : BitVec w) : BitVec (2 * w) :=
  BitVec.ofInt (2 * w) (a.toInt * b.toInt)

/- PTX definition: `mul.wide.u16` writes the complete unsigned 16-by-16 product to a 32-bit destination. -/
/- Lean modeling: this wrapper fixes `mulWideU` to 16-bit operands and its definitionally equal 32-bit result width. -/
def mulWideU16 (a b : BitVec 16) : BitVec 32 := mulWideU a b

/- PTX definition: `mul.wide.s16` writes the complete signed 16-by-16 product to a 32-bit destination. -/
/- Lean modeling: this wrapper fixes `mulWideS` to 16-bit signed operands and its definitionally equal 32-bit result width. -/
def mulWideS16 (a b : BitVec 16) : BitVec 32 := mulWideS a b

/- PTX definition: `mul.wide.u32` writes the complete unsigned 32-by-32 product to a 64-bit destination. -/
/- Lean modeling: this wrapper fixes `mulWideU` to 32-bit operands and its definitionally equal 64-bit result width. -/
def mulWideU32 (a b : BitVec 32) : BitVec 64 := mulWideU a b

/- PTX definition: `mul.wide.s32` writes the complete signed 32-by-32 product to a 64-bit destination. -/
/- Lean modeling: this wrapper fixes `mulWideS` to 32-bit signed operands and its definitionally equal 64-bit result width. -/
def mulWideS32 (a b : BitVec 32) : BitVec 64 := mulWideS a b

/- PTX definition: `mul.hi.uN` writes bits 2N-1 through N only for legal N = 16, 32, or 64 forms. -/
/- Lean modeling: the generic shift/projection is algebraically valid for any `w`, but denotes PTX only at the explicit legal-width wrappers. -/
def mulHiU {w : Nat} (a b : BitVec w) : BitVec w :=
  ((mulWideU a b) >>> w).setWidth w

/- PTX definition: `mul.hi.sN` writes bits 2N-1 through N only for legal N = 16, 32, or 64 forms. -/
/- Lean modeling: the generic arithmetic shift/projection is valid for any `w`, but denotes PTX only at the explicit legal-width wrappers. -/
def mulHiS {w : Nat} (a b : BitVec w) : BitVec w :=
  ((mulWideS a b).sshiftRight w).setWidth w

/- PTX definition: `mul.hi.u16` writes bits 31 through 16 of the full unsigned product. -/
/- Lean modeling: the 32-bit unsigned product projection implements GX's strict fixed-width 16-bit target policy. -/
def mulHiU16 (a b : BitVec 16) : BitVec 16 := mulHiU a b

/- PTX definition: `mul.hi.s16` writes bits 31 through 16 of the full signed product. -/
/- Lean modeling: the 32-bit signed product projection implements GX's strict fixed-width 16-bit target policy. -/
def mulHiS16 (a b : BitVec 16) : BitVec 16 := mulHiS a b

/- PTX definition: `mul.hi.u32` writes bits 63 through 32 of the full unsigned product. -/
/- Lean modeling: instantiating `mulHiU` at 32 bits selects exactly that unsigned high-half pattern. -/
def mulHiU32 (a b : BitVec 32) : BitVec 32 := mulHiU a b

/- PTX definition: `mul.hi.s32` writes bits 63 through 32 of the full signed product. -/
/- Lean modeling: instantiating `mulHiS` at 32 bits selects exactly that two's-complement high-half pattern. -/
def mulHiS32 (a b : BitVec 32) : BitVec 32 := mulHiS a b

/- PTX definition: `mul.hi.u64` writes bits 127 through 64 of the full unsigned product. -/
/- Lean modeling: instantiating `mulHiU` at 64 bits selects exactly that unsigned high-half pattern. -/
def mulHiU64 (a b : BitVec 64) : BitVec 64 := mulHiU a b

/- PTX definition: `mul.hi.s64` writes bits 127 through 64 of the full signed product. -/
/- Lean modeling: instantiating `mulHiS` at 64 bits selects exactly that two's-complement high-half pattern. -/
def mulHiS64 (a b : BitVec 64) : BitVec 64 := mulHiS a b

/- PTX definition: `mad.lo` adds `c` to the low product for legal 16-, 32-, and 64-bit integer forms. -/
/- Lean modeling: `BitVec w` multiplication/addition is valid generically, but denotes a PTX instruction only at an ISA-legal width. -/
def madLo (a b c : BitVec w) : BitVec w := a * b + c

/- PTX definition: `mad.lo.u16` adds a 16-bit addend to the low 16 product bits. -/
/- Lean modeling: modular multiply-add implements GX's documented strict fixed-width 16-bit target policy. -/
def madLoU16 (a b c : BitVec 16) : BitVec 16 := madLo a b c

/- PTX definition: `mad.lo.s16` adds a signed addend to the low 16 signed-product bits. -/
/- Lean modeling: low two's-complement multiply-add bits are identical to width-16 modular arithmetic. -/
def madLoS16 (a b c : BitVec 16) : BitVec 16 := madLo a b c

/- PTX definition: `mad.lo.u32` adds a 32-bit addend to the low 32 product bits. -/
/- Lean modeling: width-32 `BitVec` multiply and addition compute the result modulo `2^32`. -/
def madLoU32 (a b c : BitVec 32) : BitVec 32 := madLo a b c

/- PTX definition: `mad.lo.s32` stores the low 32 bits of the signed product plus addend. -/
/- Lean modeling: width-32 modular arithmetic gives the same low two's-complement result bits without signed-overflow assumptions. -/
def madLoS32 (a b c : BitVec 32) : BitVec 32 := madLo a b c

/- PTX definition: `mad.lo.u64` adds a 64-bit addend to the low 64 product bits. -/
/- Lean modeling: width-64 `BitVec` multiply and addition compute the result modulo `2^64`. -/
def madLoU64 (a b c : BitVec 64) : BitVec 64 := madLo a b c

/- PTX definition: `mad.lo.s64` stores the low 64 bits of the signed product plus addend. -/
/- Lean modeling: width-64 modular arithmetic gives the same low two's-complement result bits without signed-overflow assumptions. -/
def madLoS64 (a b c : BitVec 64) : BitVec 64 := madLo a b c

/- PTX definition: `mad.hi.uN` adds `c` to the high product half only for legal N = 16, 32, or 64 forms. -/
/- Lean modeling: the composition is valid for any `w`, but denotes PTX only when specialized to one of those legal widths. -/
def madHiU {w : Nat} (a b c : BitVec w) : BitVec w := mulHiU a b + c

/- PTX definition: `mad.hi.sN` adds `c` to the high product half only for legal N = 16, 32, or 64 forms. -/
/- Lean modeling: the composition is valid for any `w`, but denotes PTX only when specialized to one of those legal widths. -/
def madHiS {w : Nat} (a b c : BitVec w) : BitVec w := mulHiS a b + c

/- PTX definition: `mad.hi.u16` adds a 16-bit addend to unsigned product bits 31 through 16. -/
/- Lean modeling: specializing `madHiU` at 16 bits selects that high half and wraps the final addition at 16 bits. -/
def madHiU16 (a b c : BitVec 16) : BitVec 16 := madHiU a b c

/- PTX definition: `mad.hi.s16` adds a 16-bit addend to signed product bits 31 through 16. -/
/- Lean modeling: specializing `madHiS` at 16 bits uses signed high-half extraction and width-16 wrapping addition. -/
def madHiS16 (a b c : BitVec 16) : BitVec 16 := madHiS a b c

/- PTX definition: `mad.hi.u32` adds a 32-bit addend to unsigned product bits 63 through 32. -/
/- Lean modeling: specializing `madHiU` at 32 bits selects that high half and wraps the final addition at 32 bits. -/
def madHiU32 (a b c : BitVec 32) : BitVec 32 := madHiU a b c

/- PTX definition: `mad.hi.s32` adds a 32-bit addend to signed product bits 63 through 32. -/
/- Lean modeling: specializing `madHiS` at 32 bits uses signed high-half extraction and width-32 wrapping addition. -/
def madHiS32 (a b c : BitVec 32) : BitVec 32 := madHiS a b c

/- PTX definition: `mad.hi.u64` adds a 64-bit addend to unsigned product bits 127 through 64. -/
/- Lean modeling: specializing `madHiU` at 64 bits selects that high half and wraps the final addition at 64 bits. -/
def madHiU64 (a b c : BitVec 64) : BitVec 64 := madHiU a b c

/- PTX definition: `mad.hi.s64` adds a 64-bit addend to signed product bits 127 through 64. -/
/- Lean modeling: specializing `madHiS` at 64 bits uses signed high-half extraction and width-64 wrapping addition. -/
def madHiS64 (a b c : BitVec 64) : BitVec 64 := madHiS a b c

/- PTX definition: `mad.wide.uN` adds a 2N-bit addend to the complete product for legal N = 16 or 32. -/
/- Lean modeling: the full `mulWideU` product and a same-width `BitVec (2*w)` addend give the specified wrapping 2N-bit sum. -/
def madWideU {w : Nat} (a b : BitVec w) (c : BitVec (2 * w)) : BitVec (2 * w) :=
  mulWideU a b + c

/- PTX definition: `mad.wide.sN` adds a 2N-bit addend to the complete signed product for legal N = 16 or 32. -/
/- Lean modeling: `mulWideS` supplies the 2N-bit two's-complement product and fixed-width addition supplies the required result bits. -/
def madWideS {w : Nat} (a b : BitVec w) (c : BitVec (2 * w)) : BitVec (2 * w) :=
  mulWideS a b + c

/- PTX definition: `mad.wide.u16` adds a 32-bit addend to the full 16-by-16 unsigned product. -/
/- Lean modeling: the explicit 16/32-bit signature enforces the instruction's source and destination widths. -/
def madWideU16 (a b : BitVec 16) (c : BitVec 32) : BitVec 32 := madWideU a b c

/- PTX definition: `mad.wide.s16` adds a 32-bit addend to the full 16-by-16 signed product. -/
/- Lean modeling: the explicit signature and signed widening kernel produce exactly the 32-bit two's-complement result pattern. -/
def madWideS16 (a b : BitVec 16) (c : BitVec 32) : BitVec 32 := madWideS a b c

/- PTX definition: `mad.wide.u32` adds a 64-bit addend to the full 32-by-32 unsigned product. -/
/- Lean modeling: the explicit 32/64-bit signature enforces the instruction's source and destination widths. -/
def madWideU32 (a b : BitVec 32) (c : BitVec 64) : BitVec 64 := madWideU a b c

/- PTX definition: `mad.wide.s32` adds a 64-bit addend to the full 32-by-32 signed product. -/
/- Lean modeling: the explicit signature and signed widening kernel produce exactly the 64-bit two's-complement result pattern. -/
def madWideS32 (a b : BitVec 32) (c : BitVec 64) : BitVec 64 := madWideS a b c

end Gxptx.Ptx
