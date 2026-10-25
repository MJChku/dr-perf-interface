/-
  Characterization and C-equivalence proofs for scalar PTX arithmetic,
  including every legal non-saturating integer multiply/multiply-add shape.

  These theorems make the specification boundary for `Ptx.add` explicit:
  the PDF equation is transcribed as a shallow `BitVec` function, its modular
  unsigned and two's-complement signed readings are characterized separately,
  and the named C templates are then proved equal to those PTX definitions.
-/
import Gxptx.Semantics.PTX.Arithmetic
import Gxptx.Semantics.C.Expression
import Gxptx.Semantics.C.Template
import Gxptx.Verification.Template

namespace Gxptx.Ptx

/- PTX definition: unsaturated scalar `add` stores the low `w` bits of the mathematical sum. -/
/- Lean modeling: unfolding `Ptx.add` and `BitVec.toNat_add` yields the specification's exact modulo-`2^w` characterization for every modeled width. -/
theorem add_toNat_mod {w : Nat} (a b : BitVec w) :
    (add a b).toNat = (a.toNat + b.toNat) % 2 ^ w := by
  rfl

/- PTX definition: signed scalar `add` stores the width-`w` two's-complement representation of the integer sum. -/
/- Lean modeling: `BitVec.toInt_add` shows directly that `Ptx.add` has the signed sum reduced into the representable width-`w` interval. -/
theorem add_toInt_bmod {w : Nat} (a b : BitVec w) :
    (add a b).toInt = (a.toInt + b.toInt).bmod (2 ^ w) := by
  simp [add]

/- PTX definition: unsaturated `add.u32` returns the low 32 bits of the mathematical sum. -/
/- Lean modeling: `BitVec.toNat_add` exposes exactly the required reduction modulo `2^32`, independently characterizing the shallow definition's result. -/
theorem addU32_mod (a b : BitVec 32) :
    (addU32 a b).toNat = (a.toNat + b.toNat) % 2 ^ 32 := by
  exact add_toNat_mod a b

/- PTX definition: unsaturated `add.s32` adds the signed operands and stores the resulting 32-bit two's-complement pattern. -/
/- Lean modeling: `BitVec.ofInt_add` proves that wrapping the mathematical signed sum is identical to the width-32 `BitVec` addition used by `Ptx.addS32`. -/
theorem addS32_twosComplement (a b : BitVec 32) :
    addS32 a b = BitVec.ofInt 32 (a.toInt + b.toInt) := by
  apply BitVec.eq_of_toInt_eq
  simp [addS32, add]

end Gxptx.Ptx

namespace Gxptx.CTemplate

/- PTX definition: `add.u16` stores the low 16 bits of the unsigned sum. -/
/- Lean modeling: `evalC` equals the width-16 PTX value; printed `.w32` truncation remains `gap:subword-template-certificate`. -/
theorem addU16_equiv (a b c : BitVec 16) :
    CTemplate.addU16.evalC a b c = Ptx.addU16 a b := rfl

/- PTX definition: `add.s16` stores the low 16 bits of the signed sum. -/
/- Lean modeling: `evalC` equals the width-16 PTX value; printed `.w32` truncation remains `gap:subword-template-certificate`. -/
theorem addS16_equiv (a b c : BitVec 16) :
    CTemplate.addS16.evalC a b c = Ptx.addS16 a b := rfl

/- PTX definition: `add.u32` computes the fixed-width sum of its two operands. -/
/- Lean modeling: unfolding `CTemplate.addU32`, `evalC`, and `Ptx.addU32` makes both C-side and PTX value models the same 32-bit `BitVec` addition. -/
theorem addU32_equiv (a b c : BitVec 32) :
    CTemplate.addU32.evalC a b c = Ptx.addU32 a b := rfl

/- PTX definition: `add.s32` stores the same low 32 sum bits under signed interpretation. -/
/- Lean modeling: unfolding the named template evaluator and `Ptx.addS32` makes both value models the same wrapping 32-bit `BitVec` addition. -/
theorem addS32_equiv (a b c : BitVec 32) :
    CTemplate.addS32.evalC a b c = Ptx.addS32 a b := rfl

/- PTX definition: `add.u64` computes the fixed-width sum of its two operands. -/
/- Lean modeling: unfolding `CTemplate.addU64`, `evalC`, and `Ptx.addU64` makes both C-side and PTX value models the same 64-bit `BitVec` addition. -/
theorem addU64_equiv (a b c : BitVec 64) :
    CTemplate.addU64.evalC a b c = Ptx.addU64 a b := rfl

/- PTX definition: `add.s64` stores the same low 64 sum bits under signed interpretation. -/
/- Lean modeling: unfolding the named template evaluator and `Ptx.addS64` makes both value models the same wrapping 64-bit `BitVec` addition. -/
theorem addS64_equiv (a b c : BitVec 64) :
    CTemplate.addS64.evalC a b c = Ptx.addS64 a b := rfl

/- PTX definition: `add.f16x2` adds each binary16 lane independently using default round-to-nearest-even. -/
/- Lean modeling: the fixed 32-bit template and named PTX wrapper share the exact special-value, subnormal, rounding, overflow, and packing model. -/
theorem addF16x2_equiv (a b c : BitVec 32) :
    CTemplate.addF16x2.evalC a b c = Ptx.addF16x2 a b := rfl

/- PTX definition: `mul.bf16x2` multiplies each packed bfloat16 lane independently using default round-to-nearest-even. -/
/- Lean modeling: the fixed-width template and named PTX wrapper share the exact special-value, subnormal, rounding, overflow, and packing model. -/
theorem mulBF16x2_equiv (a b c : BitVec 32) :
    CTemplate.mulBF16x2.evalC a b c = Ptx.mulBF16x2 a b := rfl

/- PTX definition: `sub.u16` stores the low 16 bits of the unsigned difference. -/
/- Lean modeling: `evalC` equals the width-16 PTX value; printed `.w32` truncation remains `gap:subword-template-certificate`. -/
theorem subU16_equiv (a b c : BitVec 16) :
    CTemplate.subU16.evalC a b c = Ptx.subU16 a b := rfl

/- PTX definition: `sub.s16` stores the low 16 bits of the signed difference. -/
/- Lean modeling: `evalC` equals the width-16 PTX value; printed `.w32` truncation remains `gap:subword-template-certificate`. -/
theorem subS16_equiv (a b c : BitVec 16) :
    CTemplate.subS16.evalC a b c = Ptx.subS16 a b := rfl

/- PTX definition: `sub.u32` stores the low 32 bits of the unsigned difference. -/
/- Lean modeling: both named shallow models reduce to the same width-32 modular subtraction. -/
theorem subU32_equiv (a b c : BitVec 32) :
    CTemplate.subU32.evalC a b c = Ptx.subU32 a b := rfl

/- PTX definition: `sub.s32` stores the low 32 bits of the signed difference. -/
/- Lean modeling: both named shallow models reduce to the same width-32 modular subtraction. -/
theorem subS32_equiv (a b c : BitVec 32) :
    CTemplate.subS32.evalC a b c = Ptx.subS32 a b := rfl

/- PTX definition: `sub.u64` stores the low 64 bits of the unsigned difference. -/
/- Lean modeling: both named shallow models reduce to the same width-64 modular subtraction. -/
theorem subU64_equiv (a b c : BitVec 64) :
    CTemplate.subU64.evalC a b c = Ptx.subU64 a b := rfl

/- PTX definition: `sub.s64` stores the low 64 bits of the signed difference. -/
/- Lean modeling: both named shallow models reduce to the same width-64 modular subtraction. -/
theorem subS64_equiv (a b c : BitVec 64) :
    CTemplate.subS64.evalC a b c = Ptx.subS64 a b := rfl

/- PTX definition: `div.u16` computes an unsigned quotient for a nonzero divisor. -/
/- Lean modeling: the generic proof gives value equality, while production printing uses separately reviewed width-exact unsigned 16-bit casts. -/
theorem divU16_equiv (a b c : BitVec 16) :
    CTemplate.divU16.evalC a b c = Ptx.divU16 a b := by
  simpa only [CTemplate.divU16, Ptx.divU16] using (udiv_equiv .w32 a b c)

/- PTX definition: `div.s16` computes a signed quotient for a nonzero divisor. -/
/- Lean modeling: the generic proof covers GX's guards, while production printing uses separately reviewed width-exact signed 16-bit casts. -/
theorem divS16_equiv (a b c : BitVec 16) :
    CTemplate.divS16.evalC a b c = Ptx.divS16 a b := by
  simpa only [CTemplate.divS16, Ptx.divS16] using (sdiv_equiv .w32 a b c)

/- PTX definition: `div.u32` computes an unsigned quotient for a nonzero divisor. -/
/- Lean modeling: specializing the generic guarded-helper proof establishes equality of the named 32-bit models. -/
theorem divU32_equiv (a b c : BitVec 32) :
    CTemplate.divU32.evalC a b c = Ptx.divU32 a b := by
  simpa only [CTemplate.divU32, Ptx.divU32] using (udiv_equiv .w32 a b c)

/- PTX definition: `div.s32` computes a signed quotient for a nonzero divisor. -/
/- Lean modeling: specializing the generic proof covers truncation toward zero and GX's exceptional-case guards. -/
theorem divS32_equiv (a b c : BitVec 32) :
    CTemplate.divS32.evalC a b c = Ptx.divS32 a b := by
  simpa only [CTemplate.divS32, Ptx.divS32] using (sdiv_equiv .w32 a b c)

/- PTX definition: `div.u64` computes an unsigned quotient for a nonzero divisor. -/
/- Lean modeling: specializing the generic guarded-helper proof establishes equality of the named 64-bit models. -/
theorem divU64_equiv (a b c : BitVec 64) :
    CTemplate.divU64.evalC a b c = Ptx.divU64 a b := by
  simpa only [CTemplate.divU64, Ptx.divU64] using (udiv_equiv .w64 a b c)

/- PTX definition: `div.s64` computes a signed quotient for a nonzero divisor. -/
/- Lean modeling: specializing the generic proof covers truncation toward zero and GX's exceptional-case guards. -/
theorem divS64_equiv (a b c : BitVec 64) :
    CTemplate.divS64.evalC a b c = Ptx.divS64 a b := by
  simpa only [CTemplate.divS64, Ptx.divS64] using (sdiv_equiv .w64 a b c)

/- PTX definition: `rem.u16` computes an unsigned remainder for a nonzero divisor. -/
/- Lean modeling: the generic proof gives value equality, while production printing uses separately reviewed width-exact unsigned 16-bit casts. -/
theorem remU16_equiv (a b c : BitVec 16) :
    CTemplate.remU16.evalC a b c = Ptx.remU16 a b := by
  simpa only [CTemplate.remU16, Ptx.remU16] using (urem_equiv .w32 a b c)

/- PTX definition: `rem.s16` computes a signed remainder for a nonzero divisor. -/
/- Lean modeling: the generic proof covers GX's guards, while production printing uses separately reviewed width-exact signed 16-bit casts. -/
theorem remS16_equiv (a b c : BitVec 16) :
    CTemplate.remS16.evalC a b c = Ptx.remS16 a b := by
  simpa only [CTemplate.remS16, Ptx.remS16] using (srem_equiv .w32 a b c)

/- PTX definition: `rem.u32` computes an unsigned remainder for a nonzero divisor. -/
/- Lean modeling: specializing the generic guarded-helper proof establishes equality of the named 32-bit models. -/
theorem remU32_equiv (a b c : BitVec 32) :
    CTemplate.remU32.evalC a b c = Ptx.remU32 a b := by
  simpa only [CTemplate.remU32, Ptx.remU32] using (urem_equiv .w32 a b c)

/- PTX definition: `rem.s32` computes a signed remainder for a nonzero divisor. -/
/- Lean modeling: specializing the generic proof covers GX's C-style remainder and exceptional-case guards. -/
theorem remS32_equiv (a b c : BitVec 32) :
    CTemplate.remS32.evalC a b c = Ptx.remS32 a b := by
  simpa only [CTemplate.remS32, Ptx.remS32] using (srem_equiv .w32 a b c)

/- PTX definition: `rem.u64` computes an unsigned remainder for a nonzero divisor. -/
/- Lean modeling: specializing the generic guarded-helper proof establishes equality of the named 64-bit models. -/
theorem remU64_equiv (a b c : BitVec 64) :
    CTemplate.remU64.evalC a b c = Ptx.remU64 a b := by
  simpa only [CTemplate.remU64, Ptx.remU64] using (urem_equiv .w64 a b c)

/- PTX definition: `rem.s64` computes a signed remainder for a nonzero divisor. -/
/- Lean modeling: specializing the generic proof covers GX's C-style remainder and exceptional-case guards. -/
theorem remS64_equiv (a b c : BitVec 64) :
    CTemplate.remS64.evalC a b c = Ptx.remS64 a b := by
  simpa only [CTemplate.remS64, Ptx.remS64] using (srem_equiv .w64 a b c)

/- PTX definition: `neg.s16` stores the 16-bit two's-complement negation of its operand. -/
/- Lean modeling: `evalC` equals the width-16 PTX value; printed `.w32` truncation remains `gap:subword-template-certificate`. -/
theorem negS16_equiv (a b c : BitVec 16) :
    CTemplate.negS16.evalC a b c = Ptx.negS16 a := rfl

/- PTX definition: `neg.s32` stores the 32-bit two's-complement negation of its operand. -/
/- Lean modeling: both named shallow models reduce to identical width-32 modular negation. -/
theorem negS32_equiv (a b c : BitVec 32) :
    CTemplate.negS32.evalC a b c = Ptx.negS32 a := rfl

/- PTX definition: `neg.s64` stores the 64-bit two's-complement negation of its operand. -/
/- Lean modeling: both named shallow models reduce to identical width-64 modular negation. -/
theorem negS64_equiv (a b c : BitVec 64) :
    CTemplate.negS64.evalC a b c = Ptx.negS64 a := rfl

/- PTX definition: `min.u16` selects the lesser unsigned 16-bit operand. -/
/- Lean modeling: the generic tie-safe proof gives value equality, while production printing uses a separately reviewed width-exact unsigned comparison. -/
theorem minU16_equiv (a b c : BitVec 16) :
    CTemplate.minU16.evalC a b c = Ptx.minU16 a b := by
  simpa only [CTemplate.minU16, Ptx.minU16] using (umin_equiv .w32 a b c)

/- PTX definition: `min.s16` selects the lesser signed 16-bit operand. -/
/- Lean modeling: the generic tie-safe proof gives value equality, while production printing uses a separately reviewed width-exact signed comparison. -/
theorem minS16_equiv (a b c : BitVec 16) :
    CTemplate.minS16.evalC a b c = Ptx.minS16 a b := by
  simpa only [CTemplate.minS16, Ptx.minS16] using (smin_equiv .w32 a b c)

/- PTX definition: `min.u32` selects the lesser unsigned 32-bit operand. -/
/- Lean modeling: specializing the generic tie-safe comparison proof establishes equality of the named models. -/
theorem minU32_equiv (a b c : BitVec 32) :
    CTemplate.minU32.evalC a b c = Ptx.minU32 a b := by
  simpa only [CTemplate.minU32, Ptx.minU32] using (umin_equiv .w32 a b c)

/- PTX definition: `min.s32` selects the lesser signed 32-bit operand. -/
/- Lean modeling: specializing the generic tie-safe signed comparison proof establishes equality of the named models. -/
theorem minS32_equiv (a b c : BitVec 32) :
    CTemplate.minS32.evalC a b c = Ptx.minS32 a b := by
  simpa only [CTemplate.minS32, Ptx.minS32] using (smin_equiv .w32 a b c)

/- PTX definition: `min.u64` selects the lesser unsigned 64-bit operand. -/
/- Lean modeling: specializing the generic tie-safe comparison proof establishes equality of the named models. -/
theorem minU64_equiv (a b c : BitVec 64) :
    CTemplate.minU64.evalC a b c = Ptx.minU64 a b := by
  simpa only [CTemplate.minU64, Ptx.minU64] using (umin_equiv .w64 a b c)

/- PTX definition: `min.s64` selects the lesser signed 64-bit operand. -/
/- Lean modeling: specializing the generic tie-safe signed comparison proof establishes equality of the named models. -/
theorem minS64_equiv (a b c : BitVec 64) :
    CTemplate.minS64.evalC a b c = Ptx.minS64 a b := by
  simpa only [CTemplate.minS64, Ptx.minS64] using (smin_equiv .w64 a b c)

/- PTX definition: `max.u16` selects the greater unsigned 16-bit operand. -/
/- Lean modeling: the generic tie-safe proof gives value equality, while production printing uses a separately reviewed width-exact unsigned comparison. -/
theorem maxU16_equiv (a b c : BitVec 16) :
    CTemplate.maxU16.evalC a b c = Ptx.maxU16 a b := by
  simpa only [CTemplate.maxU16, Ptx.maxU16] using (umax_equiv .w32 a b c)

/- PTX definition: `max.s16` selects the greater signed 16-bit operand. -/
/- Lean modeling: the generic tie-safe proof gives value equality, while production printing uses a separately reviewed width-exact signed comparison. -/
theorem maxS16_equiv (a b c : BitVec 16) :
    CTemplate.maxS16.evalC a b c = Ptx.maxS16 a b := by
  simpa only [CTemplate.maxS16, Ptx.maxS16] using (smax_equiv .w32 a b c)

/- PTX definition: `max.u32` selects the greater unsigned 32-bit operand. -/
/- Lean modeling: specializing the generic tie-safe comparison proof establishes equality of the named models. -/
theorem maxU32_equiv (a b c : BitVec 32) :
    CTemplate.maxU32.evalC a b c = Ptx.maxU32 a b := by
  simpa only [CTemplate.maxU32, Ptx.maxU32] using (umax_equiv .w32 a b c)

/- PTX definition: `max.s32` selects the greater signed 32-bit operand. -/
/- Lean modeling: specializing the generic tie-safe signed comparison proof establishes equality of the named models. -/
theorem maxS32_equiv (a b c : BitVec 32) :
    CTemplate.maxS32.evalC a b c = Ptx.maxS32 a b := by
  simpa only [CTemplate.maxS32, Ptx.maxS32] using (smax_equiv .w32 a b c)

/- PTX definition: `max.u64` selects the greater unsigned 64-bit operand. -/
/- Lean modeling: specializing the generic tie-safe comparison proof establishes equality of the named models. -/
theorem maxU64_equiv (a b c : BitVec 64) :
    CTemplate.maxU64.evalC a b c = Ptx.maxU64 a b := by
  simpa only [CTemplate.maxU64, Ptx.maxU64] using (umax_equiv .w64 a b c)

/- PTX definition: `max.s64` selects the greater signed 64-bit operand. -/
/- Lean modeling: specializing the generic tie-safe signed comparison proof establishes equality of the named models. -/
theorem maxS64_equiv (a b c : BitVec 64) :
    CTemplate.maxS64.evalC a b c = Ptx.maxS64 a b := by
  simpa only [CTemplate.maxS64, Ptx.maxS64] using (smax_equiv .w64 a b c)

end Gxptx.CTemplate

namespace Gxptx.CW

/- PTX definition: signed integer `abs` returns `-a` when `a < 0` and returns `a` otherwise. -/
/- Lean modeling: the C-side and PTX shallow kernels use the identical signed test and modular negation. -/
theorem absS_equiv {w : Nat} (a : BitVec w) :
    absS a = Ptx.absS a := rfl

/- PTX definition: `abs.s16` applies signed absolute value to a 16-bit operand. -/
/- Lean modeling: specializing identical kernels proves model equality, including GX's explicit minimum-pattern policy rather than an additional PDF guarantee. -/
theorem absS16_equiv (a : BitVec 16) :
    absS16 a = Ptx.absS16 a := rfl

/- PTX definition: `abs.s32` applies signed absolute value to a 32-bit operand. -/
/- Lean modeling: specializing identical kernels proves model equality, including GX's explicit minimum-pattern policy rather than an additional PDF guarantee. -/
theorem absS32_equiv (a : BitVec 32) :
    absS32 a = Ptx.absS32 a := rfl

/- PTX definition: `abs.s64` applies signed absolute value to a 64-bit operand. -/
/- Lean modeling: specializing identical kernels proves model equality, including GX's explicit minimum-pattern policy rather than an additional PDF guarantee. -/
theorem absS64_equiv (a : BitVec 64) :
    absS64 a = Ptx.absS64 a := rfl

/- PTX definition: `mul.wide.uN` is legal only for N = 16 or 32 and returns the complete product. -/
/- Lean modeling: equality holds for every `w`, but serves as a PTX certificate only through the legal U16/U32 specializations. -/
theorem mulWideU_equiv {w : Nat} (a b : BitVec w) :
    mulWideU a b = Ptx.mulWideU a b := by
  apply BitVec.eq_of_toNat_eq
  have hle : (2 : Nat) ^ w ≤ 2 ^ (2 * w) :=
    Nat.pow_le_pow_right (by decide) (by omega)
  simp only [mulWideU, Ptx.mulWideU, BitVec.toNat_mul, BitVec.toNat_setWidth,
    BitVec.toNat_ofNat]
  rw [Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le a.isLt hle),
      Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le b.isLt hle)]

/- PTX definition: `mul.wide.sN` is legal only for N = 16 or 32 and returns the complete product. -/
/- Lean modeling: equality holds for every `w`, but serves as a PTX certificate only through the legal S16/S32 specializations. -/
theorem mulWideS_equiv {w : Nat} (a b : BitVec w) :
    mulWideS a b = Ptx.mulWideS a b :=
  (BitVec.ofInt_mul _ _).symm

/- PTX definition: `mul.wide.u16` returns the complete 32-bit unsigned product. -/
/- Lean modeling: specializing the generic C/PTX wide-product equality proves the two explicitly named 16-bit unsigned definitions coincide. -/
theorem mulWideU16_equiv (a b : BitVec 16) :
    mulWideU16 a b = Ptx.mulWideU16 a b := by
  simpa only [mulWideU16, Ptx.mulWideU16] using (mulWideU_equiv a b)

/- PTX definition: `mul.wide.s16` returns the complete 32-bit signed product pattern. -/
/- Lean modeling: specializing the generic C/PTX wide-product equality proves the two explicitly named 16-bit signed definitions coincide. -/
theorem mulWideS16_equiv (a b : BitVec 16) :
    mulWideS16 a b = Ptx.mulWideS16 a b := by
  simpa only [mulWideS16, Ptx.mulWideS16] using (mulWideS_equiv a b)

/- PTX definition: `mul.wide.u32` returns the complete 64-bit unsigned product. -/
/- Lean modeling: specializing the generic C/PTX wide-product equality proves the two explicitly named 32-bit unsigned definitions coincide. -/
theorem mulWideU32_equiv (a b : BitVec 32) :
    mulWideU32 a b = Ptx.mulWideU32 a b := by
  simpa only [mulWideU32, Ptx.mulWideU32] using (mulWideU_equiv a b)

/- PTX definition: `mul.wide.s32` returns the complete 64-bit signed product pattern. -/
/- Lean modeling: specializing the generic C/PTX wide-product equality proves the two explicitly named 32-bit signed definitions coincide. -/
theorem mulWideS32_equiv (a b : BitVec 32) :
    mulWideS32 a b = Ptx.mulWideS32 a b := by
  simpa only [mulWideS32, Ptx.mulWideS32] using (mulWideS_equiv a b)

/- PTX definition: `mul.hi.uN` is legal only for N = 16, 32, or 64 and selects the high half. -/
/- Lean modeling: equality holds for every `w`, but certifies a PTX instruction only at an explicit legal-width specialization. -/
theorem mulHiU_equiv {w : Nat} (a b : BitVec w) :
    mulHiU a b = Ptx.mulHiU a b := by
  unfold mulHiU Ptx.mulHiU
  rw [mulWideU_equiv]

/- PTX definition: `mul.hi.sN` is legal only for N = 16, 32, or 64 and selects the high half. -/
/- Lean modeling: equality holds for every `w`, but certifies a PTX instruction only at an explicit legal-width specialization. -/
theorem mulHiS_equiv {w : Nat} (a b : BitVec w) :
    mulHiS a b = Ptx.mulHiS a b := by
  unfold mulHiS Ptx.mulHiS
  rw [mulWideS_equiv]

/- PTX definition: `mul.hi.u16` returns bits 31 through 16 of the unsigned product. -/
/- Lean modeling: specializing `mulHiU_equiv` proves the explicitly named 16-bit C and PTX high-half definitions coincide. -/
theorem mulHiU16_equiv (a b : BitVec 16) :
    mulHiU16 a b = Ptx.mulHiU16 a b := by
  simpa only [mulHiU16, Ptx.mulHiU16] using (mulHiU_equiv a b)

/- PTX definition: `mul.hi.s16` returns bits 31 through 16 of the signed product. -/
/- Lean modeling: specializing `mulHiS_equiv` proves the explicitly named 16-bit C and PTX signed high-half definitions coincide. -/
theorem mulHiS16_equiv (a b : BitVec 16) :
    mulHiS16 a b = Ptx.mulHiS16 a b := by
  simpa only [mulHiS16, Ptx.mulHiS16] using (mulHiS_equiv a b)

/- PTX definition: `mul.hi.u32` returns bits 63 through 32 of the complete unsigned product. -/
/- Lean modeling: specializing `mulHiU_equiv` to 32 bits proves the C high-half model equals the explicitly named PTX wrapper. -/
theorem mulHiU32_equiv (a b : BitVec 32) :
    mulHiU32 a b = Ptx.mulHiU32 a b := by
  simpa only [mulHiU32, Ptx.mulHiU32] using (mulHiU_equiv a b)

/- PTX definition: `mul.hi.s32` returns bits 63 through 32 of the complete signed product. -/
/- Lean modeling: specializing `mulHiS_equiv` to 32 bits proves the C signed high-half model equals the explicitly named PTX wrapper. -/
theorem mulHiS32_equiv (a b : BitVec 32) :
    mulHiS32 a b = Ptx.mulHiS32 a b := by
  simpa only [mulHiS32, Ptx.mulHiS32] using (mulHiS_equiv a b)

/- PTX definition: `mul.hi.u64` returns bits 127 through 64 of the unsigned product. -/
/- Lean modeling: specializing `mulHiU_equiv` proves the explicitly named 64-bit C and PTX high-half definitions coincide. -/
theorem mulHiU64_equiv (a b : BitVec 64) :
    mulHiU64 a b = Ptx.mulHiU64 a b := by
  simpa only [mulHiU64, Ptx.mulHiU64] using (mulHiU_equiv a b)

/- PTX definition: `mul.hi.s64` returns bits 127 through 64 of the signed product. -/
/- Lean modeling: specializing `mulHiS_equiv` proves the explicitly named 64-bit C and PTX signed high-half definitions coincide. -/
theorem mulHiS64_equiv (a b : BitVec 64) :
    mulHiS64 a b = Ptx.mulHiS64 a b := by
  simpa only [mulHiS64, Ptx.mulHiS64] using (mulHiS_equiv a b)

/- PTX definition: `mad.hi.uN` is legal only for N = 16, 32, or 64 and adds `c` to the high half. -/
/- Lean modeling: equality is generic algebraically, but certifies PTX only at legal-width specializations and not the printer route. -/
theorem madHiU_equiv {w : Nat} (a b c : BitVec w) :
    madHiU a b c = Ptx.madHiU a b c := by
  unfold madHiU Ptx.madHiU
  rw [mulHiU_equiv]

/- PTX definition: `mad.hi.sN` is legal only for N = 16, 32, or 64 and adds `c` to the high half. -/
/- Lean modeling: equality is generic algebraically, but certifies PTX only at legal-width specializations and not the printer route. -/
theorem madHiS_equiv {w : Nat} (a b c : BitVec w) :
    madHiS a b c = Ptx.madHiS a b c := by
  unfold madHiS Ptx.madHiS
  rw [mulHiS_equiv]

/- PTX definition: `mad.hi.u16` adds a 16-bit addend to unsigned product bits 31 through 16. -/
/- Lean modeling: specializing the generic equality proves the explicitly named 16-bit C and PTX definitions coincide. -/
theorem madHiU16_equiv (a b c : BitVec 16) :
    madHiU16 a b c = Ptx.madHiU16 a b c := by
  simpa only [madHiU16, Ptx.madHiU16] using (madHiU_equiv a b c)

/- PTX definition: `mad.hi.s16` adds a 16-bit addend to signed product bits 31 through 16. -/
/- Lean modeling: specializing the generic equality proves the explicitly named 16-bit C and PTX definitions coincide. -/
theorem madHiS16_equiv (a b c : BitVec 16) :
    madHiS16 a b c = Ptx.madHiS16 a b c := by
  simpa only [madHiS16, Ptx.madHiS16] using (madHiS_equiv a b c)

/- PTX definition: `mad.hi.u32` adds a 32-bit addend to unsigned product bits 63 through 32. -/
/- Lean modeling: specializing the generic equality proves the explicitly named 32-bit C and PTX definitions coincide. -/
theorem madHiU32_equiv (a b c : BitVec 32) :
    madHiU32 a b c = Ptx.madHiU32 a b c := by
  simpa only [madHiU32, Ptx.madHiU32] using (madHiU_equiv a b c)

/- PTX definition: non-saturating `mad.hi.s32` adds a 32-bit addend to signed product bits 63 through 32. -/
/- Lean modeling: specializing the generic equality proves the explicitly named non-saturating 32-bit C and PTX definitions coincide. -/
theorem madHiS32_equiv (a b c : BitVec 32) :
    madHiS32 a b c = Ptx.madHiS32 a b c := by
  simpa only [madHiS32, Ptx.madHiS32] using (madHiS_equiv a b c)

/- PTX definition: `mad.hi.u64` adds a 64-bit addend to unsigned product bits 127 through 64. -/
/- Lean modeling: specializing the generic equality proves the explicitly named 64-bit C and PTX definitions coincide. -/
theorem madHiU64_equiv (a b c : BitVec 64) :
    madHiU64 a b c = Ptx.madHiU64 a b c := by
  simpa only [madHiU64, Ptx.madHiU64] using (madHiU_equiv a b c)

/- PTX definition: `mad.hi.s64` adds a 64-bit addend to signed product bits 127 through 64. -/
/- Lean modeling: specializing the generic equality proves the explicitly named 64-bit C and PTX definitions coincide. -/
theorem madHiS64_equiv (a b c : BitVec 64) :
    madHiS64 a b c = Ptx.madHiS64 a b c := by
  simpa only [madHiS64, Ptx.madHiS64] using (madHiS_equiv a b c)

/- PTX definition: `mad.wide.uN` is legal only for N = 16 or 32 and adds a 2N-bit addend to the complete product. -/
/- Lean modeling: rewriting the already-proved full-product equality leaves identical fixed-width addition on both sides. -/
theorem madWideU_equiv {w : Nat} (a b : BitVec w) (c : BitVec (2 * w)) :
    madWideU a b c = Ptx.madWideU a b c := by
  unfold madWideU Ptx.madWideU
  rw [mulWideU_equiv]

/- PTX definition: `mad.wide.sN` is legal only for N = 16 or 32 and adds a 2N-bit addend to the complete signed product. -/
/- Lean modeling: rewriting the signed full-product equality leaves identical fixed-width addition on both sides. -/
theorem madWideS_equiv {w : Nat} (a b : BitVec w) (c : BitVec (2 * w)) :
    madWideS a b c = Ptx.madWideS a b c := by
  unfold madWideS Ptx.madWideS
  rw [mulWideS_equiv]

/- PTX definition: `mad.wide.u16` adds a 32-bit addend to the full unsigned 16-bit product. -/
/- Lean modeling: specializing the generic equality proves the explicit 16/32-bit C and PTX wrappers coincide. -/
theorem madWideU16_equiv (a b : BitVec 16) (c : BitVec 32) :
    madWideU16 a b c = Ptx.madWideU16 a b c := by
  simpa only [madWideU16, Ptx.madWideU16] using (madWideU_equiv a b c)

/- PTX definition: `mad.wide.s16` adds a 32-bit addend to the full signed 16-bit product. -/
/- Lean modeling: specializing the generic equality proves the explicit signed 16/32-bit C and PTX wrappers coincide. -/
theorem madWideS16_equiv (a b : BitVec 16) (c : BitVec 32) :
    madWideS16 a b c = Ptx.madWideS16 a b c := by
  simpa only [madWideS16, Ptx.madWideS16] using (madWideS_equiv a b c)

/- PTX definition: `mad.wide.u32` adds a 64-bit addend to the full unsigned 32-bit product. -/
/- Lean modeling: specializing the generic equality proves the explicit 32/64-bit C and PTX wrappers coincide. -/
theorem madWideU32_equiv (a b : BitVec 32) (c : BitVec 64) :
    madWideU32 a b c = Ptx.madWideU32 a b c := by
  simpa only [madWideU32, Ptx.madWideU32] using (madWideU_equiv a b c)

/- PTX definition: `mad.wide.s32` adds a 64-bit addend to the full signed 32-bit product. -/
/- Lean modeling: specializing the generic equality proves the explicit signed 32/64-bit C and PTX wrappers coincide. -/
theorem madWideS32_equiv (a b : BitVec 32) (c : BitVec 64) :
    madWideS32 a b c = Ptx.madWideS32 a b c := by
  simpa only [madWideS32, Ptx.madWideS32] using (madWideS_equiv a b c)

end Gxptx.CW

/-! ## Multiply characterizations -/

namespace Gxptx.Ptx

/- PTX definition: a `mul.wide.uN` instruction exists only for legal N = 16 or 32 and contains the complete product. -/
/- Lean modeling: the bound proves the generic kernel for every `w`; it becomes a PTX certificate only at the explicit legal-width specializations. -/
theorem mulWideU_exact {w : Nat} (a b : BitVec w) :
    (mulWideU a b).toNat = a.toNat * b.toNat := by
  have hb : a.toNat * b.toNat < 2 ^ (2 * w) := by
    have h2 : (2 : Nat) ^ (2 * w) = 2 ^ w * 2 ^ w := by
      rw [← Nat.pow_add]; congr 1; omega
    rw [h2]
    exact Nat.mul_lt_mul_of_lt_of_le a.isLt (Nat.le_of_lt b.isLt)
      (Nat.two_pow_pos w)
  simp only [mulWideU, BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt hb

/- PTX definition: `mul.lo.u32` is the low 32 bits of the unsigned product. -/
/- Lean modeling: this characterization unfolds the explicit unsigned wrapper to the generic 32-bit modular multiplication model. -/
theorem mulLoU32_spec (a b : BitVec 32) : mulLoU32 a b = a * b := rfl

/- PTX definition: `mul.lo.s32` is the low 32 bits of the signed product. -/
/- Lean modeling: this separate signed characterization records that signed and unsigned multiplication have the same low 32-bit pattern. -/
theorem mulLoS32_spec (a b : BitVec 32) : mulLoS32 a b = a * b := rfl

/- PTX definition: `mul.hi.u32` is bits 63 through 32 of the complete unsigned product. -/
/- Lean modeling: this characterization exposes the natural-number product, 64-bit encoding, logical shift, and final 32-bit projection used by the wrapper. -/
theorem mulHiU32_spec (a b : BitVec 32) :
    mulHiU32 a b =
      ((BitVec.ofNat 64 (a.toNat * b.toNat)) >>> 32).setWidth 32 := rfl

/- PTX definition: `mul.hi.s32` is bits 63 through 32 of the complete signed product. -/
/- Lean modeling: this characterization exposes the signed product, 64-bit two's-complement encoding, arithmetic shift, and final 32-bit projection. -/
theorem mulHiS32_spec (a b : BitVec 32) :
    mulHiS32 a b =
      ((BitVec.ofInt 64 (a.toInt * b.toInt)).sshiftRight 32).setWidth 32 := rfl

end Gxptx.Ptx

namespace Gxptx.CTemplate

/- PTX definition: `mul.lo.u16` returns bits 15 through 0 of the unsigned product. -/
/- Lean modeling: this proves only `evalC` instantiated at `BitVec 16`; it does not prove the `.w32` printed computation followed by 16-bit assignment (`gap:subword-template-certificate`). -/
theorem mulLoU16_equiv (a b c : BitVec 16) :
    CTemplate.mulLoU16.evalC a b c = Ptx.mulLoU16 a b := rfl

/- PTX definition: `mul.lo.s16` returns bits 15 through 0 of the signed product. -/
/- Lean modeling: this proves only `evalC` instantiated at `BitVec 16`; it does not prove the `.w32` printed computation followed by 16-bit assignment (`gap:subword-template-certificate`). -/
theorem mulLoS16_equiv (a b c : BitVec 16) :
    CTemplate.mulLoS16.evalC a b c = Ptx.mulLoS16 a b := rfl

/- PTX definition: `mul.lo.u32` returns bits 31 through 0 of the unsigned product. -/
/- Lean modeling: evaluating the named unsigned 32-bit C-side value model reduces to the same modular multiplication as `Ptx.mulLoU32`. -/
theorem mulLoU32_equiv (a b c : BitVec 32) :
    CTemplate.mulLoU32.evalC a b c = Ptx.mulLoU32 a b := rfl

/- PTX definition: `mul.lo.s32` returns bits 31 through 0 of the signed product. -/
/- Lean modeling: unsigned C multiplication avoids signed-overflow UB and has identical low 32 bits, yielding exactly `Ptx.mulLoS32`. -/
theorem mulLoS32_equiv (a b c : BitVec 32) :
    CTemplate.mulLoS32.evalC a b c = Ptx.mulLoS32 a b := rfl

/- PTX definition: `mul.lo.u64` returns bits 63 through 0 of the unsigned product. -/
/- Lean modeling: evaluating the named unsigned 64-bit C-side value model reduces to the same modular multiplication as `Ptx.mulLoU64`. -/
theorem mulLoU64_equiv (a b c : BitVec 64) :
    CTemplate.mulLoU64.evalC a b c = Ptx.mulLoU64 a b := rfl

/- PTX definition: `mul.lo.s64` returns bits 63 through 0 of the signed product. -/
/- Lean modeling: unsigned C multiplication avoids signed-overflow UB and has identical low 64 bits, yielding exactly `Ptx.mulLoS64`. -/
theorem mulLoS64_equiv (a b c : BitVec 64) :
    CTemplate.mulLoS64.evalC a b c = Ptx.mulLoS64 a b := rfl

/- PTX definition: `mad.lo.u16` retains the low 16 bits of product plus addend. -/
/- Lean modeling: both shallow models use identical width-16 modular multiplication and addition; printed-width truncation remains a separate boundary. -/
theorem madLoU16_equiv (a b c : BitVec 16) :
    CTemplate.madLoU16.evalC a b c = Ptx.madLoU16 a b c := rfl

/- PTX definition: `mad.lo.s16` retains the low 16 two's-complement product-plus-addend bits. -/
/- Lean modeling: signed and unsigned low result bits coincide, so both shallow models reduce to the same width-16 modular operations. -/
theorem madLoS16_equiv (a b c : BitVec 16) :
    CTemplate.madLoS16.evalC a b c = Ptx.madLoS16 a b c := rfl

/- PTX definition: `mad.lo.u32` retains the low 32 bits of product plus addend. -/
/- Lean modeling: both definitions reduce to the same width-32 `BitVec` multiplication and addition. -/
theorem madLoU32_equiv (a b c : BitVec 32) :
    CTemplate.madLoU32.evalC a b c = Ptx.madLoU32 a b c := rfl

/- PTX definition: `mad.lo.s32` retains the low 32 two's-complement product-plus-addend bits. -/
/- Lean modeling: both shallow definitions use the common modulo-`2^32` result pattern. -/
theorem madLoS32_equiv (a b c : BitVec 32) :
    CTemplate.madLoS32.evalC a b c = Ptx.madLoS32 a b c := rfl

/- PTX definition: `mad.lo.u64` retains the low 64 bits of product plus addend. -/
/- Lean modeling: both definitions reduce to the same width-64 `BitVec` multiplication and addition. -/
theorem madLoU64_equiv (a b c : BitVec 64) :
    CTemplate.madLoU64.evalC a b c = Ptx.madLoU64 a b c := rfl

/- PTX definition: `mad.lo.s64` retains the low 64 two's-complement product-plus-addend bits. -/
/- Lean modeling: both shallow definitions use the common modulo-`2^64` result pattern. -/
theorem madLoS64_equiv (a b c : BitVec 64) :
    CTemplate.madLoS64.evalC a b c = Ptx.madLoS64 a b c := rfl

end Gxptx.CTemplate
