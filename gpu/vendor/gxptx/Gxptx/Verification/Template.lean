/-
  Equivalence proofs for the finite C-template value semantics.

  This file preserves the shift infrastructure and every `CTemplate`
  equivalence theorem formerly located at the front of `Gxptx.CSem`.
  Statements and proof bodies remain unchanged; only module ownership and
  specification/rationale comments are made explicit.
-/
import Gxptx.Semantics.C.Template
import Gxptx.Semantics.PTX.Arithmetic
import Gxptx.Semantics.PTX.Logic
import Gxptx.Semantics.PTX.Comparison

namespace Gxptx

/-! ## Width-behavior lemmas: BitVec's shifts already have PTX's clamp. -/

/- PTX definition: `shl` clamps a shift amount beyond width `N` to `N`, whose fixed-width result is zero. -/
/- Lean modeling: divisibility by `2^w` shows Lean's `BitVec` left shift is zero for every amount at least `w`. -/
  theorem shiftLeft_ge_width {w : Nat} (a : BitVec w) {n : Nat} (h : w ≤ n) :
    a <<< n = 0#w := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_shiftLeft, BitVec.toNat_ofNat, Nat.zero_mod]
  rw [Nat.shiftLeft_eq]
  obtain ⟨k, hk⟩ := Nat.pow_dvd_pow 2 h
  rw [hk, ← Nat.mul_assoc, Nat.mul_comm a.toNat (2^w), Nat.mul_assoc,
      Nat.mul_mod_right]

/- PTX definition: unsigned and bit-type `shr` clamp an excessive amount to width `N` and zero-fill the result. -/
/- Lean modeling: the represented natural value is below `2^w`, so dividing it by `2^n` for `n ≥ w` yields zero. -/
theorem ushiftRight_ge_width {w : Nat} (a : BitVec w) {n : Nat} (h : w ≤ n) :
    a >>> n = 0#w := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ushiftRight, BitVec.toNat_ofNat, Nat.zero_mod]
  rw [Nat.shiftRight_eq_div_pow]
  exact Nat.div_eq_of_lt
    (Nat.lt_of_lt_of_le a.isLt (Nat.pow_le_pow_right (by decide) h))

/- PTX definition: PTX fundamental integer and bit types have positive fixed widths; width zero arises only from this generic Lean model. -/
/- Lean modeling: `BitVec 0` has a single representable bit pattern, so its two arbitrary inhabitants have equal natural values. -/
private theorem bv0_eq (x y : BitVec 0) : x = y := by
  have h1 : (2 : Nat) ^ (0 : Nat) = 1 := rfl
  have hx := x.isLt
  have hy := y.isLt
  exact BitVec.eq_of_toNat_eq (by omega)

/- PTX definition: signed PTX integer types use fixed-width two's-complement ranges. -/
/- Lean modeling: unfolding `BitVec.toInt` and its sign test establishes the standard interval `[-2^(w-1), 2^(w-1))`. -/
private theorem toInt_bounds {w : Nat} (hw : 0 < w) (a : BitVec w) :
    -(((2 ^ (w - 1) : Nat)) : Int) ≤ a.toInt
      ∧ a.toInt < ((2 ^ (w - 1) : Nat) : Int) := by
  have hpow : (2 : Nat) ^ w = 2 ^ (w - 1) * 2 := by
    rw [← Nat.pow_succ]
    congr 1
    omega
  have hlt := a.isLt
  rw [BitVec.toInt_eq_toNat_cond]
  by_cases hc : 2 * a.toNat < 2 ^ w
  · rw [if_pos hc]; constructor <;> omega
  · rw [if_neg hc]; constructor <;> omega

/- PTX definition: signed `shr` fills with the sign bit after its amount is clamped to width `N`. -/
/- Lean modeling: once an arithmetic shift reaches `w-1`, the two's-complement range bounds force every remaining bit to equal the sign bit. -/
theorem sshiftRight_ge_width {w : Nat} (a : BitVec w) {n : Nat} (h : w - 1 ≤ n) :
    a.sshiftRight n = a.sshiftRight (w - 1) := by
  rcases Nat.eq_zero_or_pos w with hw | hw
  · subst hw; exact bv0_eq _ _
  · rw [BitVec.sshiftRight_eq, BitVec.sshiftRight_eq]
    obtain ⟨hlo, hhi⟩ := toInt_bounds hw a
    congr 1
    cases hx : a.toInt with
    | ofNat m =>
      rw [hx] at hhi
      simp only [Int.ofNat_eq_coe] at hhi ⊢
      have hm : m < 2 ^ (w - 1) := by omega
      have z1 : m >>> n = 0 := by
        rw [Nat.shiftRight_eq_div_pow]
        exact Nat.div_eq_of_lt
          (Nat.lt_of_lt_of_le hm (Nat.pow_le_pow_right (by decide) h))
      have z2 : m >>> (w - 1) = 0 := by
        rw [Nat.shiftRight_eq_div_pow]
        exact Nat.div_eq_of_lt hm
      rw [Int.natCast_shiftRight, Int.natCast_shiftRight, z1, z2]
    | negSucc m =>
      rw [hx] at hlo
      rw [Int.negSucc_eq] at hlo
      have hm : m < 2 ^ (w - 1) := by omega
      have z1 : m >>> n = 0 := by
        rw [Nat.shiftRight_eq_div_pow]
        exact Nat.div_eq_of_lt
          (Nat.lt_of_lt_of_le hm (Nat.pow_le_pow_right (by decide) h))
      have z2 : m >>> (w - 1) = 0 := by
        rw [Nat.shiftRight_eq_div_pow]
        exact Nat.div_eq_of_lt hm
      rw [Int.negSucc_shiftRight, Int.negSucc_shiftRight, z1, z2]

/-! ## The equivalence theorems: C-side value semantics = PTX semantics,
      per op, for every width (hence for 32 AND 64 bit). -/

namespace CTemplate

/- PTX definition: unsaturated scalar `add` computes `d = a + b`. -/
/- Lean modeling: both the indexed C shape and shallow PTX function use identical fixed-width `BitVec` addition. -/
theorem add_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.add t).evalC a b c = Ptx.add a b := rfl

/- PTX definition: `add.f16x2` applies default-RN binary16 addition independently to both packed lanes. -/
/- Lean modeling: the dedicated C template and PTX instruction reduce to the same exact packed bit model. -/
theorem addF16x2_template_equiv (a b c : BitVec 32) :
    CTemplate.addF16x2.evalC a b c = Ptx.addF16x2 a b := rfl

/- PTX definition: `mul.bf16x2` applies default-RN bfloat16 multiplication independently to both packed lanes. -/
/- Lean modeling: the dedicated C template and PTX instruction reduce to the same exact packed bit model. -/
theorem mulBF16x2_template_equiv (a b c : BitVec 32) :
    CTemplate.mulBF16x2.evalC a b c = Ptx.mulBF16x2 a b := rfl

/- PTX definition: unsaturated scalar `sub` computes `d = a - b`. -/
/- Lean modeling: both sides reduce to identical modulo-`2^w` `BitVec` subtraction. -/
theorem sub_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.sub t).evalC a b c = Ptx.sub a b := rfl

/- PTX definition: `mul.lo` writes the low `w` bits of the full product. -/
/- Lean modeling: both sides reduce to the same width-`w` `BitVec` multiplication and therefore retain the same low half. -/
theorem mul_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.mul t).evalC a b c = Ptx.mulLo a b := rfl

/- PTX definition: `mad.lo` adds `c` to the low `w` bits of `a * b`. -/
/- Lean modeling: both sides definitionally perform the same wrapping `BitVec` multiply followed by wrapping addition. -/
theorem mad_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.mad t).evalC a b c = Ptx.madLo a b c := rfl

/- PTX definition: signed scalar `neg` computes `d = -a`. -/
/- Lean modeling: both sides reduce to the same two's-complement `BitVec` negation modulo `2^w`. -/
theorem neg_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.neg t).evalC a b c = Ptx.neg a := rfl

/- PTX definition: non-predicate `not` computes the fixed-width one's complement `d = ~a`. -/
/- Lean modeling: both the C-side template value evaluator and PTX semantics use the identical width-`w` `BitVec.not`. -/
theorem bnot_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.bnot t).evalC a b c = Ptx.bnot a := rfl

/- PTX definition: non-predicate `and` computes `d = a & b`. -/
/- Lean modeling: both sides reduce to the same pointwise fixed-width `BitVec.and`. -/
theorem band_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.band t).evalC a b c = Ptx.band a b := rfl

/- PTX definition: non-predicate `or` computes `d = a | b`. -/
/- Lean modeling: both sides reduce to the same pointwise fixed-width `BitVec.or`. -/
theorem bor_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.bor t).evalC a b c = Ptx.bor a b := rfl

/- PTX definition: non-predicate `xor` computes `d = a ^ b`. -/
/- Lean modeling: both sides reduce to the same pointwise fixed-width `BitVec.xor`. -/
theorem bxor_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.bxor t).evalC a b c = Ptx.bxor a b := rfl

/- PTX definition: `shl` clamps an excessive amount to width `N`, yielding the same zero-filled left shift. -/
/- Lean modeling: the explicit C-side value-model guard and plain PTX `BitVec` shift coincide by `shiftLeft_ge_width`. -/
theorem shl_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.shl t).evalC a b c = Ptx.shl a b.toNat := by
  simp only [evalC, Ptx.shl]
  by_cases h : w ≤ b.toNat
  · rw [if_pos h, shiftLeft_ge_width a h]
  · rw [if_neg h]

/- PTX definition: unsigned and bit-type `shr` clamp an excessive amount and zero-fill from the left. -/
/- Lean modeling: the explicit C-side value-model guard and plain logical `BitVec` shift coincide by `ushiftRight_ge_width`. -/
theorem lshr_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.lshr t).evalC a b c = Ptx.shrU a b.toNat := by
  simp only [evalC, Ptx.shrU]
  by_cases h : w ≤ b.toNat
  · rw [if_pos h, ushiftRight_ge_width a h]
  · rw [if_neg h]

/- PTX definition: signed `shr` clamps an excessive amount and fills vacated high bits with the sign bit. -/
/- Lean modeling: shifting by `w-1` in the C-side value model equals every larger arithmetic `BitVec` shift by `sshiftRight_ge_width`. -/
theorem ashr_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.ashr t).evalC a b c = Ptx.shrS a b.toNat := by
  simp only [evalC, Ptx.shrS]
  by_cases h : w ≤ b.toNat
  · rw [if_pos h]
    exact (sshiftRight_ge_width a (by omega)).symm
  · rw [if_neg h]

/- PTX definition: division by zero yields an unspecified, machine-specific value. -/
/- Lean modeling: this lemma only identifies the helper's chosen `~0` pattern with `BitVec.allOnes`; it does not claim that choice is uniquely ISA-mandated. -/
private theorem not_zero' {w : Nat} : ~~~(0#w) = BitVec.allOnes w := by
  rw [BitVec.not_def, BitVec.xor_zero]

/- PTX definition: unsigned `div` computes `a / b`, but division by zero yields an unspecified, machine-specific value. -/
/- Lean modeling: the theorem proves the C-side value model matches GX's deliberately pinned all-ones zero-divisor policy and ordinary unsigned quotients otherwise. -/
theorem udiv_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.udiv t).evalC a b c = Ptx.divU a b := by
  simp only [evalC, Ptx.divU, not_zero']

/- PTX definition: unsigned `rem` computes `a % b`, while a zero divisor yields an unspecified, machine-specific value. -/
/- Lean modeling: both models deliberately choose `a` for the unspecified zero-divisor case and use identical unsigned `BitVec` remainder otherwise. -/
theorem urem_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.urem t).evalC a b c = Ptx.remU a b := rfl

/- PTX definition: signed `div` uses integer division, with division by zero left unspecified and machine-specific. -/
/- Lean modeling: wrapping the negated signed interpretation back into `BitVec w` equals native width-`w` two's-complement negation for the helper's `-1` guard. -/
private theorem ofInt_neg_toInt {w : Nat} (a : BitVec w) :
    BitVec.ofInt w (-a.toInt) = -a := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_ofInt, BitVec.toInt_neg]

/- PTX definition: a zero remainder is represented by the all-zero destination bit pattern. -/
/- Lean modeling: converting mathematical integer zero to `BitVec w` is exactly the width-`w` zero used by the helper's `-1` guard. -/
private theorem ofInt_zero {w : Nat} : BitVec.ofInt w 0 = 0#w := by
  apply BitVec.eq_of_toNat_eq
  simp

/- PTX definition: signed `div` computes an integer quotient, while division by zero yields an unspecified, machine-specific value. -/
/- Lean modeling: the proof shows internal agreement for GX's all-ones zero-divisor, wrapped division-by-minus-one, and truncation-toward-zero policies; it does not make the zero case ISA-unique. -/
theorem sdiv_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.sdiv t).evalC a b c = Ptx.divS a b := by
  simp only [evalC, Ptx.divS, not_zero']
  by_cases hb0 : b = 0
  · simp [hb0]
  · rw [if_neg hb0, if_neg hb0]
    by_cases hm1 : b.toInt = -1
    · rw [if_pos hm1, hm1]
      have : a.toInt.tdiv (-1) = -a.toInt := by
        rw [show (-1 : Int) = -(1 : Int) from rfl, Int.tdiv_neg, Int.tdiv_one]
      rw [this, ofInt_neg_toInt, BitVec.zero_sub]
    · rw [if_neg hm1]

/- PTX definition: signed `rem` has machine-dependent negative-number behavior, and a zero divisor yields an unspecified, machine-specific value. -/
/- Lean modeling: the proof establishes agreement for GX's chosen C-style `tmod`, `a` on `/0`, and zero on divisor `-1`, without claiming those unspecified cases are unique. -/
theorem srem_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.srem t).evalC a b c = Ptx.remS a b := by
  simp only [evalC, Ptx.remS]
  by_cases hb0 : b = 0
  · simp [hb0]
  · rw [if_neg hb0, if_neg hb0]
    by_cases hm1 : b.toInt = -1
    · rw [if_pos hm1, hm1]
      have : a.toInt.tmod (-1) = 0 := by
        rw [show (-1 : Int) = -(1 : Int) from rfl, Int.tmod_neg, Int.tmod_one]
      rw [this, ofInt_zero]
    · rw [if_neg hm1]

/- PTX definition: unsigned `min` stores the lesser of `a` and `b`. -/
/- Lean modeling: strict `<` and non-strict `≤` selectors can differ only when operands are equal, where either branch returns the same bits. -/
theorem umin_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.umin t).evalC a b c = Ptx.minU a b := by
  simp only [evalC, Ptx.minU]
  by_cases hlt : a < b
  · rw [if_pos hlt, if_pos (BitVec.le_def.mpr (Nat.le_of_lt (BitVec.lt_def.mp hlt)))]
  · by_cases heq : a = b
    · simp [heq]
    · have hnlt : ¬ a.toNat < b.toNat := fun hn => hlt (BitVec.lt_def.mpr hn)
      have hle : ¬ a ≤ b := fun hc =>
        heq (BitVec.eq_of_toNat_eq (Nat.le_antisymm (BitVec.le_def.mp hc)
          (by omega)))
      rw [if_neg hlt, if_neg hle]

/- PTX definition: unsigned `max` stores the greater of `a` and `b`. -/
/- Lean modeling: strict `>` and non-strict `≥` selectors can differ only at equality, where both selected operands have identical bits. -/
theorem umax_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.umax t).evalC a b c = Ptx.maxU a b := by
  simp only [evalC, Ptx.maxU]
  by_cases hlt : b < a
  · rw [if_pos hlt, if_pos (BitVec.le_def.mpr (Nat.le_of_lt (BitVec.lt_def.mp hlt)))]
  · by_cases heq : a = b
    · simp [heq]
    · have hnlt : ¬ b.toNat < a.toNat := fun hn => hlt (BitVec.lt_def.mpr hn)
      have hle : ¬ b ≤ a := fun hc =>
        heq (BitVec.eq_of_toNat_eq (Nat.le_antisymm
          (by omega)
          (BitVec.le_def.mp hc)))
      rw [if_neg hlt, if_neg hle]

/- PTX definition: signed `min` stores the lesser signed integer operand. -/
/- Lean modeling: the C-side value model's strict signed comparison and PTX's non-strict signed selector agree because equality makes both branches identical. -/
theorem smin_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.smin t).evalC a b c = Ptx.minS a b := by
  simp only [evalC, Ptx.minS]
  by_cases hlt : a.toInt < b.toInt
  · rw [if_pos hlt, if_pos (Int.le_of_lt hlt)]
  · by_cases heq : a.toInt = b.toInt
    · have : a = b := BitVec.toInt_inj.mp heq
      simp [this]
    · have hle : ¬ a.toInt ≤ b.toInt := fun hc =>
        heq (Int.le_antisymm hc (by omega))
      rw [if_neg hlt, if_neg hle]

/- PTX definition: signed `max` stores the greater signed integer operand. -/
/- Lean modeling: the C-side value model's strict signed comparison and PTX's non-strict signed selector agree because equal operands have identical bit patterns. -/
theorem smax_equiv (t : PW) {w : Nat} (a b c : BitVec w) :
    (CTemplate.smax t).evalC a b c = Ptx.maxS a b := by
  simp only [evalC, Ptx.maxS]
  by_cases hlt : b.toInt < a.toInt
  · rw [if_pos hlt, if_pos (Int.le_of_lt hlt)]
  · by_cases heq : a.toInt = b.toInt
    · have : a = b := BitVec.toInt_inj.mp heq
      simp [this]
    · have hle : ¬ b.toInt ≤ a.toInt := fun hc =>
        heq (Int.le_antisymm (by omega) hc)
      rw [if_neg hlt, if_neg hle]

/- PTX definition: `selp` stores `a` when predicate `p` is true and `b` otherwise. -/
/- Lean modeling: the C-side value-model conditional and shallow PTX model use the same nonzero predicate test and select the same untouched operand. -/
theorem sel_equiv {w : Nat} (p a b : BitVec w) :
    CTemplate.sel.evalC p a b = Ptx.selp p a b := rfl

/- PTX definition: integer `setp` applies its comparison with type-directed signed or unsigned interpretation. -/
/- Lean modeling: case analysis on relation and signedness shows the C-side value model's 0/1 result uses exactly the same bit equality or numeric order. -/
theorem cmp_equiv (sgn : Bool) (c : CmpOp) (t : PW) {w : Nat} (a b x : BitVec w) :
    (CTemplate.cmp sgn c t).evalC a b x
      = (if Ptx.setp sgn c a b then 1 else 0) := by
  cases c <;> cases sgn <;> simp [evalC, Ptx.setp]

/- PTX definition: basic `.b16`, `.u16`, and `.s16` `setp` forms compare exactly 16 operand bits, using signed two's-complement order only for `.s16`; `.b16` uses unsigned equality. -/
/- Lean modeling: specializing the width-generic comparison equality to `BitVec 16` proves the value model used by `printCmp16`; the printer-to-ISO-C link remains trusted. -/
theorem cmp16_equiv (sgn : Bool) (c : CmpOp) (a b x : BitVec 16) :
    (CTemplate.cmp sgn c .w32).evalC a b x
      = (if Ptx.setp sgn c a b then 1 else 0) :=
  cmp_equiv sgn c .w32 a b x

/-! Predicate templates: truth-value equivalences (GX represents PTX Boolean
    truth with zero/nonzero host words; its generated predicate destinations
    use 0/1, but the templates are correct for ANY residual host value, like
    Core's `emit*_pred` theorems).
    `0 < w` because a 0-bit register cannot represent "true" (pred registers
    use width 32 when this value model is instantiated for the C-side representation). -/

/- PTX definition: predicate registers carry Boolean True/False values, and PTX exposes no direct predicate-to-integer conversion. -/
/- Lean modeling: GX chooses 0/1 for generated C predicate destinations; at every positive host-word width, the chosen one representation differs from zero. -/
private theorem one_ne_zero' {w : Nat} (hw : 0 < w) : (1#w) ≠ 0#w := by
  have h2 : 2 ≤ 2^w := by
    have := Nat.pow_le_pow_right (by decide : 0 < 2) hw
    simpa using this
  intro h
  have h3 := congrArg BitVec.toNat h
  rw [BitVec.toNat_ofNat, BitVec.toNat_ofNat, Nat.zero_mod,
      Nat.mod_eq_of_lt (by omega)] at h3
  omega

/- PTX definition: `not.pred` is true exactly when its predicate operand is false. -/
/- Lean modeling: splitting on zero shows the C-side 0/1 value model's nonzero truth value is precisely the negation of the input's nonzero test. -/
theorem lnot_equiv {w : Nat} (hw : 0 < w) (p x y : BitVec w) :
    (CTemplate.lnot.evalC p x y ≠ 0) ↔ ¬ (p ≠ 0) := by
  show ((if p = 0 then (1#w) else 0#w) ≠ 0) ↔ _
  by_cases h : p = 0
  · rw [if_pos h]; simp [h, one_ne_zero' hw]
  · rw [if_neg h]; simpa using h

/- PTX definition: `and.pred` is true exactly when both operands are true. -/
/- Lean modeling: the template returns representable one exactly when both input bit patterns are nonzero, and zero otherwise. -/
theorem land_equiv {w : Nat} (hw : 0 < w) (p q x : BitVec w) :
    (CTemplate.land.evalC p q x ≠ 0) ↔ (p ≠ 0 ∧ q ≠ 0) := by
  show ((if p ≠ 0 ∧ q ≠ 0 then (1#w) else 0#w) ≠ 0) ↔ _
  by_cases h : p ≠ 0 ∧ q ≠ 0
  · rw [if_pos h]; simpa [one_ne_zero' hw] using h
  · rw [if_neg h]; simpa using h

/- PTX definition: `or.pred` is true exactly when either operand is true. -/
/- Lean modeling: the template returns representable one exactly when at least one input bit pattern is nonzero, and zero otherwise. -/
theorem lor_equiv {w : Nat} (hw : 0 < w) (p q x : BitVec w) :
    (CTemplate.lor.evalC p q x ≠ 0) ↔ (p ≠ 0 ∨ q ≠ 0) := by
  show ((if p ≠ 0 ∨ q ≠ 0 then (1#w) else 0#w) ≠ 0) ↔ _
  by_cases h : p ≠ 0 ∨ q ≠ 0
  · rw [if_pos h]; simpa [one_ne_zero' hw] using h
  · rw [if_neg h]; simpa using h

/- PTX definition: `xor.pred` is true exactly when the operand truth values differ. -/
/- Lean modeling: the template returns zero for equivalent nonzero tests and representable one otherwise, exactly characterizing predicate exclusive-or. -/
theorem lxor_equiv {w : Nat} (hw : 0 < w) (p q x : BitVec w) :
    (CTemplate.lxor.evalC p q x ≠ 0) ↔ ¬ ((p ≠ 0) ↔ (q ≠ 0)) := by
  show ((if (p ≠ 0) ↔ (q ≠ 0) then (0#w) else 1#w) ≠ 0) ↔ _
  by_cases h : (p ≠ 0) ↔ (q ≠ 0)
  · rw [if_pos h]; simpa using h
  · rw [if_neg h]; simpa [one_ne_zero' hw] using h

/- PTX definition: the modeled scalar template domain is the finite product of the listed operations, widths, signedness choices, and comparisons. -/
/- Lean modeling: exhaustive constructor and finite-index case splits prove that every possible `CTemplate` value occurs in the explicit review inventory. -/
theorem all_complete (t : CTemplate) : t ∈ CTemplate.all := by
  cases t with
  | add w => cases w <;> simp [all]
  | addF16x2 => simp [all]
  | mulBF16x2 => simp [all]
  | sub w => cases w <;> simp [all]
  | mul w => cases w <;> simp [all]
  | mad w => cases w <;> simp [all]
  | neg w => cases w <;> simp [all]
  | bnot w => cases w <;> simp [all]
  | band w => cases w <;> simp [all]
  | bor w => cases w <;> simp [all]
  | bxor w => cases w <;> simp [all]
  | lnot => simp [all]
  | land => simp [all]
  | lor => simp [all]
  | lxor => simp [all]
  | shl w => cases w <;> simp [all]
  | lshr w => cases w <;> simp [all]
  | ashr w => cases w <;> simp [all]
  | udiv w => cases w <;> simp [all]
  | sdiv w => cases w <;> simp [all]
  | urem w => cases w <;> simp [all]
  | srem w => cases w <;> simp [all]
  | umin w => cases w <;> simp [all]
  | umax w => cases w <;> simp [all]
  | smin w => cases w <;> simp [all]
  | smax w => cases w <;> simp [all]
  | sel => simp [all]
  | cmp signed op w => cases signed <;> cases op <;> cases w <;> simp [all]

/- PTX definition: the current template domain contains every modeled scalar operation, width, signed comparison choice, and relation. -/
/- Lean modeling: reducing the explicit inventory proves that it contains exactly 71 concrete `CTemplate` values for review and checker enumeration. -/
theorem all_count : CTemplate.all.length = 71 := by decide

end CTemplate

end Gxptx
