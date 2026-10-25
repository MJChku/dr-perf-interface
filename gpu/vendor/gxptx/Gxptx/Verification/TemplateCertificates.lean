/-
  Total certificates for every finite scalar `CTemplate` constructor.

  This layer binds the print-width index to the value width, states the PTX
  obligation for each constructor, and cites the family equivalence theorem.
  Exhaustive matches make an added-but-uncertified template a build error.
-/
import Gxptx.Verification.Template

namespace Gxptx

/- PTX definition: each typed scalar template operates at the width selected by its PTX type suffix. -/
/- Lean modeling: `spec` binds every width-carrying constructor's operands to `BitVec t.bits`, closing the former disconnect between print width `t` and evaluator width; total matching rejects an uncertified constructor. -/
def CTemplate.spec : CTemplate → Prop
  /- PTX definition: unsaturated scalar `add` stores the low-width sum. -/
  /- Lean modeling: operands use `BitVec t.bits`, so `evalC` and `Ptx.add` are compared at the template's declared width. -/
  | .add t  => ∀ (a b c : BitVec t.bits), (CTemplate.add t).evalC a b c = Ptx.add a b
  /- PTX definition: `add.f16x2` performs two independent RN-even binary16 additions in one 32-bit value. -/
  /- Lean modeling: the constructor's fixed-width obligation compares the exact packed template and PTX models. -/
  | .addF16x2 => ∀ (a b c : BitVec 32), CTemplate.addF16x2.evalC a b c = Ptx.addF16x2 a b
  /- PTX definition: unsaturated scalar `sub` stores the low-width difference. -/
  /- Lean modeling: binding operands to `t.bits` compares the same fixed-width subtraction on both sides. -/
  | .sub t  => ∀ (a b c : BitVec t.bits), (CTemplate.sub t).evalC a b c = Ptx.sub a b
  /- PTX definition: `mul.lo` stores the low half of the full product. -/
  /- Lean modeling: `BitVec t.bits` multiplication retains exactly the low `t.bits` product bits on both sides. -/
  | .mul t  => ∀ (a b c : BitVec t.bits), (CTemplate.mul t).evalC a b c = Ptx.mulLo a b
  /- PTX definition: `mul.bf16x2` performs two independent RN-even bfloat16 products in one 32-bit value. -/
  /- Lean modeling: the fixed-width obligation compares the exact packed template and PTX multiplication models. -/
  | .mulBF16x2 => ∀ (a b c : BitVec 32), CTemplate.mulBF16x2.evalC a b c = Ptx.mulBF16x2 a b
  /- PTX definition: `mad.lo` adds `c` to the low product at destination width. -/
  /- Lean modeling: both value models perform multiplication and addition in the same `BitVec t.bits` ring. -/
  | .mad t  => ∀ (a b c : BitVec t.bits), (CTemplate.mad t).evalC a b c = Ptx.madLo a b c
  /- PTX definition: scalar `neg` stores the fixed-width two's-complement negation. -/
  /- Lean modeling: the obligation compares identical wrapping negation at the width selected by `t`. -/
  | .neg t  => ∀ (a b c : BitVec t.bits), (CTemplate.neg t).evalC a b c = Ptx.neg a
  /- PTX definition: non-predicate `not` complements every destination-width bit. -/
  /- Lean modeling: both sides operate pointwise on the same `BitVec t.bits` operand. -/
  | .bnot t => ∀ (a b c : BitVec t.bits), (CTemplate.bnot t).evalC a b c = Ptx.bnot a
  /- PTX definition: non-predicate `and` combines corresponding operand bits. -/
  /- Lean modeling: the obligation compares the same pointwise `BitVec.and` at `t.bits`. -/
  | .band t => ∀ (a b c : BitVec t.bits), (CTemplate.band t).evalC a b c = Ptx.band a b
  /- PTX definition: non-predicate `or` combines corresponding operand bits. -/
  /- Lean modeling: the obligation compares the same pointwise `BitVec.or` at `t.bits`. -/
  | .bor t  => ∀ (a b c : BitVec t.bits), (CTemplate.bor t).evalC a b c = Ptx.bor a b
  /- PTX definition: non-predicate `xor` differs exactly where operand bits differ. -/
  /- Lean modeling: the obligation compares the same pointwise `BitVec.xor` at `t.bits`. -/
  | .bxor t => ∀ (a b c : BitVec t.bits), (CTemplate.bxor t).evalC a b c = Ptx.bxor a b
  /- PTX definition: `not.pred` complements predicate truth. -/
  /- Lean modeling: the positive-width premise makes modeled one distinguishable from zero under the nonzero-is-true convention. -/
  | .lnot   => ∀ {w : Nat}, 0 < w → ∀ (p x y : BitVec w),
      (CTemplate.lnot.evalC p x y ≠ 0) ↔ ¬ (p ≠ 0)
  /- PTX definition: `and.pred` conjoins predicate truth values. -/
  /- Lean modeling: nonzero words model true, and positive width guarantees the returned one is observably nonzero. -/
  | .land   => ∀ {w : Nat}, 0 < w → ∀ (p q x : BitVec w),
      (CTemplate.land.evalC p q x ≠ 0) ↔ (p ≠ 0 ∧ q ≠ 0)
  /- PTX definition: `or.pred` disjoins predicate truth values. -/
  /- Lean modeling: the obligation states the C-side 0/1 value convention directly as a nonzero equivalence. -/
  | .lor    => ∀ {w : Nat}, 0 < w → ∀ (p q x : BitVec w),
      (CTemplate.lor.evalC p q x ≠ 0) ↔ (p ≠ 0 ∨ q ≠ 0)
  /- PTX definition: `xor.pred` computes predicate exclusive-or. -/
  /- Lean modeling: the proposition negates equivalence of the two nonzero tests, exactly characterizing exclusive-or. -/
  | .lxor   => ∀ {w : Nat}, 0 < w → ∀ (p q x : BitVec w),
      (CTemplate.lxor.evalC p q x ≠ 0) ↔ ¬ ((p ≠ 0) ↔ (q ≠ 0))
  /- PTX definition: `shl` clamps excessive amounts and zero-fills vacated bits. -/
  /- Lean modeling: `b.toNat` supplies the common amount while `BitVec t.bits` fixes the common clamp width. -/
  | .shl t  => ∀ (a b c : BitVec t.bits), (CTemplate.shl t).evalC a b c = Ptx.shl a b.toNat
  /- PTX definition: unsigned `shr` clamps excessive amounts and zero-fills high bits. -/
  /- Lean modeling: both sides use the same natural shift amount and destination width selected by `t`. -/
  | .lshr t => ∀ (a b c : BitVec t.bits), (CTemplate.lshr t).evalC a b c = Ptx.shrU a b.toNat
  /- PTX definition: signed `shr` clamps excessive amounts and sign-fills high bits. -/
  /- Lean modeling: both sides interpret `a` at `t.bits` and use the same natural shift amount from `b`. -/
  | .ashr t => ∀ (a b c : BitVec t.bits), (CTemplate.ashr t).evalC a b c = Ptx.shrS a b.toNat
  /- PTX definition: unsigned `div` returns the quotient and leaves division by zero machine-specific. -/
  /- Lean modeling: this obligation compares ordinary quotients and GX's deliberately shared all-ones zero-divisor choice. -/
  | .udiv t => ∀ (a b c : BitVec t.bits), (CTemplate.udiv t).evalC a b c = Ptx.divU a b
  /- PTX definition: signed `div` uses signed quotient semantics and leaves division by zero machine-specific. -/
  /- Lean modeling: the equality covers GX's shared zero and overflow guards plus truncation-toward-zero quotient model. -/
  | .sdiv t => ∀ (a b c : BitVec t.bits), (CTemplate.sdiv t).evalC a b c = Ptx.divS a b
  /- PTX definition: unsigned `rem` returns the remainder and leaves a zero divisor machine-specific. -/
  /- Lean modeling: this obligation compares ordinary remainders and GX's deliberately shared dividend-on-zero choice. -/
  | .urem t => ∀ (a b c : BitVec t.bits), (CTemplate.urem t).evalC a b c = Ptx.remU a b
  /- PTX definition: signed `rem` has machine-dependent negative and zero-divisor behavior. -/
  /- Lean modeling: the equality compares the same `tmod`, dividend-on-zero, and zero-on-minus-one choices. -/
  | .srem t => ∀ (a b c : BitVec t.bits), (CTemplate.srem t).evalC a b c = Ptx.remS a b
  /- PTX definition: unsigned `min` stores the operand smaller in unsigned order. -/
  /- Lean modeling: both operands and the selected result share `BitVec t.bits`, including the equal-operands case. -/
  | .umin t => ∀ (a b c : BitVec t.bits), (CTemplate.umin t).evalC a b c = Ptx.minU a b
  /- PTX definition: unsigned `max` stores the operand greater in unsigned order. -/
  /- Lean modeling: both operands and the selected result share `BitVec t.bits`, including the equal-operands case. -/
  | .umax t => ∀ (a b c : BitVec t.bits), (CTemplate.umax t).evalC a b c = Ptx.maxU a b
  /- PTX definition: signed `min` stores the operand smaller in two's-complement order. -/
  /- Lean modeling: both sides compare `BitVec t.bits` through the same signed `toInt` interpretation. -/
  | .smin t => ∀ (a b c : BitVec t.bits), (CTemplate.smin t).evalC a b c = Ptx.minS a b
  /- PTX definition: signed `max` stores the operand greater in two's-complement order. -/
  /- Lean modeling: both sides compare `BitVec t.bits` through the same signed `toInt` interpretation. -/
  | .smax t => ∀ (a b c : BitVec t.bits), (CTemplate.smax t).evalC a b c = Ptx.maxS a b
  /- PTX definition: `selp` chooses its first data operand for true and its second for false. -/
  /- Lean modeling: both sides use the identical nonzero predicate convention without changing either selected operand. -/
  | .sel    => ∀ {w : Nat} (p a b : BitVec w), CTemplate.sel.evalC p a b = Ptx.selp p a b
  /- PTX definition: integer `setp` applies the selected relation under signed or unsigned interpretation. -/
  /- Lean modeling: `sgn`, `c`, and `t.bits` jointly fix the same comparison relation, interpretation, and operand width. -/
  | .cmp sgn c t => ∀ (a b x : BitVec t.bits),
      (CTemplate.cmp sgn c t).evalC a b x = (if Ptx.setp sgn c a b then 1 else 0)

/- PTX definition: each modeled C scalar shape must implement its corresponding PTX value rule. -/
/- Lean modeling: exhaustive constructor matching specializes the family theorem at `t.bits`, so every `CTemplate` has a proof and adding an unproved constructor fails elaboration. -/
theorem CTemplate.certified : ∀ t : CTemplate, t.spec
  /- PTX definition: scalar `add` stores the fixed-width sum. -/
  /- Lean modeling: `add_equiv t` proves exactly the `.add t` proposition after its operands specialize to `t.bits`. -/
  | .add t  => CTemplate.add_equiv t
  /- PTX definition: packed binary16 add applies the scalar operation independently to both lanes. -/
  /- Lean modeling: the dedicated template theorem inhabits the complete fixed-width constructor obligation. -/
  | .addF16x2 => CTemplate.addF16x2_template_equiv
  /- PTX definition: scalar `sub` stores the fixed-width difference. -/
  /- Lean modeling: `sub_equiv t` directly inhabits the width-specialized subtraction obligation. -/
  | .sub t  => CTemplate.sub_equiv t
  /- PTX definition: `mul.lo` stores the low product bits. -/
  /- Lean modeling: `mul_equiv t` directly proves equality of the indexed multiplication and shallow `Ptx.mulLo` values. -/
  | .mul t  => CTemplate.mul_equiv t
  /- PTX definition: packed bfloat16 multiply applies the scalar operation independently to both lanes. -/
  /- Lean modeling: the dedicated template theorem inhabits the complete fixed-width constructor obligation. -/
  | .mulBF16x2 => CTemplate.mulBF16x2_template_equiv
  /- PTX definition: `mad.lo` adds `c` to the low product. -/
  /- Lean modeling: `mad_equiv t` directly inhabits the width-specialized low multiply-add obligation. -/
  | .mad t  => CTemplate.mad_equiv t
  /- PTX definition: scalar `neg` computes fixed-width two's-complement negation. -/
  /- Lean modeling: `neg_equiv t` directly inhabits the indexed negation proposition. -/
  | .neg t  => CTemplate.neg_equiv t
  /- PTX definition: non-predicate `not` complements every bit. -/
  /- Lean modeling: `bnot_equiv t` directly proves the required fixed-width complement equality. -/
  | .bnot t => CTemplate.bnot_equiv t
  /- PTX definition: non-predicate `and` performs bitwise conjunction. -/
  /- Lean modeling: `band_equiv t` directly proves the required pointwise conjunction equality. -/
  | .band t => CTemplate.band_equiv t
  /- PTX definition: non-predicate `or` performs bitwise disjunction. -/
  /- Lean modeling: `bor_equiv t` directly proves the required pointwise disjunction equality. -/
  | .bor t  => CTemplate.bor_equiv t
  /- PTX definition: non-predicate `xor` performs bitwise exclusive-or. -/
  /- Lean modeling: `bxor_equiv t` directly proves the required pointwise exclusive-or equality. -/
  | .bxor t => CTemplate.bxor_equiv t
  /- PTX definition: `not.pred` complements predicate truth. -/
  /- Lean modeling: the lambda forwards the positive-width premise and operands to the theorem matching `.lnot.spec`. -/
  | .lnot   => fun hw p x y => CTemplate.lnot_equiv hw p x y
  /- PTX definition: `and.pred` conjoins predicate truth. -/
  /- Lean modeling: the lambda forwards the positive-width premise and operands to the theorem matching `.land.spec`. -/
  | .land   => fun hw p q x => CTemplate.land_equiv hw p q x
  /- PTX definition: `or.pred` disjoins predicate truth. -/
  /- Lean modeling: the lambda forwards the positive-width premise and operands to the theorem matching `.lor.spec`. -/
  | .lor    => fun hw p q x => CTemplate.lor_equiv hw p q x
  /- PTX definition: `xor.pred` computes predicate exclusive-or. -/
  /- Lean modeling: the lambda forwards the positive-width premise and operands to the theorem matching `.lxor.spec`. -/
  | .lxor   => fun hw p q x => CTemplate.lxor_equiv hw p q x
  /- PTX definition: `shl` clamps excessive amounts and shifts left with zero fill. -/
  /- Lean modeling: `shl_equiv t` directly inhabits the width-specialized guarded-shift obligation. -/
  | .shl t  => CTemplate.shl_equiv t
  /- PTX definition: unsigned `shr` clamps excessive amounts and zero-fills. -/
  /- Lean modeling: `lshr_equiv t` directly inhabits the width-specialized unsigned-shift obligation. -/
  | .lshr t => CTemplate.lshr_equiv t
  /- PTX definition: signed `shr` clamps excessive amounts and sign-fills. -/
  /- Lean modeling: `ashr_equiv t` directly inhabits the width-specialized signed-shift obligation. -/
  | .ashr t => CTemplate.ashr_equiv t
  /- PTX definition: unsigned `div` computes a quotient with a machine-specific zero case. -/
  /- Lean modeling: `udiv_equiv t` proves exactly GX's shared shallow value policy required by this arm. -/
  | .udiv t => CTemplate.udiv_equiv t
  /- PTX definition: signed `div` computes `d = a / b`, while division by zero is unspecified and machine-specific. -/
  /- Lean modeling: `sdiv_equiv t` proves GX's chosen truncation and zero/`MIN ÷ -1` guards; PTX does not uniquely mandate those policies. -/
  | .sdiv t => CTemplate.sdiv_equiv t
  /- PTX definition: unsigned `rem` computes remainder with a machine-specific zero case. -/
  /- Lean modeling: `urem_equiv t` directly proves GX's shared shallow remainder policy required here. -/
  | .urem t => CTemplate.urem_equiv t
  /- PTX definition: negative signed operands are machine-dependent, while a zero divisor is unspecified and machine-specific. -/
  /- Lean modeling: `srem_equiv t` directly proves the shared guarded `tmod` policy stated by this arm. -/
  | .srem t => CTemplate.srem_equiv t
  /- PTX definition: unsigned `min` selects the smaller unsigned operand. -/
  /- Lean modeling: `umin_equiv t` proves the selector equality at the exact width required by `.umin t`. -/
  | .umin t => CTemplate.umin_equiv t
  /- PTX definition: unsigned `max` selects the greater unsigned operand. -/
  /- Lean modeling: `umax_equiv t` proves the selector equality at the exact width required by `.umax t`. -/
  | .umax t => CTemplate.umax_equiv t
  /- PTX definition: signed `min` selects the smaller two's-complement operand. -/
  /- Lean modeling: `smin_equiv t` proves the signed selector equality at the exact width required by `.smin t`. -/
  | .smin t => CTemplate.smin_equiv t
  /- PTX definition: signed `max` selects the greater two's-complement operand. -/
  /- Lean modeling: `smax_equiv t` proves the signed selector equality at the exact width required by `.smax t`. -/
  | .smax t => CTemplate.smax_equiv t
  /- PTX definition: `selp` chooses between two operands using predicate truth. -/
  /- Lean modeling: `sel_equiv` is width-generic and directly inhabits the complete `.sel` proposition. -/
  | .sel    => CTemplate.sel_equiv
  /- PTX definition: integer `setp` applies a typed relation and returns predicate truth. -/
  /- Lean modeling: passing `sgn`, `c`, and `t` specializes `cmp_equiv` to exactly the constructor's dependent proposition. -/
  | .cmp sgn c t => CTemplate.cmp_equiv sgn c t

end Gxptx
