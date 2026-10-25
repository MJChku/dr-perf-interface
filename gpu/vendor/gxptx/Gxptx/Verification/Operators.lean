/-
  Correctness theorems for the type-directed C operator selectors.

  These are all the theorems formerly declared in `Gxptx.Core`; their names,
  statements, and proof terms are preserved exactly while the proofs are now
  visibly separated from PTX and C definitions.
-/
import Gxptx.Semantics.C.Expression

namespace Gxptx

/- PTX definition: predicate false is the false branch used by predicate manipulation. -/
/- Lean modeling: unfolding `truthy` at the zero word reduces definitionally to Boolean false. -/
@[simp] theorem truthy_zero : truthy 0 = false := by decide

/- PTX definition: predicate manipulation distinguishes true from false. -/
/- Lean modeling: the nonzero hypothesis selects the true branch of the C word-to-predicate encoding. -/
@[simp] theorem truthy_ne {w : Word} (h : w ≠ 0) : truthy w = true := by
  simp only [truthy, if_neg h]

/- PTX definition: `not.pred` complements its predicate operand. -/
/- Lean modeling: splitting on zero proves C logical `!` reverses `truthy` for every represented word. -/
theorem emitNot_pred (env) (a : CExpr) :
    truthy ((emitNot .pred a).eval env) = ! truthy (a.eval env) := by
  show truthy (if a.eval env = 0 then (1 : Word) else 0) = ! truthy (a.eval env)
  by_cases h : a.eval env = 0
  · rw [h]; decide
  · rw [if_neg h, truthy_zero, truthy_ne h, Bool.not_true]

/- PTX definition: `and.pred` is true exactly when both predicates are true. -/
/- Lean modeling: exhaustive zero/nonzero cases show the emitted C logical conjunction preserves the two truth values. -/
theorem emitBin_pred_and (env) (a b : CExpr) :
    truthy ((emitBin .pred .and a b).eval env)
      = (truthy (a.eval env) && truthy (b.eval env)) := by
  show truthy (if a.eval env ≠ 0 ∧ b.eval env ≠ 0 then (1 : Word) else 0) = _
  by_cases ha : a.eval env = 0 <;> by_cases hb : b.eval env = 0 <;> simp_all [truthy]

/- PTX definition: `or.pred` is true exactly when either predicate is true. -/
/- Lean modeling: exhaustive zero/nonzero cases show the emitted C logical disjunction preserves the two truth values. -/
theorem emitBin_pred_or (env) (a b : CExpr) :
    truthy ((emitBin .pred .or a b).eval env)
      = (truthy (a.eval env) || truthy (b.eval env)) := by
  show truthy (if a.eval env ≠ 0 ∨ b.eval env ≠ 0 then (1 : Word) else 0) = _
  by_cases ha : a.eval env = 0 <;> by_cases hb : b.eval env = 0 <;> simp_all [truthy]

/- PTX definition: `xor.pred` is true exactly when the predicates differ. -/
/- Lean modeling: exhaustive zero/nonzero cases show the emitted C inequality form preserves Boolean exclusive-or. -/
theorem emitBin_pred_xor (env) (a b : CExpr) :
    truthy ((emitBin .pred .xor a b).eval env)
      = (truthy (a.eval env) ^^ truthy (b.eval env)) := by
  show truthy (if (a.eval env ≠ 0) ↔ (b.eval env ≠ 0) then (0 : Word) else 1) = _
  by_cases ha : a.eval env = 0 <;> by_cases hb : b.eval env = 0 <;> simp_all [truthy]

/- PTX definition: non-predicate `not` performs fixed-width bitwise complement. -/
/- Lean modeling: case analysis on every integer tag reduces the emitted node to `Ptx.bnot`, which is `BitVec.not`. -/
theorem emitNot_int (env) {t : Ty} (ht : t.isInt) (a : CExpr) :
    (emitNot t a).eval env = ~~~ (a.eval env) := by
  cases t <;> simp_all [Ty.isInt, emitNot, CExpr.eval, Ptx.bnot]

/- PTX definition: non-predicate `and` performs fixed-width bitwise conjunction. -/
/- Lean modeling: case analysis on every integer tag reduces the emitted node to `Ptx.band`, which is `BitVec.and`. -/
theorem emitBin_int_and (env) {t : Ty} (ht : t.isInt) (a b : CExpr) :
    (emitBin t .and a b).eval env = (a.eval env) &&& (b.eval env) := by
  cases t <;> simp_all [Ty.isInt, emitBin, CExpr.eval, Ptx.band]

/- PTX definition: `shr.s32` fills vacated high bits with the sign bit. -/
/- Lean modeling: the `.s32` selector is definitionally the arithmetic-shift expression evaluated by signed `BitVec` shift. -/
theorem emitShr_signed (env) (a b : CExpr) :
    (emitShr .s32 a b).eval env = (a.eval env).sshiftRight' (b.eval env) := rfl

/- PTX definition: `shr.s64` fills vacated high bits with the sign bit. -/
/- Lean modeling: the `.s64` selector is definitionally the arithmetic-shift expression retained by the 32-bit C-expression slice. -/
theorem emitShr_signed64 (env) (a b : CExpr) :
    (emitShr .s64 a b).eval env = (a.eval env).sshiftRight' (b.eval env) := rfl

/- PTX definition: `shr.u32` zero-fills vacated high bits. -/
/- Lean modeling: the `.u32` selector is definitionally the logical-right-shift expression evaluated by `BitVec` shift. -/
theorem emitShr_unsigned (env) (a b : CExpr) :
    (emitShr .u32 a b).eval env = (a.eval env) >>> (b.eval env) := rfl

/- PTX definition: `shr.u64` zero-fills vacated high bits. -/
/- Lean modeling: the `.u64` selector is definitionally the logical-right-shift expression retained by the 32-bit C-expression slice. -/
theorem emitShr_unsigned64 (env) (a b : CExpr) :
    (emitShr .u64 a b).eval env = (a.eval env) >>> (b.eval env) := rfl

/- PTX definition: signed and unsigned right shifts differ by sign-fill versus zero-fill. -/
/- Lean modeling: evaluation of the sign-bit-set witness produces distinct `0xC0000000` and `0x40000000` words. -/
theorem shr_signed_ne_unsigned :
    (emitShr .s32 (.lit 0x80000000) (.lit 1)).eval (fun _ => 0)
      ≠ (emitShr .u32 (.lit 0x80000000) (.lit 1)).eval (fun _ => 0) := by
  decide

/- PTX definition: signed `div` interprets operands as signed integers. -/
/- Lean modeling: case analysis on `Ty.isSigned` proves every admitted tag selects the node evaluated by `Ptx.divS`. -/
theorem emitDiv_signed (env) {t : Ty} (ht : t.isSigned) (a b : CExpr) :
    (emitDiv t a b).eval env = Ptx.divS (a.eval env) (b.eval env) := by
  cases t <;> simp_all [Ty.isSigned, emitDiv, CExpr.eval]

/- PTX definition: unsigned `div` interprets operands as unsigned integers. -/
/- Lean modeling: case analysis on integer and nonsigned hypotheses proves every admitted tag selects the node evaluated by `Ptx.divU`. -/
theorem emitDiv_unsigned (env) {t : Ty} (ht : t.isInt ∧ ¬ t.isSigned) (a b : CExpr) :
    (emitDiv t a b).eval env = Ptx.divU (a.eval env) (b.eval env) := by
  cases t <;> simp_all [Ty.isSigned, Ty.isInt, emitDiv, CExpr.eval]

/- PTX definition: signed `rem` uses the signed interpretation selected by its type. -/
/- Lean modeling: case analysis on `Ty.isSigned` proves every admitted tag selects the node evaluated by the existing `Ptx.remS` policy. -/
theorem emitRem_signed (env) {t : Ty} (ht : t.isSigned) (a b : CExpr) :
    (emitRem t a b).eval env = Ptx.remS (a.eval env) (b.eval env) := by
  cases t <;> simp_all [Ty.isSigned, emitRem, CExpr.eval]

/- PTX definition: signed `min` compares operands in signed order. -/
/- Lean modeling: case analysis on `Ty.isSigned` proves every admitted tag selects the node evaluated by `Ptx.minS`. -/
theorem emitMin_signed (env) {t : Ty} (ht : t.isSigned) (a b : CExpr) :
    (emitMin t a b).eval env = Ptx.minS (a.eval env) (b.eval env) := by
  cases t <;> simp_all [Ty.isSigned, emitMin, CExpr.eval]

/- PTX definition: signed `max` compares operands in signed order. -/
/- Lean modeling: case analysis on `Ty.isSigned` proves every admitted tag selects the node evaluated by `Ptx.maxS`. -/
theorem emitMax_signed (env) {t : Ty} (ht : t.isSigned) (a b : CExpr) :
    (emitMax t a b).eval env = Ptx.maxS (a.eval env) (b.eval env) := by
  cases t <;> simp_all [Ty.isSigned, emitMax, CExpr.eval]

/- PTX definition: signed and unsigned division interpret the same bit pattern differently. -/
/- Lean modeling: the concrete `-8` bit pattern divided by two evaluates to distinct signed and unsigned quotients. -/
theorem div_signed_ne_unsigned :
    (emitDiv .s32 (.lit (BitVec.ofInt 32 (-8))) (.lit 2)).eval (fun _ => 0)
      ≠ (emitDiv .u32 (.lit (BitVec.ofInt 32 (-8))) (.lit 2)).eval (fun _ => 0) := by
  decide

end Gxptx
