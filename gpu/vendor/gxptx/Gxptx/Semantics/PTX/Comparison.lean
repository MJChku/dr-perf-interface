/-
  Shallow PTX integer comparison and selection semantics.

  Arithmetic min/max definitions live with the other scalar computational
  instructions; this module contains only predicate comparison and selection.
-/
import Gxptx.Semantics.PTX.Arithmetic

namespace Gxptx.Ptx

/- PTX definition: `selp` stores `a` for a true predicate and `b` otherwise. -/
/- Lean modeling: PTX exposes Boolean truth but no integer encoding; GX chooses zero/nonzero words as its host representation of those truth states. -/
def selp (p a b : BitVec w) : BitVec w := if p ≠ 0 then a else b

/- PTX definition: integer `setp` applies eq, ne, lt, le, gt, or ge with type-directed signedness. -/
/- Lean modeling: equality compares bits, while `toInt` and `toNat` provide exactly the signed and unsigned integer orders. -/
def setp (sgn : Bool) (c : CmpOp) (a b : BitVec w) : Bool :=
  match c with
  | .eq => decide (a = b)
  | .ne => decide (a ≠ b)
  | .lt => if sgn then decide (a.toInt < b.toInt) else decide (a.toNat < b.toNat)
  | .le => if sgn then decide (a.toInt ≤ b.toInt) else decide (a.toNat ≤ b.toNat)
  | .gt => if sgn then decide (b.toInt < a.toInt) else decide (b.toNat < a.toNat)
  | .ge => if sgn then decide (b.toInt ≤ a.toInt) else decide (b.toNat ≤ a.toNat)

end Gxptx.Ptx
