/-
  Proof that the emitted nested C integer casts implement the shallow PTX
  integer-conversion semantics.
-/
import Gxptx.Semantics.PTX.Conversion
import Gxptx.Semantics.C.Conversion

namespace Gxptx

/- PTX definition: an unsigned integer source is zero-extended when widened and truncated when narrowed. -/
/- Lean modeling: expressing `setWidth` as `ofNat` connects Lean's resize operation directly to the C unsigned-cast transcription. -/
private theorem setWidth_eq_ofNat {v w : Nat} (x : BitVec w) :
    x.setWidth v = BitVec.ofNat v x.toNat :=
  BitVec.eq_of_toNat_eq (by simp)

namespace CCvt

/- PTX definition: integer `cvt` extends by source signedness or truncates to the destination width. -/
/- Lean modeling: case analysis on source signedness reduces the C cast model to either the unsigned resize lemma or the identical signed `ofInt` definition. -/
theorem cvtI_equiv (sgn : Bool) (dw : Nat) {sw : Nat} (a : BitVec sw) :
    evalC sgn dw a = Ptx.cvtI sgn dw a := by
  cases sgn
  · show BitVec.ofNat dw a.toNat = a.setWidth dw
    exact (setWidth_eq_ofNat a).symm
  · rfl

end CCvt
end Gxptx
