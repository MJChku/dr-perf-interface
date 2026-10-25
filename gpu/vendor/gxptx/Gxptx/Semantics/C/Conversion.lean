/-
  Shallow value model for the nested C casts emitted for integer `cvt`.
-/
import Gxptx.Semantics.Types

namespace Gxptx.CCvt

/- PTX definition: integer `cvt` preserves the source signed value or unsigned value before conversion to the destination width. -/
/- Lean modeling: `ofInt` and `ofNat` transcribe `(DT)((ST)(x))` by converting the source interpretation and reducing its bit pattern to `dw` bits. -/
def evalC (sgn : Bool) (dw : Nat) {sw : Nat} (a : BitVec sw) : BitVec dw :=
  if sgn then BitVec.ofInt dw a.toInt else BitVec.ofNat dw a.toNat

end Gxptx.CCvt
