/-
  Shallow PTX semantics for the modeled integer-to-integer `cvt` family.
-/
import Gxptx.Semantics.Types

namespace Gxptx.Ptx

/- PTX definition: integer `cvt` truncates when narrowing and extends according to the source type's signedness when widening. -/
/- Lean modeling: `signExtend` and `setWidth` implement those two source-directed bit-pattern conversions for every source and destination width. -/
def cvtI (sgn : Bool) (dw : Nat) {sw : Nat} (a : BitVec sw) : BitVec dw :=
  if sgn then a.signExtend dw else a.setWidth dw

end Gxptx.Ptx
