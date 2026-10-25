/-
  Verification facts for the C emission indices.

  The theorem here checks the single-source-of-truth invariant connecting a
  memory element's printed C name to its vector-address stride.  String
  rendering itself remains the explicitly documented trusted boundary.
-/
import Gxptx.Translation.C.Types

namespace Gxptx

/- PTX definition: vector load/store stride equals the byte size of the scalar element type. -/
/- Lean modeling: exhaustive reduction of `MemTy` verifies every emitted C type name is paired with its intended host `sizeof` value. -/
theorem MemTy.cname_bytes_ok : ∀ t : MemTy,
    (t.cname, t.bytes) ∈ [("uint8_t", 1), ("int8_t", 1),
      ("uint16_t", 2), ("int16_t", 2), ("uint32_t", 4), ("int32_t", 4),
      ("uint64_t", 8), ("int64_t", 8), ("float", 4), ("double", 8)] := by
  intro t
  cases t <;> decide

end Gxptx
