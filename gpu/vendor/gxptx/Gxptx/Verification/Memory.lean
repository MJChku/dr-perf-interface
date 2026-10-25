/-
  Proofs for vector address arithmetic and for equivalence between GX's two
  presentations of its fake-buffer runtime guard.
-/
import Gxptx.Semantics.PTX.Memory
import Gxptx.Semantics.C.Memory

namespace Gxptx.CMem

/- PTX definition: vector element `i` is addressed at the base plus `i * sizeof(element)`. -/
/- Lean modeling: unfolding `elemOff` makes the emitted C address model definitionally equal to the shallow PTX address function. -/
theorem elemAddr_equiv (base : BitVec 64) (i sz : Nat) :
    elemAddr base i sz = Ptx.elemAddr base i sz := rfl

/- PTX definition: PTX itself has no fake-address guard; this order fact supports GX runtime policy only. -/
/- Lean modeling: converting the `BitVec` inequalities to natural-number inequalities proves that `b ≤ a` excludes `a < b`. -/
private theorem bv_not_lt_of_le {a b : BitVec 64} (h : b ≤ a) : ¬ a < b := by
  rw [BitVec.le_def] at h
  rw [BitVec.lt_def]
  omega

/- PTX definition: PTX itself has no fake-address guard; this complementary order fact supports GX runtime policy only. -/
/- Lean modeling: totality of the underlying natural-number order turns failure of `b ≤ a` into the strict comparison `a < b`. -/
private theorem bv_lt_of_not_le {a b : BitVec 64} (h : ¬ b ≤ a) : a < b := by
  rw [BitVec.le_def] at h
  rw [BitVec.lt_def]
  omega

/- PTX definition: PTX specifies the load itself, while fake-address suppression is GX runtime policy rather than PTX semantics. -/
/- Lean modeling: complementary unsigned-order cases prove the C `fakeBase ≤ a` guard and abstract PTX-side `a < fakeBase` policy choose identical results. -/
theorem gld_equiv (read : BitVec 64 → BitVec w) (a : BitVec 64) :
    gld read a = Ptx.gld read a := by
  simp only [gld, Ptx.gld]
  by_cases h : Ptx.fakeBase ≤ a
  · rw [if_pos h, if_neg (bv_not_lt_of_le h)]
  · rw [if_neg h, if_pos (bv_lt_of_not_le h)]

/- PTX definition: PTX specifies the store itself, while fake-address suppression is GX runtime policy rather than PTX semantics. -/
/- Lean modeling: complementary unsigned-order cases prove the C `fakeBase ≤ a` guard and abstract PTX-side `a < fakeBase` policy choose identical memory functions. -/
theorem gst_equiv (mem : BitVec 64 → BitVec w) (a : BitVec 64) (v : BitVec w) :
    gst mem a v = Ptx.gst mem a v := by
  simp only [gst, Ptx.gst]
  by_cases h : Ptx.fakeBase ≤ a
  · rw [if_pos h, if_neg (bv_not_lt_of_le h)]
  · rw [if_neg h, if_pos (bv_lt_of_not_le h)]

end Gxptx.CMem
