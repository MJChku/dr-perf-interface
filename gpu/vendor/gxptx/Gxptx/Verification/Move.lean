/-
  Equivalence and round-trip proofs connecting GX's emitted move helpers to
  the shallow PTX move semantics.
-/
import Gxptx.Semantics.PTX.Move
import Gxptx.Semantics.C.Move

namespace Gxptx.CMov

/- PTX definition: scalar `mov.bN` copies the complete N-bit source pattern. -/
/- Lean modeling: both the C helper model and the PTX shallow semantics are identity at the matched width. -/
theorem movB_equiv {w : Nat} (a : BitVec w) : movB a = Ptx.movB a := rfl

/- PTX definition: brace-source `mov` packs its first element low and its second element high. -/
/- Lean modeling: bit-index extensionality proves the C shift-and-or construction equals the PTX concatenation at every position. -/
theorem pack_equiv {h : Nat} (lo hi : BitVec h) :
    pack lo hi = Ptx.movPack lo hi := by
  apply BitVec.eq_of_getLsbD_eq
  intro ⟨i, hilt⟩
  simp only [pack, Ptx.movPack, BitVec.getLsbD_or, BitVec.getLsbD_setWidth,
    BitVec.getLsbD_shiftLeft, BitVec.getLsbD_append]
  by_cases hlo : i < h
  · have h1 : ¬ h ≤ i := by omega
    simp [hlo, h1, BitVec.getLsbD_ge lo i]
    intro hle
    omega
  · have h1 : ¬ i < h := hlo
    have h2 : i - h < h := by omega
    simp [h1, hilt, BitVec.getLsbD_ge lo i (by omega)]
    omega

/- PTX definition: brace-destination `mov` returns the packed scalar's low half as its first element. -/
/- Lean modeling: equality of natural encodings proves C narrowing and PTX low-bit extraction produce the same `h`-bit value. -/
theorem unpackLo_equiv {h : Nat} (a : BitVec (h + h)) :
    unpackLo a = Ptx.movUnpackLo a := by
  apply BitVec.eq_of_toNat_eq
  simp [unpackLo, Ptx.movUnpackLo, BitVec.extractLsb']

/- PTX definition: brace-destination `mov` returns the packed scalar's high half as its second element. -/
/- Lean modeling: equality of natural encodings proves C shift-and-narrow and PTX high-half extraction produce the same `h`-bit value. -/
theorem unpackHi_equiv {h : Nat} (a : BitVec (h + h)) :
    unpackHi a = Ptx.movUnpackHi a := by
  apply BitVec.eq_of_toNat_eq
  simp [unpackHi, Ptx.movUnpackHi, BitVec.extractLsb']

/- PTX definition: unpacking the first destination element after a brace-source move recovers the original low element. -/
/- Lean modeling: the two equivalence theorems reduce the round trip to low-bit extraction from `hi ++ lo`, which is exactly `lo`. -/
theorem unpack_pack_lo {h : Nat} (lo hi : BitVec h) :
    unpackLo (pack lo hi) = lo := by
  rw [pack_equiv, unpackLo_equiv]
  apply BitVec.eq_of_getLsbD_eq
  intro ⟨i, hilt⟩
  simp [Ptx.movUnpackLo, Ptx.movPack, BitVec.getLsbD_append, hilt]

/- PTX definition: unpacking the second destination element after a brace-source move recovers the original high element. -/
/- Lean modeling: the two equivalence theorems reduce the round trip to high-bit extraction from `hi ++ lo`, which is exactly `hi`. -/
theorem unpack_pack_hi {h : Nat} (lo hi : BitVec h) :
    unpackHi (pack lo hi) = hi := by
  rw [pack_equiv, unpackHi_equiv]
  apply BitVec.eq_of_getLsbD_eq
  intro ⟨i, hilt⟩
  simp only [Ptx.movUnpackHi, Ptx.movPack, BitVec.getLsbD_extractLsb',
    BitVec.getLsbD_append]
  have h1 : ¬ h + i < h := by omega
  have h2 : h + i - h = i := by omega
  simp [hilt, h1, h2]

/- PTX definition: vector mov packs four fields low-to-high in vector order, independently of register names. -/
/- Lean modeling: composing the proved two-field equality proves the complete four-field layout for arbitrary widths and values. -/
theorem pack4_equiv {h : Nat} (a b c d : BitVec h) :
    pack4 a b c d = Ptx.movPack4 a b c d := by
  simp only [pack4, Ptx.movPack4, pack_equiv]

/- PTX definition: vector mov unpack returns the indexed field from a packed scalar, first field least significant. -/
/- Lean modeling: each selector branch composes proved high/low extraction equalities, covering every legal field index. -/
theorem unpack4_equiv {h : Nat} (a : BitVec ((h + h) + (h + h))) (i : Fin 4) :
    unpack4 a i = Ptx.movUnpack4 a i := by
  simp only [unpack4, Ptx.movUnpack4, unpackLo_equiv, unpackHi_equiv]
  rfl

/- PTX definition: unpacking a just-packed four-field vector recovers every original field, including high-bit encodings. -/
/- Lean modeling: two arbitrary-width two-field round trips prove this indexed round trip without testing-only assumptions. -/
theorem unpack4_pack4 {h : Nat} (a b c d : BitVec h) (i : Fin 4) :
    unpack4 (pack4 a b c d) i =
      (match i.val with | 0 => a | 1 => b | 2 => c | _ => d) := by
  simp only [unpack4, pack4, unpack_pack_lo, unpack_pack_hi]
  rfl

end Gxptx.CMov
