/-
  Shallow value models for the GX C helpers emitted for scalar and packed
  moves.
-/
import Gxptx.Semantics.Types

namespace Gxptx.CMov

/- PTX definition: scalar `mov.bN` preserves the complete N-bit source pattern. -/
/- Lean modeling: width-matched `gx_movb` uses `memcpy`, whose value-level effect on the represented bit pattern is identity. -/
def movB {w : Nat} (a : BitVec w) : BitVec w := a

/- PTX definition: brace-source `mov` places its first element in the scalar's low half and its second in the high half. -/
/- Lean modeling: the two zero-extended halves and shifted high half transcribe `gx_movpack` on the little-endian hosts supported by GX. -/
def pack {h : Nat} (lo hi : BitVec h) : BitVec (h + h) :=
  lo.setWidth (h + h) ||| (hi.setWidth (h + h) <<< h)

/- PTX definition: brace-destination `mov` extracts the packed scalar's low half into its first element. -/
/- Lean modeling: narrowing the packed value to `h` bits transcribes the first half-width `memcpy` performed by `gx_movunpack`. -/
def unpackLo {h : Nat} (a : BitVec (h + h)) : BitVec h := a.setWidth h

/- PTX definition: brace-destination `mov` extracts the packed scalar's high half into its second element. -/
/- Lean modeling: logical shift by `h` followed by narrowing transcribes the second half-width `memcpy` performed by `gx_movunpack`. -/
def unpackHi {h : Nat} (a : BitVec (h + h)) : BitVec h := (a >>> h).setWidth h

/- PTX definition: four equal-width vector fields are packed in increasing significance order, first field low. -/
/- Lean modeling: composing shift/OR two-field models produces the four-field helper layout; C memcpy/printing remains an explicit trusted boundary. -/
def pack4 {h : Nat} (a b c d : BitVec h) : BitVec ((h + h) + (h + h)) :=
  pack (pack a b) (pack c d)

/- PTX definition: destination field i receives the i-th h-bit scalar portion, with optional sinks omitting writes only. -/
/- Lean modeling: nested logical shift-and-narrow models the selected portion; this pure value model does not prove C storage/sink routing. -/
def unpack4 {h : Nat} (a : BitVec ((h + h) + (h + h))) (i : Fin 4) : BitVec h :=
  match i.val with
  | 0 => unpackLo (unpackLo a)
  | 1 => unpackHi (unpackLo a)
  | 2 => unpackLo (unpackHi a)
  | _ => unpackHi (unpackHi a)

end Gxptx.CMov
