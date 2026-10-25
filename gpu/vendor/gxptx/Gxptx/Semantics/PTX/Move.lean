/-
  Shallow PTX semantics for scalar bit moves and brace-vector packing and
  unpacking.
-/
import Gxptx.Semantics.Types

namespace Gxptx.Ptx

/- PTX definition: scalar `mov.bN` copies the source value's N-bit pattern to the destination. -/
/- Lean modeling: identity is the exact value-level meaning of a width-matched bit-pattern copy. -/
def movB {w : Nat} (a : BitVec w) : BitVec w := a

/- PTX definition: brace-source `mov` packs the first half-width element into the low half and the second into the high half. -/
/- Lean modeling: `hi ++ lo` concatenates those two fixed-width patterns in precisely that high-to-low layout. -/
def movPack {h : Nat} (lo hi : BitVec h) : BitVec (h + h) := hi ++ lo

/- PTX definition: brace-destination `mov` writes the packed scalar's low half to its first element. -/
/- Lean modeling: extracting `h` bits from least-significant offset zero returns exactly that first element. -/
def movUnpackLo {h : Nat} (a : BitVec (h + h)) : BitVec h := a.extractLsb' 0 h

/- PTX definition: brace-destination `mov` writes the packed scalar's high half to its second element. -/
/- Lean modeling: extracting `h` bits from least-significant offset `h` returns exactly that second element. -/
def movUnpackHi {h : Nat} (a : BitVec (h + h)) : BitVec h := a.extractLsb' h h

/- PTX definition: four-field mov concatenates equal-width fields first-field-low (b32:4x8 and b64:4x16). -/
/- Lean modeling: nested high/low concatenation states the exact layout without numerical interpretation of floating register storage. -/
def movPack4 {h : Nat} (a b c d : BitVec h) : BitVec ((h + h) + (h + h)) :=
  movPack (movPack a b) (movPack c d)

/- PTX definition: four-field unpack selects field i at low-bit offset i*h, preserving its h-bit encoding. -/
/- Lean modeling: two exact half extractions select the field; Fin 4 excludes nonexistent fields and keeps the selector total. -/
def movUnpack4 {h : Nat} (a : BitVec ((h + h) + (h + h))) (i : Fin 4) : BitVec h :=
  match i.val with
  | 0 => movUnpackLo (movUnpackLo a)
  | 1 => movUnpackHi (movUnpackLo a)
  | 2 => movUnpackLo (movUnpackHi a)
  | _ => movUnpackHi (movUnpackHi a)

end Gxptx.Ptx
