/-
  Shallow value models for the vector-address expressions and guarded memory
  helpers emitted by GX.
-/
import Gxptx.Semantics.PTX.Memory

namespace Gxptx.CMem

/- PTX definition: vector element `i` is displaced by `i` times the scalar element size. -/
/- Lean modeling: this multiplication is the compile-time byte literal used by the emitted C address expression. -/
def elemOff (i sz : Nat) : Nat := i * sz

/- PTX definition: vector memory elements use consecutive addresses from the base operand. -/
/- Lean modeling: the C model adds the emitted byte literal in 64-bit arithmetic, matching both the printed expression and target address width. -/
def elemAddr (base : BitVec 64) (i sz : Nat) : BitVec 64 :=
  base + BitVec.ofNat 64 (elemOff i sz)

/- PTX definition: PTX loads have no fake-address guard; the threshold check modeled here is GX runtime policy. -/
/- Lean modeling: this is a direct value-level transcription of `gx_gld`, whose unsigned comparison returns zero for fake addresses and reads otherwise. -/
def gld (read : BitVec 64 → BitVec w) (a : BitVec 64) : BitVec w :=
  if Ptx.fakeBase ≤ a then 0 else read a

/- PTX definition: PTX stores have no fake-address guard; the threshold check modeled here is GX runtime policy. -/
/- Lean modeling: this directly transcribes `gx_gst`, preserving memory for fake addresses and updating only the addressed cell otherwise. -/
def gst (mem : BitVec 64 → BitVec w) (a : BitVec 64) (v : BitVec w) :
    BitVec 64 → BitVec w :=
  if Ptx.fakeBase ≤ a then mem else (fun x => if x = a then v else mem x)

end Gxptx.CMem
