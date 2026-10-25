/-
  Shallow value-level address semantics for PTX vector memory operations,
  together with GX's explicitly identified fake-buffer runtime policy.
-/
import Gxptx.Semantics.Types

namespace Gxptx.Ptx

/- PTX definition: vector load/store elements occupy consecutive locations starting at the instruction address. -/
/- Lean modeling: adding the byte displacement `i * sz` in `BitVec 64` represents the selected element address with 64-bit address wraparound. -/
def elemAddr (base : BitVec 64) (i sz : Nat) : BitVec 64 :=
  base + BitVec.ofNat 64 (i * sz)

/- PTX definition: PTX defines memory addresses but no fake-buffer threshold; `fakeBase` is GX runtime policy, not PTX. -/
/- Lean modeling: the fixed 64-bit constant transcribes `GX_FAKE_BASE` so the translator's runtime-only guard can be reasoned about explicitly. -/
def fakeBase : BitVec 64 := 0x0000800000000000#64

/- PTX definition: PTX loads read the addressed value without a fake-address rule; suppressing reads at `fakeBase` is GX runtime policy. -/
/- Lean modeling: the conditional is the abstract guarded-identity contract of `gx_gld`, retaining real reads below the threshold and returning zero otherwise. -/
def gld (read : BitVec 64 → BitVec w) (a : BitVec 64) : BitVec w :=
  if a < fakeBase then read a else 0

/- PTX definition: PTX stores write the addressed value without a fake-address rule; dropping writes at `fakeBase` is GX runtime policy. -/
/- Lean modeling: the conditional updates only the addressed cell below the threshold and leaves the abstract memory unchanged otherwise. -/
def gst (mem : BitVec 64 → BitVec w) (a : BitVec 64) (v : BitVec w) :
    BitVec 64 → BitVec w :=
  if a < fakeBase then (fun x => if x = a then v else mem x) else mem

end Gxptx.Ptx
