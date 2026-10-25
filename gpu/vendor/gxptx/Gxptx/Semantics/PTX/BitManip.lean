/-
  Shallow PTX value semantics for the bit-manipulation instructions modeled by
  GX. Each definition is a direct function over fixed-width values.
-/
import Gxptx.Semantics.Types

namespace Gxptx.Ptx

/- PTX definition: `shf.r.{wrap,clamp}.b32` right-shifts the concatenation `b:a` and returns its low 32 bits. -/
/- Lean modeling: concatenation, logical shift, and low-bit extraction directly transcribe the PTX funnel-right pseudocode after amount selection. -/
def shfR (a b : BitVec 32) (n : Nat) : BitVec 32 :=
  ((b ++ a) >>> n).extractLsb' 0 32

/- PTX definition: `shf.l.{wrap,clamp}.b32` left-shifts the concatenation `b:a` and returns its high 32 bits. -/
/- Lean modeling: concatenation, logical shift, and extraction at bit 32 directly transcribe the PTX funnel-left pseudocode after amount selection. -/
def shfL (a b : BitVec 32) (n : Nat) : BitVec 32 :=
  ((b ++ a) <<< n).extractLsb' 32 32

/- PTX definition: `.wrap` uses `c & 31`, while `.clamp` uses `min(c, 32)` as the funnel-shift amount. -/
/- Lean modeling: modulo 32 and natural-number minimum compute exactly the two amount rules on the unsigned `.b32` control value. -/
def shfAmt (wrap : Bool) (c : BitVec 32) : Nat :=
  if wrap then c.toNat % 32 else min c.toNat 32

/- PTX definition: generic `prmt.b32` selects one byte from `b:a` and optionally replicates that byte's sign bit. -/
/- Lean modeling: low selector bits choose an eight-byte source position, while selector bit 3 chooses literal-byte versus sign-replicated output. -/
def prmtSel (src : BitVec 64) (ctl : Nat) : BitVec 8 :=
  let byte := src.extractLsb' ((ctl % 8) * 8) 8
  if 8 ≤ ctl % 16 then (if byte.getLsbD 7 then BitVec.allOnes 8 else 0#8)
  else byte

/- PTX definition: generic `prmt.b32 d, a, b, c` uses each control nibble of `c` to construct one destination byte. -/
/- Lean modeling: four calls to `prmtSel`, concatenated from destination byte 3 down to byte 0, directly assemble the specified 32-bit result. -/
def prmt (a b c : BitVec 32) : BitVec 32 :=
  prmtSel (b ++ a) ((c >>> 12).toNat % 16)
    ++ prmtSel (b ++ a) ((c >>> 8).toNat % 16)
    ++ prmtSel (b ++ a) ((c >>> 4).toNat % 16)
    ++ prmtSel (b ++ a) ((c >>> 0).toNat % 16)

end Gxptx.Ptx
