/-
  Shallow value models of the C helpers emitted for GX bit-manipulation
  instructions. These definitions preserve the former `Gxptx.CBits` bodies.
-/
import Gxptx.Semantics.Types

namespace Gxptx.CBits

/- PTX definition: `.wrap` restricts the `shf` amount to the low five bits of `c`. -/
/- Lean modeling: `(c &&& 31#32).toNat` is the value computed by the emitted C expression `(c & 31u)`. -/
def shfAmtWrap (c : BitVec 32) : Nat := (c &&& 31#32).toNat

/- PTX definition: `.clamp` restricts the `shf` amount to at most 32. -/
/- Lean modeling: the conditional is the exact value-level transcription of emitted C `(c > 32u ? 32u : c)`. -/
def shfAmtClamp (c : BitVec 32) : Nat :=
  if 32#32 < c then 32 else c.toNat

/- PTX definition: funnel-right returns the low word of `b:a` shifted right by an amount in 0 through 32. -/
/- Lean modeling: the three cases preserve the exact branch structure and fixed-width operations of `gx_shf_r` in `prelude.h`. -/
def shfR (lo hi : BitVec 32) (n : Nat) : BitVec 32 :=
  if n = 0 then lo
  else if 32 ≤ n then hi >>> (n - 32)
  else (lo >>> n) ||| (hi <<< (32 - n))

/- PTX definition: funnel-left returns the high word of `b:a` shifted left by an amount in 0 through 32. -/
/- Lean modeling: the three cases preserve the exact branch structure and fixed-width operations of `gx_shf_l` in `prelude.h`. -/
def shfL (lo hi : BitVec 32) (n : Nat) : BitVec 32 :=
  if n = 0 then hi
  else if 32 ≤ n then lo <<< (n - 32)
  else (hi <<< n) ||| (lo >>> (32 - n))

/- PTX definition: `bfe` and `bfi` use a low `len`-bit mask when selecting or inserting a field. -/
/- Lean modeling: complementing an all-ones value shifted left by `len` reproduces emitted C `~(~0u << len)` at width `w`. -/
def mask (w len : Nat) : BitVec w := ~~~(BitVec.allOnes w <<< len)

/- PTX definition: `bfi.bN` inserts low source bits into `base` over the position-and-length-selected destination field. -/
/- Lean modeling: this preserves the existing `gx_bfi` value model after its separate low-eight-bit normalization of `pos` and `len`. -/
def bfi (w : Nat) (ins base : BitVec w) (pos len : Nat) : BitVec w :=
  if w ≤ pos then base
  else
    let m := if w ≤ len then BitVec.allOnes w else mask w len
    (base &&& ~~~(m <<< pos)) ||| ((ins &&& m) <<< pos)

/- PTX definition: after control normalization, unsigned `bfe.uN` extracts up to `len` bits starting at `pos` and zero-fills the destination. -/
/- Lean modeling: these branches preserve the width-specific core of `gx_bfe_u32/u64`; public wrappers below apply the required low-eight-bit normalization. -/
def bfeUCore (w : Nat) (a : BitVec w) (pos len : Nat) : BitVec w :=
  if len = 0 then 0
  else if w ≤ pos then 0
  else if w ≤ len then a >>> pos
  else (a >>> pos) &&& mask w len

/- PTX definition: the `.u32` control operands for `bfe` are restricted to their low eight bits before extraction. -/
/- Lean modeling: natural-number remainder mirrors the helper's `& 0xffu`, then delegates to the branch-exact unsigned core. -/
def bfeU (w : Nat) (a : BitVec w) (pos len : Nat) : BitVec w :=
  bfeUCore w a (pos % 256) (len % 256)

/- PTX definition: signed `bfe.sN` copies the available field bits and pads every remaining destination bit with `a[min(pos+len-1,N-1)]`; zero length returns zero. -/
/- Lean modeling: `copied` is the source range still inside the register, the low result uses the unsigned core, and the complemented copied-field mask performs the documented sign padding without host-language shifts. -/
def bfeS (w : Nat) (a : BitVec w) (pos len : Nat) : BitVec w :=
  let p := pos % 256
  let l := len % 256
  if l = 0 then 0
  else
    let copied := min l (w - p)
    let low := bfeUCore w a p copied
    let sbit := a.getLsbD (min (p + l - 1) (w - 1))
    if sbit then low ||| ~~~(mask w copied) else low

/- PTX definition: one generic `prmt` control nibble selects a source byte and optionally sign-replicates it. -/
/- Lean modeling: this definition preserves one unsigned-width iteration of `gx_prmt`, including its byte mask and C truth test. -/
def prmtByte (src : BitVec 64) (s : Nat) : BitVec 32 :=
  let byte : BitVec 32 := (src >>> ((s % 8) * 8)).setWidth 32 &&& 255#32
  if 8 ≤ s % 16 then (if (byte &&& 128#32) ≠ 0#32 then 255#32 else 0#32)
  else byte

/- PTX definition: generic `prmt.b32` assembles four independently selected bytes into `d`. -/
/- Lean modeling: this is the existing four-iteration `gx_prmt` loop unrolled with the same 64-bit source and little-endian shifts. -/
def prmt (a b sel : BitVec 32) : BitVec 32 :=
  prmtByte (a.setWidth 64 ||| (b.setWidth 64 <<< 32)) ((sel >>> 0 &&& 15#32).toNat)
    ||| (prmtByte (a.setWidth 64 ||| (b.setWidth 64 <<< 32)) ((sel >>> 4 &&& 15#32).toNat) <<< 8)
    ||| (prmtByte (a.setWidth 64 ||| (b.setWidth 64 <<< 32)) ((sel >>> 8 &&& 15#32).toNat) <<< 16)
    ||| (prmtByte (a.setWidth 64 ||| (b.setWidth 64 <<< 32)) ((sel >>> 12 &&& 15#32).toNat) <<< 24)

end Gxptx.CBits
