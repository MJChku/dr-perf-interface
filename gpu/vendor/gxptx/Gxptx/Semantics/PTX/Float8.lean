/-
  Bit-pattern value model for the two observed E4M3 conversion instructions.
  Authority: NVIDIA PTX ISA, cvt and Alternate Floating-Point Data Formats.
  This defines finite precision (not real-number arithmetic). It does not prove
  parser/printer/C compilation correctness. Canonical positive NaN is a permitted
  payload/sign policy, not a claim about a particular GPU's NaN bits.
-/
import Gxptx.Semantics.Types

namespace Gxptx.Ptx

/- PTX definition: E4M3 has bias 7, subnormal quantum 2^-9 and max 448. -/
/- Lean modeling: widening finite values is exact; 0x7f/0xff become quiet half NaN. -/
def e4m3ToF16 (x : BitVec 8) : BitVec 16 :=
  let sign := (x.toNat / 128) * 32768
  let e := (x.toNat / 8) % 16
  let f := x.toNat % 8
  if e = 15 && f = 7 then BitVec.ofNat 16 0x7e00
  else if e ≠ 0 then BitVec.ofNat 16 (sign + (e + 8) * 1024 + f * 128)
  else if f = 0 then BitVec.ofNat 16 sign
  else
    let top := if 4 ≤ f then 2 else if 2 ≤ f then 1 else 0
    BitVec.ofNat 16 (sign + (top + 6) * 1024 + f * 2 ^ (10 - top) - 1024)

/- PTX definition: input byte positions correspond to output half positions. -/
/- Lean modeling: lower and upper lanes are converted independently. -/
def cvtRNF16x2E4M3x2 (x : BitVec 16) : BitVec 32 :=
  let lo := e4m3ToF16 (BitVec.ofNat 8 x.toNat)
  let hi := e4m3ToF16 (BitVec.ofNat 8 (x.toNat / 256))
  BitVec.ofNat 32 (lo.toNat + hi.toNat * 65536)

/- PTX definition: nearest-even conversion preserves sign, saturates magnitude above MAX_NORM, and maps NaN to an E4M3 NaN. -/
/- Lean modeling: rounding operates on exact binary32 significand bits. -/
def f32ToE4M3RNSatfinite (x : BitVec 32) : BitVec 8 :=
  let sign := (x.toNat / 2 ^ 31) * 128
  let mag := x.toNat % 2 ^ 31
  if 0x7f800000 < mag then BitVec.ofNat 8 0x7f
  else if 0x43e00000 ≤ mag then BitVec.ofNat 8 (sign + 0x7e)
  else
    let e := mag / 2 ^ 23
    if e < 117 then BitVec.ofNat 8 sign
    else
      let significand := mag % 2 ^ 23 + 2 ^ 23
      let shift := if e < 121 then 141 - e else 20
      let scale := 2 ^ shift
      let q := significand / scale
      let rem := significand % scale
      let halfway := scale / 2
      let rounded := if halfway < rem || (rem = halfway && q % 2 = 1) then q + 1 else q
      let encoded := if e < 121 then rounded else (e - 121) * 8 + rounded
      BitVec.ofNat 8 (sign + encoded)

/- PTX definition: first .f32 source occupies high byte, second low byte. -/
/- Lean modeling: conversion and byte concatenation remain separate operations. -/
def cvtRNSatfiniteE4M3x2F32 (a b : BitVec 32) : BitVec 16 :=
  BitVec.ofNat 16 ((f32ToE4M3RNSatfinite a).toNat * 256 +
                   (f32ToE4M3RNSatfinite b).toNat)

/- PTX definition: the first f32 operand supplies the high byte and the second supplies the low byte. -/
/- Lean modeling: this structural equality certifies explicit model composition only, not emitted helper C or conversion correctness. -/
theorem cvtRNSatfiniteE4M3x2F32_pack (a b : BitVec 32) :
    cvtRNSatfiniteE4M3x2F32 a b =
      BitVec.ofNat 16 ((f32ToE4M3RNSatfinite a).toNat * 256 +
                       (f32ToE4M3RNSatfinite b).toNat) := rfl

/- PTX definition: each packed input byte widens into the half in the corresponding position. -/
/- Lean modeling: this structural equality certifies model lane composition, not the emitted helper C. -/
theorem cvtRNF16x2E4M3x2_pack (x : BitVec 16) :
    cvtRNF16x2E4M3x2 x = BitVec.ofNat 32 (
      (e4m3ToF16 (BitVec.ofNat 8 x.toNat)).toNat +
      (e4m3ToF16 (BitVec.ofNat 8 (x.toNat / 256))).toNat * 65536) := rfl

/- PTX definition: widening finite E4M3 to binary16 preserves its exact value. -/
/- Lean modeling: magnitudes use integer units of 2^-9; the widening theorem compares with half units of 2^-24 without host floating arithmetic. -/
def e4m3MagnitudeUnits (x : BitVec 8) : Nat :=
  let e := (x.toNat / 8) % 16
  let f := x.toNat % 8
  if e = 0 then f else (8 + f) * 2 ^ (e - 1)

/- PTX definition: binary16 subnormal values are integer multiples of 2^-24. -/
/- Lean modeling: positive binary16 encodings are measured in those units for the finite widening proof. -/
def f16MagnitudeUnitsForFloat8 (x : BitVec 16) : Nat :=
  let e := (x.toNat / 1024) % 32
  let f := x.toNat % 1024
  if e = 0 then f else (1024 + f) * 2 ^ (e - 1)

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
/- PTX definition: widening finite E4M3 values to binary16 preserves their magnitude exactly. -/
/- Lean modeling: finite-domain decide is kernel checked; this theorem proves model values, not C code or the parser. -/
theorem e4m3ToF16_finite_magnitude_exact :
    ∀ x : BitVec 8, x.toNat % 128 ≠ 127 →
      f16MagnitudeUnitsForFloat8 (e4m3ToF16 x) = e4m3MagnitudeUnits x * 32768 := by
  decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
/- PTX definition: widening finite E4M3 values, including signed zero, preserves the sign. -/
/- Lean modeling: finite-domain decide checks every input encoding excluding the two NaNs without native_decide. -/
theorem e4m3ToF16_finite_sign_exact :
    ∀ x : BitVec 8, x.toNat % 128 ≠ 127 →
      (e4m3ToF16 x).toNat / 32768 = x.toNat / 128 := by
  decide

end Gxptx.Ptx
