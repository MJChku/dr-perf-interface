/-
  Shared indices used by the C translation layer.

  This module contains only the small type/tag definitions and their direct
  mappings to C type names.  It deliberately contains neither C expression
  rendering nor verification theorems, so width selection can be inspected
  independently from the strings and proofs that consume it.
-/
import Gxptx.Semantics.Types
import Gxptx.Semantics.C.Template

namespace Gxptx

/-! ## Expression-emission width -/

/- PTX definition: PTX does not define this emission tag; integer suffixes select scalar width and signedness. -/
/- Lean modeling: `EK` records the integer print width selected by translation or preserves the existing float/legacy branch. -/
inductive EK | int (t : PW) | f
  deriving Repr, DecidableEq

/- PTX definition: PTX does not define C spellings; `.u32`/`.b32` and `.u64`/`.b64` carry 32- and 64-bit patterns. -/
/- Lean modeling: this total two-case table returns the exact C cast spelling used by every unsigned width-indexed template. -/
def PW.uN : PW → String
  | .w32 => "uint32_t"
  | .w64 => "uint64_t"

/- PTX definition: PTX does not define C spellings; `.s32` and `.s64` interpret 32- and 64-bit operands as signed integers. -/
/- Lean modeling: this total two-case table returns the exact C cast spelling used by every signed width-indexed template. -/
def PW.iN : PW → String
  | .w32 => "int32_t"
  | .w64 => "int64_t"

/- PTX definition: comparison conditions are eq, ne, lt, le, gt, and ge in the modeled subset. -/
/- Lean modeling: each constructor maps directly to the corresponding C comparison token used by the comparison template. -/
def CmpOp.c : CmpOp → String
  | .eq => "==" | .ne => "!="
  | .lt => "<"  | .le => "<="
  | .gt => ">"  | .ge => ">="

/-! ## Conversion widths -/

/- PTX definition: integer conversion operands can have widths 8, 16, 32, or 64. -/
/- Lean modeling: `IW` is the finite width index shared by conversion casts and their C integer type spellings. -/
inductive IW | w8 | w16 | w32 | w64
  deriving Repr, DecidableEq

/- PTX definition: the numeric suffix of each integer type is its bit width. -/
/- Lean modeling: this table exposes that suffix as a natural number for width-directed translation. -/
def IW.bits : IW → Nat
  | .w8 => 8 | .w16 => 16 | .w32 => 32 | .w64 => 64

/- PTX definition: unsigned integer conversion types are `.u8`, `.u16`, `.u32`, and `.u64`. -/
/- Lean modeling: each width maps to the fixed-width unsigned C type with the same number of value bits. -/
def IW.uN : IW → String
  | .w8 => "uint8_t" | .w16 => "uint16_t" | .w32 => "uint32_t" | .w64 => "uint64_t"

/- PTX definition: signed integer conversion types are `.s8`, `.s16`, `.s32`, and `.s64`. -/
/- Lean modeling: each width maps to the fixed-width signed C type used to interpret the source or destination bit pattern. -/
def IW.iN : IW → String
  | .w8 => "int8_t" | .w16 => "int16_t" | .w32 => "int32_t" | .w64 => "int64_t"

/-! ## Memory-access types -/

/- PTX definition: modeled scalar loads/stores use signed, unsigned, or floating element types of fixed widths. -/
/- Lean modeling: one tag determines both the emitted C access type and vector element stride, preventing those choices from diverging. -/
inductive MemTy | u8 | s8 | u16 | s16 | u32 | s32 | u64 | s64 | f32 | f64
  deriving Repr, DecidableEq

/- PTX definition: adjacent vector elements are separated by the byte size of the scalar element type. -/
/- Lean modeling: each tag maps to its exact scalar byte width and is reused as the emitted vector stride. -/
def MemTy.bytes : MemTy → Nat
  | .u8 => 1 | .s8 => 1 | .u16 => 2 | .s16 => 2
  | .u32 => 4 | .s32 => 4 | .u64 => 8 | .s64 => 8
  | .f32 => 4 | .f64 => 8

/- PTX definition: load/store signedness and width determine the scalar value interpretation. -/
/- Lean modeling: fixed-width C integer names preserve that width/signedness, while `float` and `double` preserve the existing f32/f64 access spellings. -/
def MemTy.cname : MemTy → String
  | .u8 => "uint8_t"  | .s8 => "int8_t"
  | .u16 => "uint16_t" | .s16 => "int16_t"
  | .u32 => "uint32_t" | .s32 => "int32_t"
  | .u64 => "uint64_t" | .s64 => "int64_t"
  | .f32 => "float"    | .f64 => "double"

end Gxptx
