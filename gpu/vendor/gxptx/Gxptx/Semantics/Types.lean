/-
  Shared scalar types for the shallow PTX semantics and C-expression model.

  These declarations retain their original `Gxptx.*` names so code importing
  the historical `Gxptx.Core` facade remains source-compatible.
-/
namespace Gxptx

/- PTX definition: `.b32` is a 32-bit untyped fundamental value. -/
/- Lean modeling: `BitVec 32` represents exactly the fixed-width bit pattern carried by a `.b32` register. -/
abbrev Word := BitVec 32

/- PTX definition: PTX fundamental types distinguish predicates, bit types, signed and unsigned integers, and floats. -/
/- Lean modeling: this finite tag set records precisely the scalar families and widths currently routed by GX. -/
inductive Ty | pred | b32 | u32 | s32 | b64 | u64 | s64 | f32
  deriving Repr, DecidableEq

/- PTX definition: signed, unsigned, and untyped bit values are distinct integer-capable scalar categories. -/
/- Lean modeling: the predicate recognizes exactly the six integer/bit tags accepted by GX bitwise emitters. -/
def Ty.isInt : Ty → Bool
  | .b32 | .u32 | .s32 | .b64 | .u64 | .s64 => true
  | _ => false

/- PTX definition: instruction type suffixes determine whether an integer operand is interpreted as signed. -/
/- Lean modeling: only `.s32` and `.s64` require signed C operators in the currently modeled scalar set. -/
def Ty.isSigned : Ty → Bool
  | .s32 | .s64 => true
  | _ => false

/- PTX definition: the `.b64`, `.u64`, and `.s64` fundamental types each have width 64. -/
/- Lean modeling: this predicate centralizes the exact three tags that select 64-bit templates. -/
def Ty.is64 : Ty → Bool
  | .b64 | .u64 | .s64 => true
  | _ => false

/- PTX definition: integer comparisons include eq, ne, lt, le, gt, and ge relations. -/
/- Lean modeling: signedness is carried separately, so one constructor per relational meaning suffices. -/
inductive CmpOp | eq | ne | lt | le | gt | ge
  deriving Repr, DecidableEq

end Gxptx
