/-
  Small generated-C expression syntax, its 32-bit evaluator, and the shallow
  C-side arithmetic kernels whose result shapes do not fit `CTemplate`.

  PTX itself remains shallowly embedded as ordinary functions over `BitVec`.
  `CExpr` is a deliberately small deep syntax used only to choose and print C
  expression forms; it is not a deep PTX AST.  Its historical evaluator
  delegates value operations to the PTX functions, so the selector theorems
  establish operator routing, not an independent formal semantics of ISO C.
  The separately reviewable `CTemplate.evalC`/printer boundary and the trusted
  inventory state what remains outside that proof.
-/
import Gxptx.Semantics.Types
import Gxptx.Semantics.PTX.Arithmetic
import Gxptx.Semantics.PTX.Logic
import Gxptx.Semantics.PTX.Comparison

namespace Gxptx

/- PTX definition: result-producing instructions read source operands and write a destination value. -/
/- Lean modeling: this AST records exactly the C expression forms GX selects for the modeled PTX value operators. -/
inductive CExpr
  | var  (n : String)
  | lit  (w : Word)
  | lnot (e : CExpr)                 -- !e
  | bnot (e : CExpr)                 -- ~e
  | land (a b : CExpr)               -- a && b
  | lor  (a b : CExpr)               -- a || b
  | lxor (a b : CExpr)               -- (a) != (b)  (predicate xor)
  | band (a b : CExpr)               -- a & b
  | bor  (a b : CExpr)               -- a | b
  | bxor (a b : CExpr)               -- a ^ b
  | add  (a b : CExpr)               -- a + b
  | sub  (a b : CExpr)               -- a - b
  | mul  (a b : CExpr)               -- a * b
  | shl  (a b : CExpr)               -- a << b
  | lshr (a b : CExpr)               -- logical  >>  (unsigned)
  | ashr (a b : CExpr)               -- arith    >>  (signed)
  | neg  (a : CExpr)                 -- -a
  | mad  (a b c : CExpr)             -- a*b + c
  | udiv (a b : CExpr)               -- unsigned /
  | sdiv (a b : CExpr)               -- signed /
  | urem (a b : CExpr)               -- unsigned %
  | srem (a b : CExpr)               -- signed %
  | umin (a b : CExpr) | umax (a b : CExpr)
  | smin (a b : CExpr) | smax (a b : CExpr)
  | sel  (p a b : CExpr)             -- p ? a : b
  deriving Repr, Inhabited

/- PTX definition: each instruction description defines the destination as a function of its source operands. -/
/- Lean modeling: recursive evaluation resolves variables then delegates to named `Ptx` functions at the existing 32-bit slice; this supports selector proofs but is not an independent ISO-C semantics. -/
def CExpr.eval (env : String → Word) : CExpr → Word
  | .var n    => env n
  | .lit w    => w
  | .lnot e   => if e.eval env = 0 then 1 else 0
  | .bnot e   => Ptx.bnot (e.eval env)
  | .land a b => if a.eval env ≠ 0 ∧ b.eval env ≠ 0 then 1 else 0
  | .lor  a b => if a.eval env ≠ 0 ∨ b.eval env ≠ 0 then 1 else 0
  | .lxor a b => if (a.eval env ≠ 0) ↔ (b.eval env ≠ 0) then 0 else 1
  | .band a b => Ptx.band (a.eval env) (b.eval env)
  | .bor  a b => Ptx.bor  (a.eval env) (b.eval env)
  | .bxor a b => Ptx.bxor (a.eval env) (b.eval env)
  | .add  a b => Ptx.add  (a.eval env) (b.eval env)
  | .sub  a b => Ptx.sub  (a.eval env) (b.eval env)
  | .mul  a b => Ptx.mulLo (a.eval env) (b.eval env)
  | .shl  a b => Ptx.shl  (a.eval env) (b.eval env).toNat
  | .lshr a b => Ptx.shrU (a.eval env) (b.eval env).toNat
  | .ashr a b => Ptx.shrS (a.eval env) (b.eval env).toNat
  | .neg  a   => Ptx.neg  (a.eval env)
  | .mad  a b c => Ptx.madLo (a.eval env) (b.eval env) (c.eval env)
  | .udiv a b => Ptx.divU (a.eval env) (b.eval env)
  | .sdiv a b => Ptx.divS (a.eval env) (b.eval env)
  | .urem a b => Ptx.remU (a.eval env) (b.eval env)
  | .srem a b => Ptx.remS (a.eval env) (b.eval env)
  | .umin a b => Ptx.minU (a.eval env) (b.eval env)
  | .umax a b => Ptx.maxU (a.eval env) (b.eval env)
  | .smin a b => Ptx.minS (a.eval env) (b.eval env)
  | .smax a b => Ptx.maxS (a.eval env) (b.eval env)
  | .sel p a b => Ptx.selp (p.eval env) (a.eval env) (b.eval env)

/- PTX definition: predicate operations distinguish false from true predicate values. -/
/- Lean modeling: PTX exposes Boolean truth but no integer encoding; GX chooses zero for false and nonzero for true on the C-side word boundary. -/
def truthy (w : Word) : Bool := if w = 0 then false else true

/- PTX definition: and, or, xor, and not operate on both bit values and predicates. -/
/- Lean modeling: this tag records the three binary families whose C operator must be chosen from the operand type. -/
inductive BinOp | and | or | xor
  deriving Repr, DecidableEq

/- PTX definition: and/or/xor act on predicates as well as same-width bit operands. -/
/- Lean modeling: predicate tags select C truth operators and all other tags select C bitwise operators, preserving each family's value interpretation. -/
def emitBin : Ty → BinOp → CExpr → CExpr → CExpr
  | .pred, .and, a, b => .land a b
  | .pred, .or,  a, b => .lor  a b
  | .pred, .xor, a, b => .lxor a b
  | _,     .and, a, b => .band a b
  | _,     .or,  a, b => .bor  a b
  | _,     .xor, a, b => .bxor a b

/- PTX definition: `not` complements a bit value or predicate. -/
/- Lean modeling: predicate tags select C logical `!`, while other tags select fixed-width bitwise `~`. -/
def emitNot : Ty → CExpr → CExpr
  | .pred, a => .lnot a
  | _,     a => .bnot a

/- PTX definition: signed `shr` sign-fills, whereas unsigned and bit-type `shr` zero-fill. -/
/- Lean modeling: only `.s32` and `.s64` select the arithmetic-shift node; every other existing tag selects logical shift. -/
def emitShr : Ty → CExpr → CExpr → CExpr
  | .s32, a, b => .ashr a b
  | .s64, a, b => .ashr a b
  | _,    a, b => .lshr a b

/- PTX definition: `div` interprets operands according to its signed or unsigned instruction type. -/
/- Lean modeling: signed tags select the signed-division node and all other existing tags retain the unsigned node selected by the original translator. -/
def emitDiv : Ty → CExpr → CExpr → CExpr
  | .s32, a, b => .sdiv a b
  | .s64, a, b => .sdiv a b
  | _,    a, b => .udiv a b

/- PTX definition: `rem` interprets operands according to its signed or unsigned instruction type. -/
/- Lean modeling: signed tags select the signed-remainder node and all other existing tags retain the unsigned node selected by the original translator. -/
def emitRem : Ty → CExpr → CExpr → CExpr
  | .s32, a, b => .srem a b
  | .s64, a, b => .srem a b
  | _,    a, b => .urem a b

/- PTX definition: signed and unsigned `min` use their respective integer orders. -/
/- Lean modeling: signed tags select the signed-minimum node and all other existing tags retain the unsigned node selected by the original translator. -/
def emitMin : Ty → CExpr → CExpr → CExpr
  | .s32, a, b => .smin a b
  | .s64, a, b => .smin a b
  | _,    a, b => .umin a b

/- PTX definition: signed and unsigned `max` use their respective integer orders. -/
/- Lean modeling: signed tags select the signed-maximum node and all other existing tags retain the unsigned node selected by the original translator. -/
def emitMax : Ty → CExpr → CExpr → CExpr
  | .s32, a, b => .smax a b
  | .s64, a, b => .smax a b
  | _,    a, b => .umax a b

end Gxptx

/-! ## Widening and high-half C arithmetic value models -/

namespace Gxptx.CW

/- PTX definition: signed integer `abs` returns `-a` when `a < 0` and returns `a` otherwise. -/
/- Lean modeling: signed comparison and modular negation mirror the PTX-side shallow model, including GX's explicit minimum-pattern policy. -/
def absS (a : BitVec w) : BitVec w := if a.toInt < 0 then -a else a

/- PTX definition: `abs.s16` applies signed absolute value to a 16-bit operand. -/
/- Lean modeling: this named C-side value model specializes the exact generic signed rule to width 16. -/
def absS16 (a : BitVec 16) : BitVec 16 := absS a

/- PTX definition: `abs.s32` applies signed absolute value to a 32-bit operand. -/
/- Lean modeling: this named C-side value model specializes the exact generic signed rule to width 32. -/
def absS32 (a : BitVec 32) : BitVec 32 := absS a

/- PTX definition: `abs.s64` applies signed absolute value to a 64-bit operand. -/
/- Lean modeling: this named C-side value model specializes the exact generic signed rule to width 64. -/
def absS64 (a : BitVec 64) : BitVec 64 := absS a

/- PTX definition: `mul.wide.uN` requires a full 2N-bit result only for legal N = 16 or 32 forms. -/
/- Lean modeling: zero-extension/multiplication is defined for any `w`; it models a PTX instruction only through the legal U16/U32 specializations. -/
def mulWideU {w : Nat} (a b : BitVec w) : BitVec (2 * w) :=
  (a.setWidth (2 * w)) * (b.setWidth (2 * w))

/- PTX definition: `mul.wide.sN` requires a full 2N-bit result only for legal N = 16 or 32 forms. -/
/- Lean modeling: sign-extension/multiplication is defined for any `w`; it models a PTX instruction only through the legal S16/S32 specializations. -/
def mulWideS {w : Nat} (a b : BitVec w) : BitVec (2 * w) :=
  (a.signExtend (2 * w)) * (b.signExtend (2 * w))

/- PTX definition: `mul.wide.u16` produces the full 32-bit unsigned product. -/
/- Lean modeling: the named C model instantiates zero-extension and multiplication at the legal 16-to-32 width pair. -/
def mulWideU16 (a b : BitVec 16) : BitVec 32 := mulWideU a b

/- PTX definition: `mul.wide.s16` produces the full 32-bit signed product pattern. -/
/- Lean modeling: the named C model instantiates sign-extension and multiplication at the legal 16-to-32 width pair. -/
def mulWideS16 (a b : BitVec 16) : BitVec 32 := mulWideS a b

/- PTX definition: `mul.wide.u32` produces the full 64-bit unsigned product. -/
/- Lean modeling: the named C model instantiates zero-extension and multiplication at the legal 32-to-64 width pair. -/
def mulWideU32 (a b : BitVec 32) : BitVec 64 := mulWideU a b

/- PTX definition: `mul.wide.s32` produces the full 64-bit signed product pattern. -/
/- Lean modeling: the named C model instantiates sign-extension and multiplication at the legal 32-to-64 width pair. -/
def mulWideS32 (a b : BitVec 32) : BitVec 64 := mulWideS a b

/- PTX definition: `mul.hi.uN` selects the high half only for legal N = 16, 32, or 64 forms. -/
/- Lean modeling: the generic logical projection models PTX only through the explicit legal-width specializations. -/
def mulHiU {w : Nat} (a b : BitVec w) : BitVec w :=
  ((mulWideU a b) >>> w).setWidth w

/- PTX definition: `mul.hi.sN` selects the high half only for legal N = 16, 32, or 64 forms. -/
/- Lean modeling: the generic arithmetic projection models PTX only through the explicit legal-width specializations. -/
def mulHiS {w : Nat} (a b : BitVec w) : BitVec w :=
  ((mulWideS a b).sshiftRight w).setWidth w

/- PTX definition: `mul.hi.u16` returns bits 31 through 16 of the unsigned product. -/
/- Lean modeling: the named C model specializes the verified unsigned high-half projection to 16-bit operands. -/
def mulHiU16 (a b : BitVec 16) : BitVec 16 := mulHiU a b

/- PTX definition: `mul.hi.s16` returns bits 31 through 16 of the signed product. -/
/- Lean modeling: the named C model specializes the verified signed high-half projection to 16-bit operands. -/
def mulHiS16 (a b : BitVec 16) : BitVec 16 := mulHiS a b

/- PTX definition: `mul.hi.u32` returns bits 63 through 32 of the unsigned product. -/
/- Lean modeling: the named C model specializes the verified unsigned high-half projection to 32-bit operands. -/
def mulHiU32 (a b : BitVec 32) : BitVec 32 := mulHiU a b

/- PTX definition: `mul.hi.s32` returns bits 63 through 32 of the signed product. -/
/- Lean modeling: the named C model specializes the verified signed high-half projection to 32-bit operands. -/
def mulHiS32 (a b : BitVec 32) : BitVec 32 := mulHiS a b

/- PTX definition: `mul.hi.u64` returns bits 127 through 64 of the unsigned product. -/
/- Lean modeling: the named C model specializes the verified unsigned high-half projection to 64-bit operands. -/
def mulHiU64 (a b : BitVec 64) : BitVec 64 := mulHiU a b

/- PTX definition: `mul.hi.s64` returns bits 127 through 64 of the signed product. -/
/- Lean modeling: the named C model specializes the verified signed high-half projection to 64-bit operands. -/
def mulHiS64 (a b : BitVec 64) : BitVec 64 := mulHiS a b

/- PTX definition: `mad.hi.uN` is an instruction only for legal N = 16, 32, or 64 forms. -/
/- Lean modeling: the generic composition models PTX only at a legal width; printing and dispatcher selection remain separate boundaries. -/
def madHiU {w : Nat} (a b c : BitVec w) : BitVec w := mulHiU a b + c

/- PTX definition: `mad.hi.sN` is an instruction only for legal N = 16, 32, or 64 forms. -/
/- Lean modeling: the generic composition models PTX only at a legal width; printing and dispatcher selection remain separate boundaries. -/
def madHiS {w : Nat} (a b c : BitVec w) : BitVec w := mulHiS a b + c

/- PTX definition: `mad.hi.u16` adds a 16-bit addend to unsigned product bits 31 through 16. -/
/- Lean modeling: the explicit wrapper fixes the generic unsigned high-half model and wrapping addition to 16 bits. -/
def madHiU16 (a b c : BitVec 16) : BitVec 16 := madHiU a b c

/- PTX definition: `mad.hi.s16` adds a 16-bit addend to signed product bits 31 through 16. -/
/- Lean modeling: the explicit wrapper fixes the generic signed high-half model and wrapping addition to 16 bits. -/
def madHiS16 (a b c : BitVec 16) : BitVec 16 := madHiS a b c

/- PTX definition: `mad.hi.u32` adds a 32-bit addend to unsigned product bits 63 through 32. -/
/- Lean modeling: the explicit wrapper fixes the generic unsigned high-half model and wrapping addition to 32 bits. -/
def madHiU32 (a b c : BitVec 32) : BitVec 32 := madHiU a b c

/- PTX definition: `mad.hi.s32` adds a 32-bit addend to signed product bits 63 through 32. -/
/- Lean modeling: the explicit wrapper fixes the generic signed high-half model and wrapping addition to 32 bits. -/
def madHiS32 (a b c : BitVec 32) : BitVec 32 := madHiS a b c

/- PTX definition: `mad.hi.u64` adds a 64-bit addend to unsigned product bits 127 through 64. -/
/- Lean modeling: the explicit wrapper fixes the generic unsigned high-half model and wrapping addition to 64 bits. -/
def madHiU64 (a b c : BitVec 64) : BitVec 64 := madHiU a b c

/- PTX definition: `mad.hi.s64` adds a 64-bit addend to signed product bits 127 through 64. -/
/- Lean modeling: the explicit wrapper fixes the generic signed high-half model and wrapping addition to 64 bits. -/
def madHiS64 (a b c : BitVec 64) : BitVec 64 := madHiS a b c

/- PTX definition: `mad.wide.uN` adds a 2N-bit addend to the complete unsigned product for legal N = 16 or 32. -/
/- Lean modeling: the verified widened product and a same-width unsigned addition give the exact 2N-bit result pattern. -/
def madWideU {w : Nat} (a b : BitVec w) (c : BitVec (2 * w)) : BitVec (2 * w) :=
  mulWideU a b + c

/- PTX definition: `mad.wide.sN` adds a 2N-bit addend to the complete signed product for legal N = 16 or 32. -/
/- Lean modeling: the verified signed widened product and fixed-width addition give the exact 2N-bit two's-complement result pattern. -/
def madWideS {w : Nat} (a b : BitVec w) (c : BitVec (2 * w)) : BitVec (2 * w) :=
  mulWideS a b + c

/- PTX definition: `mad.wide.u16` adds a 32-bit addend to the complete 16-bit unsigned product. -/
/- Lean modeling: the 16-bit sources and 32-bit addend/result enforce the legal widening shape. -/
def madWideU16 (a b : BitVec 16) (c : BitVec 32) : BitVec 32 := madWideU a b c

/- PTX definition: `mad.wide.s16` adds a 32-bit addend to the complete 16-bit signed product. -/
/- Lean modeling: the 16-bit signed sources and 32-bit addend/result enforce the legal widening shape. -/
def madWideS16 (a b : BitVec 16) (c : BitVec 32) : BitVec 32 := madWideS a b c

/- PTX definition: `mad.wide.u32` adds a 64-bit addend to the complete 32-bit unsigned product. -/
/- Lean modeling: the 32-bit sources and 64-bit addend/result enforce the legal widening shape. -/
def madWideU32 (a b : BitVec 32) (c : BitVec 64) : BitVec 64 := madWideU a b c

/- PTX definition: `mad.wide.s32` adds a 64-bit addend to the complete 32-bit signed product. -/
/- Lean modeling: the 32-bit signed sources and 64-bit addend/result enforce the legal widening shape. -/
def madWideS32 (a b : BitVec 32) (c : BitVec 64) : BitVec 64 := madWideS a b c

end Gxptx.CW
