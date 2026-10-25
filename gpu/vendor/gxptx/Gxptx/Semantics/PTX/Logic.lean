/-
  Shallow, width-generic PTX logic and shift semantics.

  The definitions are grouped by ISA family while retaining the declarations
  and behavior formerly housed in `Gxptx.Core`.
-/
import Gxptx.Semantics.Types

namespace Gxptx.Ptx

/- PTX definition: `and` computes the bitwise operation `d = a & b`. -/
/- Lean modeling: `BitVec.and` combines corresponding fixed-width bits exactly as the PTX rule states. -/
def band (a b : BitVec w) : BitVec w := a &&& b

/- PTX definition: `or` computes the bitwise operation `d = a | b`. -/
/- Lean modeling: `BitVec.or` combines corresponding fixed-width bits exactly as the PTX rule states. -/
def bor  (a b : BitVec w) : BitVec w := a ||| b

/- PTX definition: `xor` computes the bitwise operation `d = a ^ b`. -/
/- Lean modeling: `BitVec.xor` combines corresponding fixed-width bits exactly as the PTX rule states. -/
def bxor (a b : BitVec w) : BitVec w := a ^^^ b

/- PTX definition: `not` computes one's-complement bitwise negation `d = ~a`. -/
/- Lean modeling: `BitVec.not` flips every one of the `w` represented bits and introduces no wider bits. -/
def bnot (a : BitVec w) : BitVec w := ~~~a

/- PTX definition: `shl` zero-fills on the right and clamps amounts greater than width `N` to `N`. -/
/- Lean modeling: `BitVec` left shift yields zero once the natural shift amount reaches `w`, matching the PTX clamp result. -/
def shl (a : BitVec w) (n : Nat) : BitVec w := a <<< n

/- PTX definition: unsigned and bit-type `shr` zero-fill and clamp amounts greater than width `N`. -/
/- Lean modeling: logical `BitVec` right shift zero-fills and yields zero for amounts at least `w`. -/
def shrU (a : BitVec w) (n : Nat) : BitVec w := a >>> n

/- PTX definition: signed `shr` fills with the sign bit and clamps amounts greater than width `N`. -/
/- Lean modeling: `BitVec.sshiftRight` performs arithmetic sign-filling and stabilizes at the sign-filled width boundary. -/
def shrS (a : BitVec w) (n : Nat) : BitVec w := a.sshiftRight n

end Gxptx.Ptx
