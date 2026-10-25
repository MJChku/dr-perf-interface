/-
  Gxptx.CSem — compatibility facade for PTX and C value semantics.

  The former monolithic file mixed PTX definitions, C models, proof helpers,
  and equivalence theorems.  Those declarations now live in family-specific
  `Gxptx.Semantics.{PTX,C}` and `Gxptx.Verification` modules.  This facade
  keeps historical imports and fully-qualified declaration names available
  while reviewed arithmetic definitions and proofs have explicit owners.

  PTX instruction meanings remain shallow embeddings: ordinary Lean
  functions over `BitVec`.  `CTemplate` is a finite index of emitted C shapes,
  not a PTX syntax tree; its value evaluator and its proofs are separated so
  the model, generated syntax, and verification boundary can be reviewed
  independently.
-/
import Gxptx.Semantics.PTX.Arithmetic
import Gxptx.Semantics.C.Expression
import Gxptx.Semantics.C.Template
import Gxptx.Verification.Template
import Gxptx.Verification.Arithmetic

import Gxptx.Semantics.PTX.Conversion
import Gxptx.Semantics.C.Conversion
import Gxptx.Verification.Conversion

import Gxptx.Semantics.PTX.Move
import Gxptx.Semantics.C.Move
import Gxptx.Verification.Move

import Gxptx.Semantics.PTX.Memory
import Gxptx.Semantics.C.Memory
import Gxptx.Verification.Memory

import Gxptx.Semantics.PTX.BitManip
import Gxptx.Semantics.C.BitManip
import Gxptx.Verification.BitManip

import Gxptx.Semantics.C.Communication
import Gxptx.Verification.Communication
