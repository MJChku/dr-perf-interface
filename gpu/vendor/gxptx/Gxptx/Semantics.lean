/-
  Gxptx.Semantics — aggregate import for the modeled value semantics.

  The PTX side is a shallow embedding: an instruction meaning is an ordinary
  Lean function over `BitVec`, grouped by instruction family.  The C side
  contains corresponding shallow helper functions plus two deliberately small
  finite representations: `CExpr` for operator-selection proofs and
  `CTemplate` for the emitted scalar C shapes.  Proofs are not owned here;
  import `Gxptx.Verification` for theorem and certificate modules.
-/
import Gxptx.Semantics.Types

import Gxptx.Semantics.PTX.Arithmetic
import Gxptx.Semantics.PTX.Logic
import Gxptx.Semantics.PTX.Comparison
import Gxptx.Semantics.PTX.Conversion
import Gxptx.Semantics.PTX.Float8
import Gxptx.Semantics.PTX.Move
import Gxptx.Semantics.PTX.Memory
import Gxptx.Semantics.PTX.BitManip
import Gxptx.Semantics.PTX.Communication

import Gxptx.Semantics.C.Expression
import Gxptx.Semantics.C.Template
import Gxptx.Semantics.C.Conversion
import Gxptx.Semantics.C.Move
import Gxptx.Semantics.C.Memory
import Gxptx.Semantics.C.BitManip
import Gxptx.Semantics.C.Communication
