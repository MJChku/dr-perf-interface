/-
  Gxptx.Core — compatibility facade for the theorem-backed modeled core.

  PTX is shallowly embedded: operations are ordinary, width-generic Lean
  functions over `BitVec`.  The generated-C side also contains a small helper
  AST whose evaluator supports operator-selection proofs; that AST is not a
  deep PTX embedding or an independent formalization of ISO C.  Theorems live
  in a separate verification layer. This facade preserves every historical
  `Gxptx.*` and `Gxptx.Ptx.*`
  declaration name for existing importers while exposing the split modules:

  * `Gxptx.Semantics.Types` — shared words, scalar tags, and comparisons;
  * `Gxptx.Semantics.PTX.Arithmetic` — add/sub/neg/div/rem and the grouped
    mul.lo/hi/wide and mad.lo/hi/wide families;
  * `Gxptx.Semantics.PTX.Logic` — and/or/xor/not and shifts;
  * `Gxptx.Semantics.PTX.Comparison` — setp/selp;
  * `Gxptx.Semantics.PTX.Communication` — atom/elect/shfl/copy values;
  * `Gxptx.Semantics.C.Expression` — C AST, evaluator, and emit selectors;
  * `Gxptx.Verification.Operators` — all former Core correctness theorems.

  The compatibility names stay available while reviewed arithmetic semantics,
  routing, and printers live in their family-owned modules.
-/
import Gxptx.Semantics.Types
import Gxptx.Semantics.PTX.Arithmetic
import Gxptx.Semantics.PTX.Logic
import Gxptx.Semantics.PTX.Comparison
import Gxptx.Semantics.PTX.Communication
import Gxptx.Semantics.C.Expression
import Gxptx.Verification.Operators
