/-
  Gxptx.Emit — compatibility facade for PTX-to-C emission.

  The former monolithic emitter mixed width/type indices, reviewed string
  templates, recursive C-expression rendering, a verification theorem, and
  the trusted-boundary inventory.  They now have explicit owners:

  * `Gxptx.Translation.C.Types` — EK/PW/IW/MemTy and C type names;
  * `Gxptx.Translation.C.Templates` — all direct instruction-shape printers;
  * `Gxptx.Translation.C.Expression` — OpK/template selection/CExpr traversal;
  * `Gxptx.Verification.Emission` — the memory type/stride theorem;
  * `Gxptx.Validation.Trusted` — the machine-readable trusted inventory.

  Importing `Gxptx.Emit` still exposes the historical fully-qualified names;
  reviewed printer and routing corrections remain owned by the split modules.
-/
-- Preserve the original facade's transitive semantic API for downstream
-- translation modules while the declarations owned here come from the split.
import Gxptx.Core
import Gxptx.CSem
import Gxptx.Translation.C.Types
import Gxptx.Translation.C.Templates
import Gxptx.Translation.C.Expression
import Gxptx.Verification.Emission
import Gxptx.Validation.Trusted
