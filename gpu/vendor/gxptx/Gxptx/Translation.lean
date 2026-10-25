/-
  Gxptx.Translation — aggregate import for the executable PTX-to-C pipeline.

  This aggregate imports implementation owners directly so the executable
  translation layer does not depend on verification through compatibility
  facades. `Gxptx.Emit`, `Gxptx.Trans`, and `Gxptx.Module` remain separate
  compatibility entry points for downstream users.
-/
import Gxptx.Translation.C.Types
import Gxptx.Translation.C.Templates
import Gxptx.Translation.C.Expression
import Gxptx.Translation.C.Operands
import Gxptx.Translation.C.Calls
import Gxptx.Translation.Parsing
import Gxptx.Translation.Statement
import Gxptx.Translation.Coverage
import Gxptx.Translation.Body
import Gxptx.Translation.Module.Registers
import Gxptx.Translation.Module.Shared
import Gxptx.Translation.Module.Functions
import Gxptx.Translation.Module.Assembly
