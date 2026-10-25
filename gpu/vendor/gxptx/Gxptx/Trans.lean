/-
  Gxptx.Trans — compatibility façade for the PTX-to-C translation pipeline.

  The implementation is organized by responsibility under Gxptx/Translation:
  Parsing recovers PTX operands and types, Statement dispatches one opcode,
  Coverage records proof/trust status, and Body translates complete bodies.
  Existing imports of Gxptx.Trans continue to expose every historical name.
-/
import Gxptx.Translation.Parsing
import Gxptx.Translation.Statement
import Gxptx.Translation.Coverage
import Gxptx.Translation.Body
