/-
  Gxptx.Module — compatibility façade for module-level PTX-to-C translation.

  The implementation is separated under Gxptx/Translation/Module:
  Registers declares C locals; Shared computes shared-memory layouts;
  Functions extracts and emits functions; Assembly builds translation units.
  Existing imports of Gxptx.Module continue to expose every historical name.
-/
import Gxptx.Translation.Module.Registers
import Gxptx.Translation.Module.Shared
import Gxptx.Translation.Module.Functions
import Gxptx.Translation.Module.Assembly
