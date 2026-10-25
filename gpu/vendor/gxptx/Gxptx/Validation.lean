/-
  Gxptx.Validation — aggregate import for validation metadata and discovery.

  Trusted assumptions and known PTX ISA gaps are intentionally separate data
  inventories.  `TheoremAudit` supplies whole-environment theorem discovery.
  The executable checks and reporting remain in `Gxptx.Check`; keeping that
  command root out of this library facade avoids importing a `main` program.
-/
import Gxptx.Validation.Trusted
import Gxptx.Validation.KnownGaps
import Gxptx.Validation.TheoremAudit
