/-
  Gxptx — canonical library root for the refactored PTX-to-C project.

  The four explicit ownership aggregates separate shallow value semantics,
  machine-checked verification, executable translation, and validation
  metadata.  The historical facades remain imported here so `lake build
  Gxptx` also elaborates the compatibility surface used by downstream code.
-/
import Gxptx.Semantics
import Gxptx.Verification
import Gxptx.Translation
import Gxptx.Validation

import Gxptx.Core
import Gxptx.CSem
import Gxptx.Emit
import Gxptx.Trans
import Gxptx.Module
import Gxptx.Cert
