/-
  Gxptx.Verification — aggregate import for machine-checked equivalence proofs.

  Family modules prove agreement between the shallow PTX meanings and the
  corresponding C-side models.  Certificate modules turn those local results
  into total obligations over every finite template index.  Importing this
  facade elaborates all production verification declarations, but the runtime
  environment/coverage report remains the separate `gxcheck` executable.
-/
import Gxptx.Verification.Operators
import Gxptx.Verification.Arithmetic
import Gxptx.Verification.Template
import Gxptx.Verification.Conversion
-- This finite-domain model currently owns its widening/lane theorems. Keep
-- them in the translator's proof dependency closure as well as gxcheck's.
import Gxptx.Semantics.PTX.Float8
import Gxptx.Verification.Move
import Gxptx.Verification.Memory
import Gxptx.Verification.BitManip
import Gxptx.Verification.Communication
import Gxptx.Verification.Emission

import Gxptx.Verification.TemplateCertificates
import Gxptx.Verification.EmissionCertificates
import Gxptx.Verification.CommunicationCertificates
