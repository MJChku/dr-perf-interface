/-
  Gxptx.Cert — compatibility facade for coverage-by-construction proofs.

  Certificate statements and total proofs are separated by responsibility:

  * `Verification.TemplateCertificates` binds each `CTemplate` width and
    proves every scalar template constructor;
  * `Verification.EmissionCertificates` covers conversion, move, multiply,
    memory, and bit-manipulation emission families;
  * `Verification.CommunicationCertificates` covers communication values.

  Existing importers retain every historical `Gxptx.*` declaration name.
  Axiom hygiene and opcode/gap coverage are checked in `Gxptx.Check`.
-/
import Gxptx.Verification.TemplateCertificates
import Gxptx.Verification.EmissionCertificates
import Gxptx.Verification.CommunicationCertificates
