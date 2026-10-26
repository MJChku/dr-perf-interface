# GLM-5.3 scaling experiment

Requested: extend standalone GPU DrPerf with supported tcgen05 execution/resource accounting, then demonstrate concrete resource-scaling issues from the GLM study. No serving agents or native GPU jobs are started by this work.

Execution source: canonical GX/NEX cc6dfacb. DrPerf baseline c8ca439f. Preserve unrelated working-tree changes. Port runtime helpers with provenance and test numeric results plus independent resource-count formulas; never count CPU emulator instructions as GPU work.

Cases to investigate: (1) repeated expanded attention history versus selected/latent history, (2) grouped MoE allocation-capacity versus actual token-bounded work. Hold query count, request count, dimensions and output count fixed when varying history. Fit on small states, reject missing PCVs, refine, test held-out states, and distinguish measured resource counts from 2K/100K extrapolation and previously measured native time. Tensor tests use bounded tiles, not production full-model numerical emulation.
