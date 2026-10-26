// out[i] = in[i] + scratch[i]; scratch is written only for i < written.
extern "C" __global__ void partial_write(float* scratch, int written) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < written) scratch[i] = 1.0f;
}
extern "C" __global__ void consume(const float* in, const float* scratch, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] + scratch[i];
}
