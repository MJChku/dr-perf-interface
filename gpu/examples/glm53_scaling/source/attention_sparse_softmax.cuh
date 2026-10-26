#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
namespace glm53 { namespace eager_attention {
// Source-eager BF16 QK -> BF16 scaling -> FP32 softmax -> BF16 probabilities.
// Scores and probabilities are DISJOINT [B,H,M,T]. IDs [B,M,K] are unique
// nonnegative keys or -1 padding, K<=2048; invalid/causally future IDs ignored.
// Offsets/lengths [B] delimit the current causal context. Finite BF16 scores,
// 0 < scale <= 1 (scaled scores remain finite).
// Grid B*H*M, block256, shared K*sizeof(float). Output includes explicit zeros
// at every unselected history position, so dense PV requires no separate clear.
// All-masked rows produce zero, the existing explicit serving extension.
__global__ void sparse_selected_softmax_bf16(const __nv_bfloat16*scores,
 __nv_bfloat16*probabilities,const int*ids,const int*offsets,const int*lengths,
 int batches,int heads,int queries,int history,int count,float scale){
 int row=blockIdx.x;if(row>=batches*heads*queries)return;
 int batch=row/(heads*queries),query=row%queries;
 const int*selected=ids+(size_t(batch)*queries+query)*count;
 extern __shared__ float probabilities_by_slot[];
 __shared__ float reduction[256];float maximum=-INFINITY;
 for(int slot=threadIdx.x;slot<count;slot+=256){
  int key=selected[slot];float x=-INFINITY;
  if(key>=0&&key<history&&key<lengths[batch]&&key<=offsets[batch]+query)
   x=__bfloat162float(__float2bfloat16_rn(__bfloat162float(scores[size_t(row)*history+key])*scale));
  probabilities_by_slot[slot]=x;maximum=fmaxf(maximum,x);
 }
 reduction[threadIdx.x]=maximum;__syncthreads();
 for(int step=128;step;step/=2){if(threadIdx.x<step)reduction[threadIdx.x]=fmaxf(reduction[threadIdx.x],reduction[threadIdx.x+step]);__syncthreads();}
 maximum=reduction[0];__syncthreads();float sum=0;
 for(int slot=threadIdx.x;slot<count;slot+=256){
  float x=probabilities_by_slot[slot];float p=(maximum==-INFINITY||x==-INFINITY)?0:expf(x-maximum);
  probabilities_by_slot[slot]=p;sum+=p;
 }
 reduction[threadIdx.x]=sum;__syncthreads();
 for(int step=128;step;step/=2){if(threadIdx.x<step)reduction[threadIdx.x]+=reduction[threadIdx.x+step];__syncthreads();}
 float denominator=reduction[0];
 for(int key=threadIdx.x;key<history;key+=256)probabilities[size_t(row)*history+key]=__float2bfloat16_rn(0.f);
 // The same CTA owns the full output row; no global clear/scatter race.
 __syncthreads();
 for(int slot=threadIdx.x;slot<count;slot+=256){
  int key=selected[slot];
  if(key>=0&&key<history&&key<lengths[batch]&&key<=offsets[batch]+query)
   probabilities[size_t(row)*history+key]=__float2bfloat16_rn(denominator>0?probabilities_by_slot[slot]/denominator:0.f);
 }
}
} }
