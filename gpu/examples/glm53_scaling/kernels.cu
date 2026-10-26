// Frozen GLM-5.3 kernels: only linkage/namespace adapted, see sources.json.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cmath>
extern "C" __global__ void materialize_kv_head_major(const float*projected,const float*rotary,
 float*keys,float*values,int batches,int history,int heads,int nope,int rope,int value){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 size_t q=size_t(nope)+rope,kv=size_t(nope)+value,rows=size_t(batches)*heads*history;
 if(i<rows*q){size_t d=i%q,keyrow=i/q,t=keyrow%history,h=(keyrow/history)%heads,b=keyrow/(size_t(history)*heads);
 keys[i]=d<size_t(nope)?projected[((b*history+t)*heads+h)*kv+d]:rotary[(b*history+t)*rope+d-nope];}
 if(i<rows*size_t(value)){size_t d=i%value,keyrow=i/value,t=keyrow%history,h=(keyrow/history)%heads,b=keyrow/(size_t(history)*heads);
 values[i]=projected[((b*history+t)*heads+h)*kv+nope+d];}
}

extern "C" __global__ void sparse_selected_softmax_bf16(const __nv_bfloat16*scores,
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

// Demonstration-only representation change: preserve selected probabilities
// in K slots. Not installed in the serving stack or its dense-PV consumer.
extern "C" __global__ void compact_selected_softmax_bf16(const __nv_bfloat16*scores,
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
 // Compact K-slot output; downstream must consume the same selected-ID layout.
 // The same CTA owns the full output row; no global clear/scatter race.
 __syncthreads();
 for(int slot=threadIdx.x;slot<count;slot+=256){
  int key=selected[slot];
  if(key>=0&&key<history&&key<lengths[batch]&&key<=offsets[batch]+query)
   probabilities[size_t(row)*count+slot]=__float2bfloat16_rn(denominator>0?probabilities_by_slot[slot]/denominator:0.f);
  else probabilities[size_t(row)*count+slot]=__float2bfloat16_rn(0.f);
 }
}
