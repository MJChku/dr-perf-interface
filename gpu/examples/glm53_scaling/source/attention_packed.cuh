#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <limits>
namespace glm53 {
struct PackedKvShape {
 size_t k_count,v_count,kb_count,rope_count;
};
inline PackedKvShape packed_kv_shape(int batches,int history,int heads,int nope,int rope,int value){
 if(batches<1||history<1||heads<1||nope<1||rope<0||value<1)throw std::invalid_argument("packed KV dimensions");
 auto mul=[](size_t a,size_t b){if(b&&a>std::numeric_limits<size_t>::max()/b)throw std::overflow_error("packed KV size");return a*b;};
 size_t rows=mul(size_t(batches),size_t(history)),headed=mul(rows,size_t(heads));
 size_t q=size_t(nope)+size_t(rope),kv=size_t(nope)+size_t(value);
 PackedKvShape s{mul(headed,q),mul(headed,size_t(value)),mul(headed,kv),mul(rows,size_t(rope))};
 for(size_t n:{s.k_count,s.v_count,s.kb_count,s.rope_count})if(n>std::numeric_limits<size_t>::max()/sizeof(float))throw std::overflow_error("packed KV bytes");
 return s;
}
// No aliasing: projected and rotary input live through this launch; outputs use
// existing disjoint k_join/v allocations. Pure loads/stores preserve FP32 bits.
__global__ void materialize_kv_head_major(const float*projected,const float*rotary,
 float*keys,float*values,int batches,int history,int heads,int nope,int rope,int value){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 size_t q=size_t(nope)+rope,kv=size_t(nope)+value,rows=size_t(batches)*heads*history;
 if(i<rows*q){size_t d=i%q,keyrow=i/q,t=keyrow%history,h=(keyrow/history)%heads,b=keyrow/(size_t(history)*heads);
 keys[i]=d<size_t(nope)?projected[((b*history+t)*heads+h)*kv+d]:rotary[(b*history+t)*rope+d-nope];}
 if(i<rows*size_t(value)){size_t d=i%value,keyrow=i/value,t=keyrow%history,h=(keyrow/history)%heads,b=keyrow/(size_t(history)*heads);
 values[i]=projected[((b*history+t)*heads+h)*kv+nope+d];}
}
// Generic transpose for standalone already-materialized inputs; disjoint output.
__global__ void pack_kv_head_major(const float*input,float*output,int batches,int history,int heads,int dim){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(batches)*history*heads*dim)return;
 size_t d=i%dim,row=i/dim,t=row%history,h=(row/history)%heads,b=row/(size_t(history)*heads);
 output[i]=input[((b*history+t)*heads+h)*dim+d];
}

// Isolated lossless head-major K/V consumers. Q/output/selection layouts and
// all arithmetic remain identical; K/V are [batch,head,history,dimension].
__global__ void selected_attention_packed_head_major(const float* q,const float* k,const float* v,
    const int* ids,float* output,const int* offsets,const int* lengths,
    int batches,int sequence,int history,int heads,int query_dim,int value_dim,
    int count,float scaling) {
  int packed=blockIdx.x,b=packed/(heads*sequence),head=(packed/sequence)%heads,s=packed%sequence,token=b*sequence+s,row=token*heads+head;
  if(b>=batches)return;
  extern __shared__ float selected[];
  __shared__ float reduction[128];
  float maximum=-INFINITY;
  for(int slot=threadIdx.x;slot<count;slot+=blockDim.x) {
    int key=ids[token*count+slot];float score=-INFINITY;
    if(key>=0 && key<history && key<lengths[b] && key<=offsets[b]+s) {
      float dot=0;
      for(int d=0;d<query_dim;++d)dot+=q[(token*heads+head)*query_dim+d]*k[((b*heads+head)*history+key)*query_dim+d];
      score=dot*scaling;
    }
    selected[slot]=score;if(score>maximum)maximum=score;
  }
  reduction[threadIdx.x]=maximum;__syncthreads();
  for(int stride=64;stride;stride/=2){if(threadIdx.x<stride && reduction[threadIdx.x+stride]>reduction[threadIdx.x])reduction[threadIdx.x]=reduction[threadIdx.x+stride];__syncthreads();}
  maximum=reduction[0];__syncthreads();float total=0;
  for(int slot=threadIdx.x;slot<count;slot+=blockDim.x) {
    float prob=maximum==-INFINITY?0:expf(selected[slot]-maximum);
    selected[slot]=prob;total+=prob;
  }
  reduction[threadIdx.x]=total;__syncthreads();
  for(int stride=64;stride;stride/=2){if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}
  float denominator=reduction[0];
  for(int d=threadIdx.x;d<value_dim;d+=blockDim.x) {
    float sum=0;
    for(int slot=0;slot<count;++slot)if(selected[slot]>0) {
      int key=ids[token*count+slot];
      sum+=selected[slot]*v[((b*heads+head)*history+key)*value_dim+d];
    }
    output[row*value_dim+d]=denominator>0?sum/denominator:0;
  }
}

__global__ void selected_attention_splitkv_packed_head_major(const float*q,const float*k,const float*v,
 const int*ids,float*partial,const int*offsets,const int*lengths,
 int batches,int sequence,int history,int heads,int query_dim,int value_dim,
 int count,int parts,float scaling){
 int packed=blockIdx.x/parts,part=blockIdx.x%parts,b=packed/(heads*sequence),head=(packed/sequence)%heads,s=packed%sequence,token=b*sequence+s,row=token*heads+head,group=row*parts+part;
 if(b>=batches)return;
 int chunk=(count+parts-1)/parts,begin=part*chunk,end=begin+chunk;if(end>count)end=count;
 int lane=threadIdx.x&31,warp=threadIdx.x/32;
 extern __shared__ float split_probabilities[];
 __shared__ float split_reduction[128];
 for(int slot=begin+warp;slot<end;slot+=4){int key=ids[size_t(token)*count+slot];float dot=0;
  bool valid=key>=0&&key<history&&key<lengths[b]&&key<=offsets[b]+s;
  if(valid)for(int d=lane;d<query_dim;d+=32)dot=fmaf(q[(size_t(token)*heads+head)*query_dim+d],k[((size_t(b)*heads+head)*history+key)*query_dim+d],dot);
  for(int step=16;step;step/=2)dot+=__shfl_down_sync(0xffffffffU,dot,step);
  if(lane==0)split_probabilities[slot-begin]=valid?dot*scaling:-INFINITY;
 }
 __syncthreads();float maximum=-INFINITY;
 for(int slot=begin+threadIdx.x;slot<end;slot+=128)maximum=fmaxf(maximum,split_probabilities[slot-begin]);
 split_reduction[threadIdx.x]=maximum;__syncthreads();
 for(int step=64;step;step/=2){if(threadIdx.x<step)split_reduction[threadIdx.x]=fmaxf(split_reduction[threadIdx.x],split_reduction[threadIdx.x+step]);__syncthreads();}
 maximum=split_reduction[0];__syncthreads();float denominator=0;
 for(int slot=begin+threadIdx.x;slot<end;slot+=128){float p=maximum==-INFINITY?0:expf(split_probabilities[slot-begin]-maximum);split_probabilities[slot-begin]=p;denominator+=p;}
 split_reduction[threadIdx.x]=denominator;__syncthreads();
 for(int step=64;step;step/=2){if(threadIdx.x<step)split_reduction[threadIdx.x]+=split_reduction[threadIdx.x+step];__syncthreads();}
 float*out=partial+size_t(group)*(value_dim+2);
 if(threadIdx.x==0){out[0]=maximum;out[1]=split_reduction[0];}
 for(int d=threadIdx.x;d<value_dim;d+=128){float sum=0;for(int slot=begin;slot<end;++slot){float p=split_probabilities[slot-begin];if(p>0){int key=ids[size_t(token)*count+slot];sum=fmaf(p,v[((size_t(b)*heads+head)*history+key)*value_dim+d],sum);}}out[2+d]=sum;}
}
} // namespace glm53
