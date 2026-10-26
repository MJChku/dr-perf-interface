// Ordering litmus: original Lamport's volatile vector load spelling plus an
// explicit release/acquire handshake and system fences between host threads.
#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ unsigned acquire32(const unsigned* p){unsigned x;asm volatile("ld.global.acquire.sys.u32 %0, [%1];":"=r"(x):"l"(p):"memory");return x;}
__device__ __forceinline__ void release32(unsigned* p,unsigned x){asm volatile("st.global.release.sys.u32 [%0], %1;"::"l"(p),"r"(x):"memory");}
extern "C" __global__ void visibility_probe(unsigned* slots,int rank,int rounds,unsigned* result){
 unsigned *mine=slots+rank*32,*peer=slots+(1-rank)*32;unsigned checksum=0;
 for(unsigned round=1;round<=unsigned(rounds);++round){
  while(acquire32(peer+5)<round-1){}
  unsigned a=3*round+rank,b=5*round+rank,c=7*round+rank,d=11*round+rank;
  asm volatile("st.volatile.global.v4.b32 [%0], {%1,%2,%3,%4};"::"l"(mine),"r"(a),"r"(b),"r"(c),"r"(d):"memory");
  asm volatile("membar.sys;":::"memory");release32(mine+4,round);
  while(acquire32(peer+4)<round){}
  asm volatile("membar.sys;":::"memory");
  asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3}, [%4];":"=r"(a),"=r"(b),"=r"(c),"=r"(d):"l"(peer):"memory");
  if(a!=3*round+1-rank||b!=5*round+1-rank||c!=7*round+1-rank||d!=11*round+1-rank){result[rank]=0xffffffff;return;}
  checksum+=a+b+c+d;asm volatile("membar.sys;":::"memory");release32(mine+5,round);
 }
 result[rank]=checksum;
}
