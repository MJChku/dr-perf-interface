#include "routed_matvec.cu"
// One CTA owns all K32 contributions of an eight-column tile. Warp-parallel
// MMA preserves each group's FP32 scale order; the final sum follows g=0..G-1.
// This replaces the global partial buffer and its separate finish launch.
extern "C" __global__ __launch_bounds__(256) void ds41_matvec_tc_fused(
    const std::uint8_t* a,const std::uint8_t* b,const std::uint8_t* a_scale,const std::uint8_t* b_scale,
    std::uint16_t* output,unsigned m,unsigned n,unsigned k,unsigned formats,unsigned a_stride,unsigned b_stride,
    unsigned as_stride,unsigned bs_stride,unsigned out_stride,const unsigned* b_pages,unsigned b_page_rows){
    extern __shared__ float partial[];
    const unsigned lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,t=lane%4,row=blockIdx.y,col=blockIdx.x*8+group,groups=k/32;
    const unsigned physical=b_pages&&col<n?b_pages[col/b_page_rows]*b_page_rows+col%b_page_rows:col;
    const bool a4=formats&2,b4=formats&1;const auto ao=static_cast<std::uint64_t>(row)*a_stride,bo=static_cast<std::uint64_t>(physical)*b_stride;
    for(unsigned g=warp;g<groups;g+=blockDim.x/32){
        unsigned a0=0,a2=0,b0=0,b1=0,zero=0;
        if(!group&&row<m){const auto* p=a+ao+g*(a4?16:32)+t*(a4?2:4);a0=load_word(p,a4);a2=load_word(p+(a4?8:16),a4);}
        if(col<n){const auto* p=b+bo+g*(b4?16:32)+t*(b4?2:4);b0=load_word(p,b4);b1=load_word(p+(b4?8:16),b4);}
        float c0=0,c1=0,c2=0,c3=0;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(zero),"r"(a2),"r"(zero),"r"(b0),"r"(b1));
        if(!group&&row<m){const float sa=ue8(a_scale[static_cast<std::uint64_t>(row)*as_stride+g]);
            #pragma unroll
            for(unsigned v=0;v<2;++v){const unsigned column=blockIdx.x*8+t*2+v;if(column<n){const unsigned p=b_pages?b_pages[column/b_page_rows]*b_page_rows+column%b_page_rows:column;const float sb=ue8(b_scale[static_cast<std::uint64_t>(b4?p:p/32)*bs_stride+g]);partial[g*8+t*2+v]=((v?c1:c0)*sa)*sb;}}
        }
    }
    __syncthreads();
    if(threadIdx.x<8&&row<m&&blockIdx.x*8+threadIdx.x<n){float sum=0;for(unsigned g=0;g<groups;++g)sum+=partial[g*8+threadIdx.x];output[static_cast<std::uint64_t>(row)*out_stride+blockIdx.x*8+threadIdx.x]=ds41_float_bf16(sum);}
}
extern "C" __global__ __launch_bounds__(256) void ds41_routed_tc_fused(
    const std::uint8_t* a,const std::uint8_t* a_scale,const std::uint64_t* pairs,const int* ids,std::uint16_t* output,
    unsigned tokens,unsigned n,unsigned k,unsigned a_stride,unsigned as_stride,unsigned b_stride,unsigned bs_stride,
    unsigned id_stride,unsigned first,unsigned owned,unsigned packed,unsigned out_stride){
    extern __shared__ float partial[];
    const unsigned lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,t=lane%4,token=blockIdx.y,col=blockIdx.x*8+group,groups=k/32;
    if(token>=tokens)return;
    #pragma unroll
    for(unsigned route=0;route<6;++route){const int id=ids[static_cast<std::uint64_t>(token)*id_stride+route];const unsigned row=token*6+route;
        if(id<int(first)||id>=int(first+owned)){
            if(threadIdx.x<8&&blockIdx.x*8+threadIdx.x<n)output[static_cast<std::uint64_t>(row)*out_stride+blockIdx.x*8+threadIdx.x]=0;
            continue;
        }
        const unsigned input_row=packed?row:token;const auto* b=reinterpret_cast<const std::uint8_t*>(pairs[2*(id-first)]);const auto* b_scale=reinterpret_cast<const std::uint8_t*>(pairs[2*(id-first)+1]);
        const auto ao=static_cast<std::uint64_t>(input_row)*a_stride,bo=static_cast<std::uint64_t>(col)*b_stride;
        for(unsigned g=warp;g<groups;g+=blockDim.x/32){
            unsigned a0=0,a2=0,b0=0,b1=0,zero=0;
            if(!group){const auto* p=a+ao+g*32+t*4;a0=load_word(p,false);a2=load_word(p+16,false);}
            if(col<n){const auto* p=b+bo+g*16+t*2;b0=load_word(p,true);b1=load_word(p+8,true);}
            float c0=0,c1=0,c2=0,c3=0;
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(zero),"r"(a2),"r"(zero),"r"(b0),"r"(b1));
            if(!group){const float sa=ue8(a_scale[static_cast<std::uint64_t>(input_row)*as_stride+g]);
                #pragma unroll
                for(unsigned v=0;v<2;++v){const unsigned column=blockIdx.x*8+t*2+v;if(column<n){const float sb=ue8(b_scale[static_cast<std::uint64_t>(column)*bs_stride+g]);partial[g*8+t*2+v]=((v?c1:c0)*sa)*sb;}}
            }
        }
        __syncthreads();
        if(threadIdx.x<8&&blockIdx.x*8+threadIdx.x<n){float sum=0;for(unsigned g=0;g<groups;++g)sum+=partial[g*8+threadIdx.x];output[static_cast<std::uint64_t>(row)*out_stride+blockIdx.x*8+threadIdx.x]=ds41_float_bf16(sum);}
        __syncthreads(); // The next local route reuses this CTA-owned array.
    }
}
