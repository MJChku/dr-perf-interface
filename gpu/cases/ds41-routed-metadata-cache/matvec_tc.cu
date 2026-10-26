#include "numeric.cuh"
__device__ inline unsigned load_word(const std::uint8_t* p,bool four){
    if(four){const unsigned nibbles=unsigned(p[0])|(unsigned(p[1])<<8);unsigned out,repeated;
        asm volatile("prmt.b32 %0,%1,%2,%3;":"=r"(out):"r"(0x3c383000u),"r"(0x4c484440u),"r"(nibbles&0x7777u));
        asm volatile("prmt.b32 %0,%1,%2,%3;":"=r"(repeated):"r"(nibbles),"r"(0u),"r"(0x1100u));
        return out|((repeated<<4)&0x00800080u)|(repeated&0x80008000u);
    }
    if((reinterpret_cast<std::uint64_t>(p)&3)==0)return *reinterpret_cast<const unsigned*>(p);
    return unsigned(p[0])|(unsigned(p[1])<<8)|(unsigned(p[2])<<16)|(unsigned(p[3])<<24);
}
__device__ inline float ue8(unsigned v){return v?__uint_as_float(v<<23):0x1p-127f;}
// One warp computes eight columns and four independent K32 contributions.
// Writing individual contributions allows the next kernel to preserve the
// original ordered FP32 scaling/sum instead of regrouping the K reduction.
extern "C" __global__ void ds41_matvec_tc_partials(
    const std::uint8_t* a,const std::uint8_t* b,const std::uint8_t* a_scale,const std::uint8_t* b_scale,
    float* output,unsigned m,unsigned n,unsigned k,unsigned formats,unsigned a_stride,unsigned b_stride,
    unsigned as_stride,unsigned bs_stride,unsigned out_stride,const unsigned* b_pages,unsigned b_page_rows){
    const unsigned lane=threadIdx.x,group=lane/4,t=lane%4,row=blockIdx.y,col=blockIdx.x*8+group;
    const unsigned physical=b_pages&&col<n?b_pages[col/b_page_rows]*b_page_rows+col%b_page_rows:col;
    const bool a4=formats&2,b4=formats&1;const auto ao=static_cast<std::uint64_t>(row)*a_stride,bo=static_cast<std::uint64_t>(physical)*b_stride;
    #pragma unroll
    for(unsigned j=0;j<4;++j){const unsigned g=blockIdx.z*4+j;if(g>=k/32)break;
        unsigned a0=0,a2=0,b0=0,b1=0;
        if(!group&&row<m){const auto* p=a+ao+g*(a4?16:32)+t*(a4?2:4);a0=load_word(p,a4);a2=load_word(p+(a4?8:16),a4);}
        if(col<n){const auto* p=b+bo+g*(b4?16:32)+t*(b4?2:4);b0=load_word(p,b4);b1=load_word(p+(b4?8:16),b4);}
        float c0=0,c1=0,c2=0,c3=0;unsigned zero=0;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(zero),"r"(a2),"r"(zero),"r"(b0),"r"(b1));
        if(!group&&row<m){const float sa=ue8(a_scale[static_cast<std::uint64_t>(row)*as_stride+g]);
            #pragma unroll
            for(unsigned v=0;v<2;++v){const unsigned column=blockIdx.x*8+t*2+v;if(column>=n)continue;
                const unsigned p=b_pages?b_pages[column/b_page_rows]*b_page_rows+column%b_page_rows:column;
                const float sb=ue8(b_scale[static_cast<std::uint64_t>(b4?p:p/32)*bs_stride+g]);
                output[(static_cast<std::uint64_t>(row)*(k/32)+g)*n+column]=((v?c1:c0)*sa)*sb;
            }
        }
    }
}
extern "C" __global__ void ds41_matvec_tc_finish(const float* partial,std::uint16_t* output,unsigned m,unsigned n,unsigned groups,unsigned stride){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m*n)return;const unsigned r=i/n,c=i%n;
    float total=0;const auto* source=partial+static_cast<std::uint64_t>(r)*groups*n+c;
    for(unsigned first=0;first<groups;first+=8){float values[8];
        #pragma unroll
        for(unsigned j=0;j<8;++j)if(first+j<groups)values[j]=source[static_cast<std::uint64_t>(first+j)*n];
        #pragma unroll
        for(unsigned j=0;j<8;++j)if(first+j<groups)total+=values[j];
    }
    output[static_cast<std::uint64_t>(r)*stride+c]=ds41_float_bf16(total);
}
