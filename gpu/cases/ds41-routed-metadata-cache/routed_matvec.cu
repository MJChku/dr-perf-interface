#include "matvec_tc.cu"
// A CTA visits the six routes of one token, skipping nonlocal experts before
// touching their pointer table. No host route readback or per-expert launch.
extern "C" __global__ void ds41_routed_tc_partials(
    const std::uint8_t* a,const std::uint8_t* a_scale,const std::uint64_t* pairs,const int* ids,float* output,
    unsigned tokens,unsigned n,unsigned k,unsigned a_stride,unsigned as_stride,unsigned b_stride,unsigned bs_stride,
    unsigned id_stride,unsigned first,unsigned owned,unsigned packed){
    const unsigned lane=threadIdx.x,group=lane/4,t=lane%4,token=blockIdx.y,col=blockIdx.x*8+group;
    if(token>=tokens)return;
    #pragma unroll
    for(unsigned route=0;route<6;++route){const int id=ids[static_cast<std::uint64_t>(token)*id_stride+route];if(id<int(first)||id>=int(first+owned))continue;
        const unsigned local=id-first,row=token*6+route,input_row=packed?row:token;
        const auto* b=reinterpret_cast<const std::uint8_t*>(pairs[2*local]);const auto* b_scale=reinterpret_cast<const std::uint8_t*>(pairs[2*local+1]);
        const auto ao=static_cast<std::uint64_t>(input_row)*a_stride,bo=static_cast<std::uint64_t>(col)*b_stride;
        #pragma unroll
        for(unsigned j=0;j<4;++j){const unsigned g=blockIdx.z*4+j;if(g>=k/32)break;
            unsigned a0=0,a2=0,b0=0,b1=0,zero=0;
            if(!group){const auto* p=a+ao+g*32+t*4;a0=load_word(p,false);a2=load_word(p+16,false);}
            if(col<n){const auto* p=b+bo+g*16+t*2;b0=load_word(p,true);b1=load_word(p+8,true);}
            float c0=0,c1=0,c2=0,c3=0;
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(zero),"r"(a2),"r"(zero),"r"(b0),"r"(b1));
            if(!group){const float sa=ue8(a_scale[static_cast<std::uint64_t>(input_row)*as_stride+g]);
                #pragma unroll
                for(unsigned v=0;v<2;++v){const unsigned column=blockIdx.x*8+t*2+v;if(column<n){const float sb=ue8(b_scale[static_cast<std::uint64_t>(column)*bs_stride+g]);output[(static_cast<std::uint64_t>(row)*(k/32)+g)*n+column]=((v?c1:c0)*sa)*sb;}}
            }
        }
    }
}
extern "C" __global__ void ds41_routed_tc_finish(const float* partial,const int* ids,std::uint16_t* output,
    unsigned tokens,unsigned n,unsigned groups,unsigned stride,unsigned id_stride,unsigned first,unsigned owned){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tokens*6*n)return;const unsigned r=i/n,c=i%n;
    const int id=ids[static_cast<std::uint64_t>(r/6)*id_stride+r%6];float total=0;
    if(id>=int(first)&&id<int(first+owned)){const auto* source=partial+static_cast<std::uint64_t>(r)*groups*n+c;
        for(unsigned first_group=0;first_group<groups;first_group+=8){float values[8];
            #pragma unroll
            for(unsigned j=0;j<8;++j)if(first_group+j<groups)values[j]=source[static_cast<std::uint64_t>(first_group+j)*n];
            #pragma unroll
            for(unsigned j=0;j<8;++j)if(first_group+j<groups)total+=values[j];
        }
    }
    output[static_cast<std::uint64_t>(r)*stride+c]=ds41_float_bf16(total);
}
extern "C" __global__ void ds41_route_token_ids(const int* ids,int* tokens,int* errors,unsigned rows,unsigned stride,unsigned first,unsigned owned){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*6)return;const int id=ids[static_cast<std::uint64_t>(i/6)*stride+i%6];
    if(id<0||id>=384)atomicExch(errors,1);
    tokens[i]=id>=int(first)&&id<int(first+owned)?int(i/6):-1;
}
