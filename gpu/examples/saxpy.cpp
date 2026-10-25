#include "drperf_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    drperf::gpu::Module module(argv[1]);
    for(unsigned n:{0u,1u,3u,31u,32u,33u,127u,128u,129u,257u,1025u}) {
        unsigned blocks=std::max(1u,(n+127)/128);
        std::vector<float>x(n+2),y(n+2),out(n+2,-777);
        for(unsigned i=0;i<n;++i){x[i]=float(i%17)-8;y[i]=float(i%13)-6;}
        auto xp=x.data(),yp=y.data(),op=out.data();float alpha=1.25f;
        void* args[]={&xp,&yp,&op,&alpha,&n};
        {
            drperf::gpu::Region region("saxpy",{{"elements",n},{"blocks",blocks},
                                               {"active_blocks",(n+127)/128}});
            module.launch("saxpy",dim3(blocks),dim3(128),args);
        }
        for(unsigned i=0;i<n;++i)
            if(out[i]!=std::fma(alpha,x[i],y[i]))throw std::runtime_error("SAXPY numerical mismatch");
        if(out[n]!=-777||out[n+1]!=-777)throw std::runtime_error("SAXPY guard overwritten");
    }
    puts("GPU SAXPY numerical and guard checks pass");
}
