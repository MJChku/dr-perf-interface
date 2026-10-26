#include "drperf_gpu.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
int main(int argc,char** argv) {
    try {
        if(argc<2)return 2;
        drperf::gpu::Module module(argv[1]);
        std::string mode=argc>2?argv[2]:"lane";
        if(mode=="lane"||mode=="unwrapped") {
            for(int n:{0,1,7,35,53}) {
                std::vector<float> input(n),output(n+2,-123.f);
                for(int i=0;i<n;++i)input[i]=float((i*7+11)%17);
                auto in=input.data(),out=output.data()+1;void* args[]={&in,&out,&n};
                if(mode=="lane") {
                    drperf::gpu::Region r("device-envelope",{{"n",n}});
                    module.launch("device_sections",std::max(1,(n+31)/32),32,args);
                } else module.launch("device_sections",std::max(1,(n+31)/32),32,args);
                require(output.front()==-123.f&&output.back()==-123.f,"lane output guard");
                for(int i=0;i<n;++i)
                    require(out[i]==2*input[i]+int(input[i])%3+1,"lane numerical oracle");
            }
        } else if(mode=="cooperative") {
            for(int width:{16,32,48,64})for(int iterations:{0,1,4}) {
                std::vector<float> output(2*width+2,-123.f);auto out=output.data()+1;
                void* args[]={&out,&iterations};
                module.launch("cooperative_sections",2,width,args);
                require(output.front()==-123.f&&output.back()==-123.f,"cooperative output guard");
                for(int i=0;i<2*width;++i)
                    require(out[i]==2*float((i/width)*width+(i+1)%width+iterations),"cooperative numerical oracle");
            }
        } else if(mode=="repeated") {
            for(int rounds:{1,2,5}) {
                std::vector<float> output(32);auto out=output.data();void* args[]={&out,&rounds};
                module.launch("repeated_sections",1,32,args);
                for(int i=0;i<32;++i)require(out[i]==i+rounds,"repeat numerical oracle");
            }
        } else if(mode=="sparse"||mode=="sparse-refined") {
            int refined=mode=="sparse-refined";
            // Vary mean history nonlinearly with selected, as well as retain
            // repeated-state spread. A fully crossed history/selected grid
            // alone has a constant history mean and legitimately fits it.
            for(int selected:{16,32,64})for(int history:{2048,100000,selected*selected}) {
                std::vector<float> output(history+32+2,-123.f);auto out=output.data()+1;
                void* args[]={&out,&history,&selected,&refined};
                module.launch("sparse_regions",1,32,args);
                require(output.front()==-123.f&&output.back()==-123.f,"sparse output guard");
                for(int i=0;i<history;++i)require(out[i]==0.f,"dense clear numerical oracle");
                for(int i=0;i<32;++i)
                    require(out[history+i]==(i<selected?(selected-1-i)/32+1:0),"selected math numerical oracle");
            }
        } else {
            int invalid=std::stoi(mode);std::vector<float> output(32);auto out=output.data();
            void* args[]={&out,&invalid};module.launch("invalid_sections",1,32,args);
            throw std::runtime_error("invalid marker fixture unexpectedly succeeded");
        }
        std::cout<<"device annotation numerics OK: "<<mode<<'\n';
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
