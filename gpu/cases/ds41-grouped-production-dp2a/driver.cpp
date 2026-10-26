#include "drperf_gpu.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv){try{if(argc!=2)return 2;drperf::gpu::Module m(argv[1]);uint32_t result=0xdeadbeef;uintptr_t p=reinterpret_cast<uintptr_t>(&result);void* args[]={&p};{drperf::gpu::Region region("dp2a_coverage",{{"lanes",1}});m.launch("ds41_dp2a_probe",{1},{1},args);}unsigned expected=5+1*1+2*2;if(result!=expected)throw std::runtime_error("independent unsigned low-halfword/low-two-byte dp2a oracle");std::cout<<"{\"numerical_status\":\"pass\",\"expected\":"<<expected<<"}\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
